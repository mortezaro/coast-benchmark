import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug[:max_len] or "item"


def audio_content(path: Path) -> Dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def grounding_metadata_text(meta: Dict[str, str]) -> str:
    bullets = [
        "Wearable egocentric audio from the smart-glasses wearer's perspective",
        f"Environment: {meta.get('environment', 'unknown').lower()}",
        f"Noise condition: {meta.get('noise_level', 'unknown').lower()}",
        f"Dominant background or competing sound: {meta.get('_noise_type', 'unknown')}",
        f"Physical location type: {meta.get('location', 'unknown')}",
    ]
    if meta.get("partner_position") and meta.get("partner_position") != "N/A":
        bullets.append(f"Nearby speech/source angle relative to wearer: {meta['partner_position']} degrees")
    if meta.get("partner_distance") and meta.get("partner_distance") != "N/A":
        bullets.append(f"Nearby speech/source distance: approximately {meta['partner_distance']}")
    if meta.get("bystander__y_n_") == "Yes":
        bullets.append("Bystander speech is present; assistant-directedness may be ambiguous")
    else:
        bullets.append("No explicit bystander flag, but environmental masking may still affect reliability")
    return "\n".join(f"- {bullet}" for bullet in bullets)


def task_question(row: Dict[str, Any]) -> str:
    task = row["task"]
    if task == "non-assistant-directed":
        return (
            "You are a smart-glasses assistant. Decide whether the speech is directed at you. "
            "If it is side-talk or directed at another person, stay silent. If it is directed at you, answer briefly."
        )
    if task == "tool_call":
        return (
            "You are a smart-glasses assistant. Decide what action to take from the spoken request. "
            "If the request is unclear or unsafe in the physical situation, ask for a short clarification."
        )
    return (
        "You are a smart-glasses assistant. Answer the spoken question briefly. "
        "If the physical audio situation makes the answer unreliable, ask the wearer to repeat or clarify."
    )


def prompt_text(row: Dict[str, Any], *, with_metadata: bool) -> str:
    metadata = ""
    if with_metadata:
        metadata = f"\nPhysical grounding metadata:\n{grounding_metadata_text(row['audio_metadata'])}\n"
    return f"""Listen to the real wearable audio clip and respond directly as the assistant would speak to the wearer.

{metadata}
Task:
{task_question(row)}

Speak only the assistant's final response, not an explanation. Keep it natural and short, usually one sentence.
Do not say "with metadata", "without metadata", "model", or "grounding" in the spoken answer."""


def call_audio_model(
    *,
    api_key: str,
    model: str,
    audio_path: Path,
    prompt: str,
    voice: str,
    fmt: str,
    max_retries: int = 6,
) -> Tuple[bytes, str, Dict[str, Any]]:
    payload = {
        "model": model,
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": fmt},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise smart-glasses voice assistant. "
                    "Respond with only the words the assistant should say aloud."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Audio clip:"},
                    audio_content(audio_path),
                ],
            },
        ],
        "temperature": 0.2,
        "max_completion_tokens": 160,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(max_retries):
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                message = parsed["choices"][0]["message"]
                audio = message.get("audio") or {}
                audio_b64 = audio.get("data")
                transcript = audio.get("transcript") or message.get("content") or ""
                if not audio_b64:
                    raise RuntimeError(f"No audio data in response: {raw[:1000]}")
                return base64.b64decode(audio_b64), transcript, parsed.get("usage", {})
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {error_body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(last_error)
        except Exception as exc:  # pragma: no cover - network failures vary.
            last_error = repr(exc)
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"audio model failed after retries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-audio")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3", "flac", "opus", "pcm16"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")

    rows = load_jsonl(args.input_root / "wearvox_grounding_demo_results.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    out_root = args.input_root / "model_audio_outputs"
    manifest = []
    for index, row in enumerate(rows, start=1):
        audio_path = args.input_root / row["local_audio_file"]
        slug = f"{index:02d}_{safe_slug(row['demo_category'])}_{safe_slug(row['id'], 42)}"
        item = {
            "index": index,
            "id": row["id"],
            "task": row["task"],
            "demo_category": row["demo_category"],
            "input_audio": str(audio_path),
        }
        for condition, with_meta in [("without_grounding", False), ("with_grounding", True)]:
            output_path = out_root / condition / f"{slug}.{args.format}"
            transcript_path = out_root / condition / f"{slug}.txt"
            if output_path.exists() and output_path.stat().st_size > 0 and transcript_path.exists():
                transcript = transcript_path.read_text().strip()
            else:
                audio_bytes, transcript, usage = call_audio_model(
                    api_key=api_key,
                    model=args.model,
                    audio_path=audio_path,
                    prompt=prompt_text(row, with_metadata=with_meta),
                    voice=args.voice,
                    fmt=args.format,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio_bytes)
                transcript_path.write_text(transcript.strip() + "\n")
                (out_root / "usage.jsonl").parent.mkdir(parents=True, exist_ok=True)
                with (out_root / "usage.jsonl").open("a") as handle:
                    handle.write(json.dumps({"id": row["id"], "condition": condition, "usage": usage}) + "\n")
            item[f"{condition}_audio"] = str(output_path)
            item[f"{condition}_transcript"] = transcript
        manifest.append(item)
        print(f"[{index}/{len(rows)}] generated model audio for {row['id']}", flush=True)

    (out_root / "model_audio_outputs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_root / "README.md").write_text(
        "# WearVox Audio-to-Audio Investor Demo\n\n"
        "This folder contains direct spoken outputs from the audio model, not text-to-speech readings of a report.\n\n"
        "- `../audio/`: original WearVox input prompt clips\n"
        "- `without_grounding/`: assistant audio responses using the input clip only\n"
        "- `with_grounding/`: assistant audio responses using the input clip plus physical grounding metadata\n"
        "- `model_audio_outputs_manifest.json`: pairs each input clip with both output clips and transcripts\n\n"
        "For each example, play the input audio first, then the matching `without_grounding` output, "
        "then the matching `with_grounding` output.\n"
    )
    print(f"Wrote {out_root}")


if __name__ == "__main__":
    main()
