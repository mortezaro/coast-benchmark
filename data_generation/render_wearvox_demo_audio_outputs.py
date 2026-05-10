import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "decision": "unknown",
            "answer_or_action": None,
            "grounding_reason": cleaned,
            "risk_or_confidence": "",
        }
    return parsed if isinstance(parsed, dict) else {"grounding_reason": str(parsed)}


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug[:max_len] or "item"


def compact_text(value: Any) -> str:
    if value is None:
        return "None."
    text = str(value).strip()
    return text if text else "None."


def narration_text(row: Dict[str, Any], variant: str) -> str:
    parsed = parse_response(row[f"{variant}_metadata_response"])
    label = "without physical grounding metadata" if variant == "without" else "with physical grounding metadata"
    decision = compact_text(parsed.get("decision"))
    answer = compact_text(parsed.get("answer_or_action"))
    reason = compact_text(parsed.get("grounding_reason"))
    risk = compact_text(parsed.get("risk_or_confidence"))
    return (
        f"Model answer {label}. "
        f"Decision: {decision}. "
        f"Answer or action: {answer}. "
        f"Reason: {reason}. "
        f"Confidence or risk: {risk}"
    )


def call_speech_api(
    *,
    api_key: str,
    text: str,
    output_path: Path,
    model: str,
    voice: str,
    instructions: str,
    max_retries: int = 6,
) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "voice": voice,
        "input": text,
        "format": output_path.suffix.lstrip(".") or "wav",
        "instructions": instructions,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(max_retries):
        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                output_path.write_bytes(response.read())
                return
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {error_body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(last_error)
        except Exception as exc:  # pragma: no cover - network errors vary by runtime.
            last_error = repr(exc)
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"speech API failed after retries: {last_error}")


def write_manifest(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    lines = [
        "index,id,category,input_audio,without_metadata_audio,with_metadata_audio,without_metadata_text,with_metadata_text"
    ]
    for row in rows:
        values = [
            str(row["index"]),
            row["id"],
            row["demo_category"],
            row["input_audio"],
            row["without_metadata_audio"],
            row["with_metadata_audio"],
            row["without_metadata_text"].replace('"', '""').replace("\n", " "),
            row["with_metadata_text"].replace('"', '""').replace("\n", " "),
        ]
        quoted = [f'"{value}"' for value in values]
        lines.append(",".join(quoted))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-4o-mini-tts")
    parser.add_argument("--voice", default="verse")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--format", default="wav", choices=["wav", "mp3", "opus", "aac", "flac", "pcm"])
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")

    results_path = args.input_root / "wearvox_grounding_demo_results.jsonl"
    rows = load_jsonl(results_path)
    if args.limit:
        rows = rows[: args.limit]

    out_root = args.input_root / "audio_outputs"
    manifest_rows = []
    instructions = (
        "Speak clearly and neutrally for an investor demo. Keep the tone polished, concise, "
        "and analytic. This is an AI-generated narration of a model response."
    )
    for index, row in enumerate(rows, start=1):
        slug = f"{index:02d}_{safe_slug(row['demo_category'])}_{safe_slug(row['id'], 42)}"
        without_text = narration_text(row, "without")
        with_text = narration_text(row, "with")
        without_path = out_root / "without_metadata" / f"{slug}.{args.format}"
        with_path = out_root / "with_metadata" / f"{slug}.{args.format}"
        call_speech_api(
            api_key=api_key,
            text=without_text,
            output_path=without_path,
            model=args.model,
            voice=args.voice,
            instructions=instructions,
        )
        call_speech_api(
            api_key=api_key,
            text=with_text,
            output_path=with_path,
            model=args.model,
            voice=args.voice,
            instructions=instructions,
        )
        manifest_rows.append(
            {
                "index": index,
                "id": row["id"],
                "demo_category": row["demo_category"],
                "input_audio": str(args.input_root / row["local_audio_file"]),
                "without_metadata_audio": str(without_path),
                "with_metadata_audio": str(with_path),
                "without_metadata_text": without_text,
                "with_metadata_text": with_text,
            }
        )
        print(f"[{index}/{len(rows)}] rendered {row['id']}", flush=True)

    write_manifest(manifest_rows, out_root / "audio_outputs_manifest.csv")
    (out_root / "README.md").write_text(
        "# WearVox Investor Demo Audio Outputs\n\n"
        "The `../audio/` folder contains the original WearVox input clips. "
        "This folder contains spoken narrations of the model responses.\n\n"
        "- `without_metadata/`: model answer when only the audio prompt was provided\n"
        "- `with_metadata/`: model answer when physical grounding metadata was also provided\n"
        "- `audio_outputs_manifest.csv`: pairs each input audio clip with both spoken outputs\n\n"
        "Disclosure: these narration files are AI-generated speech rendered from the saved model text responses.\n"
    )
    print(f"Wrote {out_root}")


if __name__ == "__main__":
    main()
