import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


HF_BASE = "https://huggingface.co/datasets/zlinao/WearVox/resolve/main"


CategoryRule = Tuple[str, Callable[[Dict[str, Any], Dict[str, str]], bool]]


CATEGORY_RULES: List[CategoryRule] = [
    (
        "bystander_side_talk",
        lambda row, meta: row.get("task") == "non-assistant-directed"
        and meta.get("bystander__y_n_") == "Yes"
        and any(token in row.get("id", "") for token in ["bystander", "unrelated", "partner"]),
    ),
    (
        "nearby_side_conversation",
        lambda row, meta: row.get("task") == "non-assistant-directed"
        and meta.get("_noise_type") == "Normal Conversation (1m)",
    ),
    (
        "vehicle_or_street_context",
        lambda row, meta: meta.get("_noise_type") == "Vehicles: Car, Bus, Truck, Motorcycle (15m)"
        or meta.get("location") in {"Inside Car (windows down)", "Street / Crosswalk", "Parking Lot (near-bus / cars)"},
    ),
    (
        "construction_or_machine_noise",
        lambda row, meta: meta.get("_noise_type") in {
            "Construction Noise: Jackhammer, other (15m)",
            "Vacuum Cleaning",
            "Lawnmower (1m)",
            "Aircraft: Jet plane, military jet take-off (30m)",
        },
    ),
    (
        "music_or_cafe_context",
        lambda row, meta: meta.get("_noise_type") == "Music"
        or meta.get("location") in {"Cafe / Break Area", "Busy Hallway"},
    ),
    (
        "orientation_distance_context",
        lambda row, meta: row.get("task") in {"tool_call", "grounding", "non-assistant-directed"}
        and meta.get("partner_position") in {"-60", "60", "-30", "30"}
        and meta.get("partner_distance") == "1.5m",
    ),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug[:max_len] or "item"


def audio_content(path: Path) -> Dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def compact_prompt(prompt: Any, limit: int = 700) -> str:
    if isinstance(prompt, str):
        return prompt[:limit]
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", ""))
            if "Sometimes, the user may need" in content:
                continue
            if len(content) > 260:
                content = content[:260] + "..."
            parts.append(f"{message.get('role', 'message')}: {content}")
        return "\n".join(parts)[-limit:]
    return str(prompt)[:limit]


def grounding_metadata_text(meta: Dict[str, str]) -> str:
    bullets = [
        "Egocentric wearable microphone from smart glasses",
        f"Environment: {meta.get('environment', 'unknown')}",
        f"Location type: {meta.get('location', 'unknown')}",
        f"Noise level: {meta.get('noise_level', 'unknown')}",
        f"Competing/background sound: {meta.get('_noise_type', 'unknown')}",
    ]
    if meta.get("partner_position") != "N/A":
        bullets.append(f"Primary nearby speech/source direction: {meta.get('partner_position', 'unknown')} degrees from wearer")
    if meta.get("partner_distance") != "N/A":
        bullets.append(f"Primary nearby speech/source distance: {meta.get('partner_distance', 'unknown')}")
    bullets.append(f"Bystander speech present: {meta.get('bystander__y_n_', 'unknown')}")
    return "\n".join(f"- {bullet}" for bullet in bullets)


def demo_question(row: Dict[str, Any], with_metadata: bool) -> str:
    metadata = ""
    if with_metadata:
        metadata = f"\nPhysical grounding metadata available to the assistant:\n{grounding_metadata_text(row['audio_metadata'])}\n"
    return f"""You are demonstrating a smart-glasses audio assistant.

Listen to the input audio and give a short spoken diagnostic answer for an investor demo.

{metadata}
Original task context:
{compact_prompt(row.get('text_prompt'))}

Known transcript from the dataset, for reference only:
{row.get('gt_transcript', '')}

Question:
What is happening physically and socially, and what should the assistant do?

Speak 1-2 natural sentences. Make the answer audible; never answer with silence.
Use first person: "I would..." or "I should...".
If the clip sounds like side conversation, say whether you would avoid responding.
If the physical setting is noisy, distant, angled, moving, or public, say how that changes confidence or action.
Do not say "metadata", "grounding", or "model" in the spoken answer."""


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
                    "You are concise and good at explaining situated audio behavior. "
                    "Speak only the answer requested; do not reveal hidden inputs."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Input audio clip:"},
                    audio_content(audio_path),
                ],
            },
        ],
        "temperature": 0.3,
        "max_completion_tokens": 220,
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
                parsed = json.loads(response.read().decode("utf-8"))
                message = parsed["choices"][0]["message"]
                audio = message.get("audio") or {}
                if not audio.get("data"):
                    raise RuntimeError(f"No audio data returned: {json.dumps(parsed)[:1000]}")
                return base64.b64decode(audio["data"]), audio.get("transcript", ""), parsed.get("usage", {})
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {error_text}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(last_error)
        except Exception as exc:  # pragma: no cover
            last_error = repr(exc)
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"audio model failed after retries: {last_error}")


def transcript_score(without_text: str, with_text: str) -> int:
    a = without_text.lower()
    b = with_text.lower()
    score = 0
    contrast_pairs = [
        ("respond", "avoid"),
        ("respond", "not respond"),
        ("answer", "clarify"),
        ("do", "repeat"),
        ("confident", "uncertain"),
    ]
    for left, right in contrast_pairs:
        if left in a and right in b:
            score += 3
    physical_terms = [
        "bystander",
        "side conversation",
        "nearby",
        "distance",
        "angle",
        "noise",
        "noisy",
        "traffic",
        "vehicle",
        "public",
        "room",
        "construction",
        "music",
        "unclear",
        "repeat",
        "clarify",
    ]
    score += sum(1 for term in physical_terms if term in b and term not in a)
    if a.strip() != b.strip():
        score += 1
    return score


def select_examples(data: List[Dict[str, Any]], meta_by_id: Dict[str, Dict[str, str]], per_category: int) -> List[Dict[str, Any]]:
    selected = []
    seen_audio = set()
    for category, rule in CATEGORY_RULES:
        count = 0
        for row in data:
            meta = meta_by_id.get(row.get("id"))
            if not meta or not row.get("audio_query"):
                continue
            if row["audio_query"] in seen_audio:
                continue
            if not rule(row, meta):
                continue
            item = dict(row)
            item["demo_category"] = category
            item["audio_metadata"] = meta
            selected.append(item)
            seen_audio.add(row["audio_query"])
            count += 1
            if count >= per_category:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-audio")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3", "flac", "opus", "pcm16"])
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--keep-top", type=int, default=30)
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --api-key.")

    data = load_json(args.cache_root / "data_public.json")
    meta_rows = load_json(args.cache_root / "audio_metadata.json")
    meta_by_id = {row["id"]: row["audio_metadata"] for row in meta_rows}
    candidates = select_examples(data, meta_by_id, args.per_category)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selected_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    print(f"selected {len(candidates)} candidates: {Counter(row['demo_category'] for row in candidates)}", flush=True)

    manifest = []
    for index, row in enumerate(candidates, start=1):
        rel_audio = row["audio_query"]
        local_audio = args.output_root / "audio" / Path(rel_audio).name
        download(f"{HF_BASE}/{rel_audio}", local_audio)
        slug = f"{index:03d}_{safe_slug(row['demo_category'])}_{safe_slug(row['id'], 42)}"
        item = {
            "candidate_index": index,
            "id": row["id"],
            "task": row["task"],
            "demo_category": row["demo_category"],
            "input_audio": str(local_audio),
            "metadata": row["audio_metadata"],
            "gt_transcript": row.get("gt_transcript", ""),
            "ground_truth": row.get("ground_truth", ""),
        }
        for condition, with_metadata in [("without_grounding", False), ("with_grounding", True)]:
            out_audio = args.output_root / "all_model_outputs" / condition / f"{slug}.{args.format}"
            out_text = args.output_root / "all_model_outputs" / condition / f"{slug}.txt"
            if out_audio.exists() and out_audio.stat().st_size > 0 and out_text.exists():
                transcript = out_text.read_text().strip()
            else:
                audio_bytes, transcript, usage = call_audio_model(
                    api_key=api_key,
                    model=args.model,
                    audio_path=local_audio,
                    prompt=demo_question(row, with_metadata=with_metadata),
                    voice=args.voice,
                    fmt=args.format,
                )
                out_audio.parent.mkdir(parents=True, exist_ok=True)
                out_audio.write_bytes(audio_bytes)
                out_text.write_text(transcript.strip() + "\n")
                with (args.output_root / "usage.jsonl").open("a") as handle:
                    handle.write(json.dumps({"id": row["id"], "condition": condition, "usage": usage}) + "\n")
            item[f"{condition}_audio"] = str(out_audio)
            item[f"{condition}_transcript"] = transcript
        item["contrast_score"] = transcript_score(item["without_grounding_transcript"], item["with_grounding_transcript"])
        manifest.append(item)
        print(
            f"[{index}/{len(candidates)}] score={item['contrast_score']} {row['demo_category']} {row['id']}",
            flush=True,
        )

    ranked = sorted(manifest, key=lambda row: row["contrast_score"], reverse=True)
    keep = ranked[: args.keep_top]
    (args.output_root / "all_outputs_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_root / "curated_top_manifest.json").write_text(json.dumps(keep, indent=2) + "\n")
    guide_lines = [
        "# WearVox Grounding Contrast Demo",
        "",
        "This package is designed for audio-in/audio-out investor demos.",
        "",
        "Play order for each curated example:",
        "1. input audio",
        "2. `without_grounding` model speech",
        "3. `with_grounding` model speech",
        "",
        "Curated examples are ranked by transcript-level contrast; human listening should make the final selection.",
        "",
    ]
    for item in keep:
        guide_lines.extend(
            [
                f"## {item['candidate_index']:03d} — {item['demo_category']} — score {item['contrast_score']}",
                f"- Input: `{item['input_audio']}`",
                f"- Without grounding: `{item['without_grounding_audio']}`",
                f"- With grounding: `{item['with_grounding_audio']}`",
                f"- Transcript: {item['gt_transcript']}",
                f"- Without: {item['without_grounding_transcript']}",
                f"- With: {item['with_grounding_transcript']}",
                "",
            ]
        )
    (args.output_root / "CURATED_INVESTOR_GUIDE.md").write_text("\n".join(guide_lines))
    print(f"wrote {args.output_root}")


if __name__ == "__main__":
    main()
