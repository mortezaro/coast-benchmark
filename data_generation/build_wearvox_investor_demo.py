import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


HF_BASE = "https://huggingface.co/datasets/zlinao/WearVox/resolve/main"


CATEGORY_RULES = [
    (
        "side_talk_nearby_conversation",
        lambda row, meta: row["task"] == "non-assistant-directed"
        and meta.get("_noise_type") == "Normal Conversation (1m)",
    ),
    (
        "side_talk_with_bystander",
        lambda row, meta: row["task"] == "non-assistant-directed"
        and meta.get("bystander__y_n_") == "Yes",
    ),
    (
        "tool_call_noisy_environment",
        lambda row, meta: row["task"] == "tool_call" and meta.get("noise_level") == "Noisy / Loud",
    ),
    (
        "traffic_or_vehicle_noise",
        lambda row, meta: meta.get("_noise_type") == "Vehicles: Car, Bus, Truck, Motorcycle (15m)",
    ),
    (
        "music_or_public_space_noise",
        lambda row, meta: meta.get("_noise_type") == "Music"
        or meta.get("location") in {"Cafe / Break Area", "Busy Hallway"},
    ),
    (
        "indoor_quiet_control",
        lambda row, meta: meta.get("environment") == "Indoors"
        and meta.get("noise_level") == "Quiet"
        and row["task"] in {"tool_call", "closed_book", "grounding"},
    ),
]


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read())


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())


def normalize_prompt(prompt: Any, max_chars: int = 1200) -> str:
    if isinstance(prompt, str):
        return prompt[:max_chars]
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "message")
            content = str(message.get("content", ""))
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"{role}: {content}")
        return "\n".join(parts)[-max_chars:]
    return str(prompt)[:max_chars]


def select_examples(data: List[Dict[str, Any]], meta_by_id: Dict[str, Dict[str, str]], total: int) -> List[Dict[str, Any]]:
    selected = []
    seen_audio = set()
    per_category_target = max(1, total // len(CATEGORY_RULES))

    for category, rule in CATEGORY_RULES:
        count = 0
        for row in data:
            meta = meta_by_id.get(row["id"])
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
            if count >= per_category_target:
                break

    if len(selected) < total:
        for row in data:
            meta = meta_by_id.get(row["id"])
            if not meta or not row.get("audio_query") or row["audio_query"] in seen_audio:
                continue
            if row["task"] not in {"non-assistant-directed", "tool_call", "grounding", "closed_book"}:
                continue
            item = dict(row)
            item["demo_category"] = "balanced_fill"
            item["audio_metadata"] = meta
            selected.append(item)
            seen_audio.add(row["audio_query"])
            if len(selected) >= total:
                break

    return selected[:total]


def grounding_metadata_text(meta: Dict[str, str]) -> str:
    bullets = [
        "Wearable egocentric audio from smart-glasses perspective",
        f"Environment: {meta.get('environment', 'unknown').lower()}",
        f"Noise condition: {meta.get('noise_level', 'unknown').lower()}",
        f"Dominant background/competing sound: {meta.get('_noise_type', 'unknown')}",
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
            "Decide whether the smart-glasses assistant should respond or stay silent. "
            "If the speech seems directed to another person or is side-talk, choose mute."
        )
    if task == "tool_call":
        return (
            "Decide what the smart-glasses assistant should do next. "
            "If a tool/action is appropriate, describe the action; if the situation is unsafe or ambiguous, say so."
        )
    return (
        "Answer the spoken question, but also state whether the physical audio situation changes your confidence "
        "or suggests asking for repetition."
    )


def prompt_text(row: Dict[str, Any], *, with_metadata: bool) -> str:
    context = ""
    if with_metadata:
        context = f"\nPhysical grounding metadata:\n{grounding_metadata_text(row['audio_metadata'])}\n"
    return f"""You are a smart-glasses voice assistant evaluating one real wearable audio clip.

{context}
Task:
{task_question(row)}

Return compact JSON with:
- decision: one of respond, mute, answer, clarify, execute_action
- answer_or_action: short answer/action, or null
- grounding_reason: one sentence explaining the physical/social context
- risk_or_confidence: one sentence

Important: do not mention that metadata was provided. Use it as latent context for interpreting the audio."""


def audio_content(path: Path) -> Dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def call_openai(
    *,
    api_key: str,
    model: str,
    audio_path: Path,
    prompt: str,
    max_retries: int = 6,
) -> Tuple[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are careful, concise, and physically grounded. Return only valid JSON.",
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
        "temperature": 0,
        "max_completion_tokens": 350,
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
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                text = parsed["choices"][0]["message"].get("content", "").strip()
                return text, parsed.get("usage")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code in {429, 500, 502, 503, 504}:
                time.sleep(min(90, 8 * (attempt + 1)))
                continue
            break
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(min(60, 5 * (attempt + 1)))
    return json.dumps({"error": last_error}), None


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_report(rows: List[Dict[str, Any]], model: str) -> str:
    counts = Counter(row["demo_category"] for row in rows)
    lines = [
        "# WearVox Grounding Metadata Investor Demo",
        "",
        f"Model: `{model}`",
        "",
        "Each example was run twice with the same audio: first without grounding metadata, then with physical grounding metadata.",
        "",
        "## Category Mix",
        "",
    ]
    for category, count in counts.most_common():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Examples", ""])
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"### {index}. {row['demo_category']} / {row['task']} / `{row['id']}`",
                "",
                f"Transcript for presenter: `{row.get('gt_transcript', '')}`",
                "",
                f"Ground truth: `{row.get('ground_truth', '')}`",
                "",
                "Physical grounding metadata used in metadata condition:",
                "",
                grounding_metadata_text(row["audio_metadata"]),
                "",
                "**Without grounding metadata**",
                "",
                "```json",
                row["without_metadata_response"],
                "```",
                "",
                "**With grounding metadata**",
                "",
                "```json",
                row["with_metadata_response"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="gpt-audio")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    api_key = ""
    if not args.skip_api:
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required unless --skip-api is used")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    audio_dir = args.output_dir / "audio"
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    data_path = cache_dir / "data_public.json"
    meta_path = cache_dir / "audio_metadata.json"
    if not data_path.exists():
        data_path.write_text(
            json.dumps(fetch_json(f"{HF_BASE}/data_public.json"), ensure_ascii=False),
            encoding="utf-8",
        )
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(fetch_json(f"{HF_BASE}/audio_metadata.json"), ensure_ascii=False),
            encoding="utf-8",
        )

    data = json.loads(data_path.read_text(encoding="utf-8"))
    meta_rows = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_by_id = {row["id"]: row["audio_metadata"] for row in meta_rows}
    selected = select_examples(data, meta_by_id, args.count)

    output_rows = []
    for index, row in enumerate(selected, start=1):
        audio_rel = row["audio_query"]
        audio_path = audio_dir / Path(audio_rel).name
        download(f"{HF_BASE}/{audio_rel}", audio_path)
        row["local_audio_file"] = str(audio_path)
        row["without_metadata_prompt"] = prompt_text(row, with_metadata=False)
        row["with_metadata_prompt"] = prompt_text(row, with_metadata=True)

        if args.skip_api:
            row["without_metadata_response"] = "{}"
            row["with_metadata_response"] = "{}"
            row["without_metadata_usage"] = None
            row["with_metadata_usage"] = None
        else:
            no_meta, no_usage = call_openai(
                api_key=api_key,
                model=args.model,
                audio_path=audio_path,
                prompt=row["without_metadata_prompt"],
            )
            time.sleep(args.sleep_sec)
            with_meta, with_usage = call_openai(
                api_key=api_key,
                model=args.model,
                audio_path=audio_path,
                prompt=row["with_metadata_prompt"],
            )
            row["without_metadata_response"] = no_meta
            row["with_metadata_response"] = with_meta
            row["without_metadata_usage"] = no_usage
            row["with_metadata_usage"] = with_usage
            time.sleep(args.sleep_sec)

        output_rows.append(row)
        write_jsonl(args.output_dir / "wearvox_grounding_demo_results.jsonl", output_rows)
        print(f"[{index}/{len(selected)}] {row['demo_category']} {row['id']}", flush=True)

    write_jsonl(args.output_dir / "wearvox_grounding_demo_selected_examples.jsonl", selected)
    report = render_report(output_rows, args.model)
    (args.output_dir / "wearvox_grounding_demo_report.md").write_text(report, encoding="utf-8")
    status = {
        "model": args.model,
        "count": len(output_rows),
        "category_counts": dict(Counter(row["demo_category"] for row in output_rows)),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "wearvox_grounding_demo_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
