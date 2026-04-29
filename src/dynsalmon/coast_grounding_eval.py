import json
import math
import random
import shlex
import statistics
import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_text(text: str) -> list[str]:
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else " ")
    return [token for token in "".join(cleaned).split() if token]


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(_normalize_text(a))
    b_tokens = set(_normalize_text(b))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return overlap / float(union)


@dataclass(slots=True)
class AudioFeatures:
    duration_sec: float
    rms: float
    onset_strength: float
    periodicity_score: float
    zero_crossing_rate: float


def _load_wav_samples(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
        sample_width = handle.getsampwidth()
        channels = handle.getnchannels()
    if sample_width != 2:
        raise ValueError(f"Unsupported wav sample width for {path}: {sample_width}")
    samples_array = array("h")
    samples_array.frombytes(frames)
    raw = samples_array.tolist()
    if channels > 1:
        mono: list[float] = []
        for index in range(0, len(raw), channels):
            frame = raw[index : index + channels]
            mono.append(sum(frame) / float(len(frame)))
    else:
        mono = [float(sample) for sample in raw]
    return [sample / 32768.0 for sample in mono], sample_rate


def _frame_energy(samples: list[float], frame_size: int, hop: int) -> list[float]:
    if not samples:
        return [0.0]
    energies: list[float] = []
    for start in range(0, max(1, len(samples) - frame_size + 1), hop):
        frame = samples[start : start + frame_size]
        if not frame:
            break
        energies.append(math.sqrt(sum(sample * sample for sample in frame) / float(len(frame))))
    return energies or [0.0]


def _autocorrelation_peak(values: list[float], min_lag: int, max_lag: int) -> float:
    if not values or max_lag <= min_lag:
        return 0.0
    centered = [value - statistics.fmean(values) for value in values]
    denom = sum(value * value for value in centered)
    if denom <= 1e-9:
        return 0.0
    best = 0.0
    upper = min(max_lag, len(centered) - 1)
    for lag in range(max(min_lag, 1), upper + 1):
        score = sum(centered[index] * centered[index + lag] for index in range(len(centered) - lag)) / denom
        if score > best:
            best = score
    return best


def extract_audio_features(path: Path) -> AudioFeatures:
    samples, sample_rate = _load_wav_samples(path)
    if not samples:
        return AudioFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
    rms = math.sqrt(sum(sample * sample for sample in samples) / float(len(samples)))
    duration_sec = len(samples) / float(sample_rate)
    frame_size = max(64, int(sample_rate * 0.032))
    hop = max(32, int(sample_rate * 0.010))
    energies = _frame_energy(samples, frame_size, hop)
    diffs = [max(0.0, energies[index + 1] - energies[index]) for index in range(len(energies) - 1)]
    onset_strength = max(diffs) if diffs else 0.0
    min_lag = max(1, int(0.08 / (hop / float(sample_rate))))
    max_lag = max(min_lag + 1, int(0.75 / (hop / float(sample_rate))))
    periodicity_score = _autocorrelation_peak(energies, min_lag=min_lag, max_lag=max_lag)
    zero_crossings = 0
    for left, right in zip(samples, samples[1:]):
        if (left >= 0 > right) or (left < 0 <= right):
            zero_crossings += 1
    zero_crossing_rate = zero_crossings / max(1.0, duration_sec)
    return AudioFeatures(
        duration_sec=duration_sec,
        rms=rms,
        onset_strength=onset_strength,
        periodicity_score=periodicity_score,
        zero_crossing_rate=zero_crossing_rate,
    )


class GroundingTaskModel:
    def predict(self, task_name: str, row: dict[str, Any], conditioning: str) -> dict[str, Any]:
        raise NotImplementedError


class RandomGroundingModel(GroundingTaskModel):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def predict(self, task_name: str, row: dict[str, Any], conditioning: str) -> dict[str, Any]:
        if "candidate_audio_paths" in row:
            scores = [self._rng.random() for _ in row["candidate_audio_paths"]]
            predicted_index = max(range(len(scores)), key=scores.__getitem__)
            return {"scores": scores, "predicted_index": predicted_index}
        options = row.get("text_options") or row.get("label_space") or []
        scores = [self._rng.random() for _ in options]
        predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
        return {"scores": scores, "predicted_index": predicted_index}


class HybridHeuristicGroundingModel(GroundingTaskModel):
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._audio_cache: dict[str, AudioFeatures] = {}

    def _features(self, path: str) -> AudioFeatures:
        if path not in self._audio_cache:
            self._audio_cache[path] = extract_audio_features(Path(path))
        return self._audio_cache[path]

    def _candidate_scores(self, row: dict[str, Any], conditioning: str) -> list[float]:
        prompt = str(row.get("prompt", ""))
        labels = row.get("candidate_labels") or []
        paths = row.get("candidate_audio_paths") or []
        scores: list[float] = []
        for index, path in enumerate(paths):
            label = labels[index] if index < len(labels) else ""
            features = self._features(path)
            score = 0.0
            if conditioning == "audio_text":
                score += 2.0 * _token_overlap_score(prompt, str(label))
            score += 0.2 * features.rms
            score += 0.1 * features.onset_strength
            score += self._rng.random() * 1e-6
            scores.append(score)
        return scores

    def _text_option_scores(self, task_name: str, row: dict[str, Any], conditioning: str) -> list[float]:
        prompt = str(row.get("prompt", ""))
        options = row.get("text_options") or row.get("label_space") or []
        query_audio_path = row.get("query_audio_path")
        features = self._features(query_audio_path) if query_audio_path else AudioFeatures(0.0, 0.0, 0.0, 0.0, 0.0)
        scores: list[float] = []
        for option in options:
            option_text = str(option)
            score = 0.0
            if conditioning == "audio_text":
                score += 1.5 * _token_overlap_score(prompt, option_text)
            if task_name == "periodicity_aware_grounding":
                periodic_bias = features.periodicity_score
                if "repetitive" in option_text.lower() or "rhythmic" in option_text.lower():
                    score += 2.0 * periodic_bias
                if "single" in option_text.lower() or "impact" in option_text.lower():
                    score += 2.0 * (1.0 - periodic_bias)
            elif task_name == "onset_sharpness_grounding":
                sharp_bias = min(1.0, features.onset_strength * 8.0)
                if "sharp" in option_text.lower() or "sudden" in option_text.lower():
                    score += 2.0 * sharp_bias
                if "diffuse" in option_text.lower() or "smeared" in option_text.lower():
                    score += 2.0 * (1.0 - sharp_bias)
            elif task_name == "foreground_event_focus":
                if option_text.lower() == "foreground":
                    score += 1.8 * min(1.0, features.rms * 6.0)
                elif option_text.lower() == "background":
                    score += 1.0 * min(1.0, features.rms * 4.0)
                elif option_text.lower() == "absent":
                    score += 1.2 * max(0.0, 1.0 - features.onset_strength * 8.0)
            scores.append(score + self._rng.random() * 1e-6)
        return scores

    def predict(self, task_name: str, row: dict[str, Any], conditioning: str) -> dict[str, Any]:
        if "candidate_audio_paths" in row:
            scores = self._candidate_scores(row, conditioning)
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            if task_name == "acoustic_plausibility_ranking":
                predicted_ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
                return {
                    "scores": scores,
                    "predicted_index": predicted_index,
                    "predicted_ranking": predicted_ranking,
                }
            return {"scores": scores, "predicted_index": predicted_index}
        scores = self._text_option_scores(task_name, row, conditioning)
        predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
        return {"scores": scores, "predicted_index": predicted_index}


class ExternalCommandGroundingModel(GroundingTaskModel):
    def __init__(self, command: str) -> None:
        self._command = shlex.split(command)

    def predict(self, task_name: str, row: dict[str, Any], conditioning: str) -> dict[str, Any]:
        payload = {
            "task_name": task_name,
            "conditioning": conditioning,
            "row": row,
        }
        result = subprocess.run(
            self._command,
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        parsed = json.loads(result.stdout.decode("utf-8"))
        if "predicted_index" not in parsed and "predicted_ranking" not in parsed:
            raise ValueError("External model must return predicted_index or predicted_ranking")
        return parsed


def build_grounding_model(*, backend: str, seed: int = 0, command: str | None = None) -> GroundingTaskModel:
    if backend == "random":
        return RandomGroundingModel(seed=seed)
    if backend == "hybrid_heuristic":
        return HybridHeuristicGroundingModel(seed=seed)
    if backend == "external_command":
        if not command:
            raise ValueError("--command is required for external_command backend")
        return ExternalCommandGroundingModel(command=command)
    raise ValueError(f"Unknown backend: {backend}")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stable_seed_offset(text: str) -> int:
    total = 0
    for index, char in enumerate(text):
        total += (index + 1) * ord(char)
    return total % 100000


def _bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "bootstrap_samples": bootstrap_samples}
    rng = random.Random(seed)
    n = len(values)
    distribution: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        distribution.append(statistics.fmean(sample))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean": statistics.fmean(values),
        "ci_low": _quantile(distribution, alpha),
        "ci_high": _quantile(distribution, 1.0 - alpha),
        "bootstrap_samples": bootstrap_samples,
    }


def _predicted_ranking(row: dict[str, Any], prediction: dict[str, Any]) -> list[int]:
    if "predicted_ranking" in prediction and prediction["predicted_ranking"] is not None:
        return [int(index) for index in prediction["predicted_ranking"]]
    if "scores" in prediction and prediction["scores"] is not None:
        scores = list(prediction["scores"])
        return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    candidate_count = len(row.get("candidate_audio_paths") or row.get("text_options") or row.get("label_space") or [])
    predicted_index = int(prediction.get("predicted_index", 0))
    remaining = [index for index in range(candidate_count) if index != predicted_index]
    return [predicted_index, *remaining]


def _dcg(relevances: list[float]) -> float:
    total = 0.0
    for index, rel in enumerate(relevances, start=1):
        total += rel / math.log2(index + 1.0)
    return total


def _average_precision(relevance_flags: list[int]) -> float:
    relevant_total = sum(relevance_flags)
    if relevant_total <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, flag in enumerate(relevance_flags, start=1):
        if flag:
            hits += 1
            precision_sum += hits / float(index)
    return precision_sum / float(relevant_total)


def _safe_label_from_row(row: dict[str, Any], index: int) -> str | None:
    if "label_space" in row:
        labels = row["label_space"]
        if 0 <= index < len(labels):
            return str(labels[index])
    if "text_options" in row:
        options = row["text_options"]
        if 0 <= index < len(options):
            return str(options[index])
    return None


def _score_prediction(task_name: str, row: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    if task_name == "acoustic_plausibility_ranking":
        gold_ranking = row["gold_ranking"]
        predicted_ranking = _predicted_ranking(row, prediction)
        top1_correct = int(predicted_ranking[0] == gold_ranking[0])
        exact_ranking = int(predicted_ranking == gold_ranking)
        try:
            reciprocal_rank = 1.0 / float(predicted_ranking.index(gold_ranking[0]) + 1)
        except ValueError:
            reciprocal_rank = 0.0
        relevance_by_index = {
            index: float(len(gold_ranking) - rank_position)
            for rank_position, index in enumerate(gold_ranking)
        }
        predicted_relevances = [relevance_by_index.get(index, 0.0) for index in predicted_ranking]
        ideal_relevances = sorted(relevance_by_index.values(), reverse=True)
        dcg = _dcg(predicted_relevances)
        idcg = _dcg(ideal_relevances)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return {
            "primary_metric": float(top1_correct),
            "top1_accuracy": float(top1_correct),
            "exact_ranking_accuracy": float(exact_ranking),
            "mrr": reciprocal_rank,
            "ndcg": ndcg,
            "gold": gold_ranking,
            "predicted": predicted_ranking,
        }
    if "gold_label" in row:
        label_space = row["label_space"]
        predicted_index = int(prediction["predicted_index"])
        predicted_label = label_space[predicted_index]
        correct = int(predicted_label == row["gold_label"])
        return {
            "primary_metric": float(correct),
            "accuracy": float(correct),
            "gold": row["gold_label"],
            "predicted": predicted_label,
            "gold_label_text": str(row["gold_label"]),
            "predicted_label_text": str(predicted_label),
        }
    gold_index = int(row["gold_index"])
    predicted_ranking = _predicted_ranking(row, prediction)
    predicted_index = int(predicted_ranking[0]) if predicted_ranking else 0
    correct = int(gold_index == predicted_index)
    result = {
        "primary_metric": float(correct),
        "accuracy": float(correct),
        "gold": gold_index,
        "predicted": predicted_index,
    }
    if "candidate_audio_paths" in row:
        try:
            rank = predicted_ranking.index(gold_index) + 1
        except ValueError:
            rank = max(1, len(predicted_ranking) + 1)
        reciprocal_rank = 1.0 / float(rank)
        relevance_flags = [1 if index == gold_index else 0 for index in predicted_ranking]
        ndcg = 1.0 / math.log2(rank + 1.0) if rank >= 1 else 0.0
        result.update(
            {
                "rank": float(rank),
                "mrr": reciprocal_rank,
                "average_precision": _average_precision(relevance_flags),
                "ndcg": ndcg,
                "recall_at_1": float(rank <= 1),
                "recall_at_2": float(rank <= 2),
                "recall_at_3": float(rank <= 3),
            }
        )
    gold_text = _safe_label_from_row(row, gold_index)
    predicted_text = _safe_label_from_row(row, predicted_index)
    if gold_text is not None and predicted_text is not None:
        result["gold_label_text"] = gold_text
        result["predicted_label_text"] = predicted_text
    return result


def _macro_classification_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    def extract_label(record: dict[str, Any], key: str) -> str | None:
        if record["metrics"].get(f"{key}_label_text") is not None:
            return str(record["metrics"][f"{key}_label_text"])
        value = record["metrics"].get(key)
        return str(value) if isinstance(value, str) else None

    labels = set()
    for record in records:
        gold_label = extract_label(record, "gold")
        predicted_label = extract_label(record, "predicted")
        if gold_label is not None:
            labels.add(gold_label)
        if predicted_label is not None:
            labels.add(predicted_label)
    labels = sorted(labels)
    if not labels:
        return {}
    per_label = []
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for record in records:
            gold = extract_label(record, "gold")
            predicted = extract_label(record, "predicted")
            if predicted == label and gold == label:
                tp += 1
            elif predicted == label and gold != label:
                fp += 1
            elif predicted != label and gold == label:
                fn += 1
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / float(precision + recall)) if (precision + recall) else 0.0
        per_label.append((precision, recall, f1))
    return {
        "macro_precision": statistics.fmean(value[0] for value in per_label),
        "macro_recall": statistics.fmean(value[1] for value in per_label),
        "macro_f1": statistics.fmean(value[2] for value in per_label),
    }


def _task_summary_with_bootstrap(
    *,
    task_name: str,
    records: list[dict[str, Any]],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    numeric_metric_names = sorted(
        {
            key
            for record in records
            for key, value in record["metrics"].items()
            if isinstance(value, (int, float)) and key not in {"gold", "predicted"}
        }
    )
    summary = {
        "task_name": task_name,
        "count": len(records),
    }
    metric_cis: dict[str, dict[str, float]] = {}
    for metric_name in numeric_metric_names:
        values = [float(record["metrics"][metric_name]) for record in records]
        metric_cis[metric_name] = _bootstrap_mean_ci(
            values,
            seed=seed + _stable_seed_offset(f"{task_name}:{metric_name}"),
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
        )
        summary[metric_name] = metric_cis[metric_name]["mean"]
    summary["primary_metric_mean"] = metric_cis.get("primary_metric", {}).get(
        "mean",
        statistics.fmean(float(record["metrics"]["primary_metric"]) for record in records),
    )
    if "average_precision" in summary:
        summary["map"] = summary["average_precision"]
    if "recall_at_1" in summary:
        summary["top1_recall"] = summary["recall_at_1"]
    summary["metric_confidence_intervals"] = metric_cis

    label_records = [
        record
        for record in records
        if record["metrics"].get("gold") is not None and record["metrics"].get("predicted") is not None
    ]
    if label_records:
        summary.update(_macro_classification_summary(label_records))
    return summary


def _paired_effect_size(differences: list[float]) -> float:
    if len(differences) < 2:
        return 0.0
    mean_diff = statistics.fmean(differences)
    sd_diff = statistics.stdev(differences)
    if sd_diff <= 1e-12:
        return 0.0
    d_z = mean_diff / sd_diff
    correction = 1.0 - (3.0 / max(1.0, (4.0 * len(differences) - 5.0)))
    return correction * d_z


def _bootstrap_compare_metrics(
    current_values: list[float],
    reference_values: list[float],
    *,
    seed: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> dict[str, float]:
    if not current_values or not reference_values or len(current_values) != len(reference_values):
        return {
            "count": 0,
            "observed_difference": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value_two_sided": 1.0,
            "hedges_g": 0.0,
            "bootstrap_samples": bootstrap_samples,
        }
    rng = random.Random(seed)
    n = len(current_values)
    observed = statistics.fmean(current_values) - statistics.fmean(reference_values)
    differences = [current_values[index] - reference_values[index] for index in range(n)]
    distribution: list[float] = []
    for _ in range(bootstrap_samples):
        sample_diffs = [differences[rng.randrange(n)] for _ in range(n)]
        distribution.append(statistics.fmean(sample_diffs))
    alpha = (1.0 - confidence_level) / 2.0
    positive_tail = (sum(1 for diff in distribution if diff >= 0.0) + 1.0) / float(bootstrap_samples + 1)
    negative_tail = (sum(1 for diff in distribution if diff <= 0.0) + 1.0) / float(bootstrap_samples + 1)
    return {
        "count": n,
        "observed_difference": observed,
        "ci_low": _quantile(distribution, alpha),
        "ci_high": _quantile(distribution, 1.0 - alpha),
        "alternative": "two_sided",
        "p_value_two_sided": min(1.0, 2.0 * min(positive_tail, negative_tail)),
        "hedges_g": _paired_effect_size(differences),
        "bootstrap_samples": bootstrap_samples,
    }


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _read_jsonl(path)


def _compare_against_reference(
    *,
    current_prediction_rows: list[dict[str, Any]],
    reference_prediction_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    current_index = {
        (row["task_dir"], row["id"]): row
        for row in current_prediction_rows
    }
    reference_index = {
        (row["task_dir"], row["id"]): row
        for row in reference_prediction_rows
    }
    shared_keys = sorted(set(current_index) & set(reference_index))
    overall_current: list[float] = []
    overall_reference: list[float] = []
    by_task: dict[str, dict[str, list[float]]] = {}
    for key in shared_keys:
        current_row = current_index[key]
        reference_row = reference_index[key]
        task_dir = current_row["task_dir"]
        current_value = float(current_row["metrics"]["primary_metric"])
        reference_value = float(reference_row["metrics"]["primary_metric"])
        overall_current.append(current_value)
        overall_reference.append(reference_value)
        by_task.setdefault(task_dir, {"current": [], "reference": []})
        by_task[task_dir]["current"].append(current_value)
        by_task[task_dir]["reference"].append(reference_value)
    task_results = {
        task_dir: _bootstrap_compare_metrics(
            values["current"],
            values["reference"],
            seed=seed + _stable_seed_offset(task_dir),
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
        )
        for task_dir, values in sorted(by_task.items())
    }
    return {
        "shared_prediction_count": len(shared_keys),
        "overall": _bootstrap_compare_metrics(
            overall_current,
            overall_reference,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
        ),
        "tasks": task_results,
    }


def evaluate_grounding_suite(
    *,
    grounding_suite_dir: Path,
    output_dir: Path,
    backend: str = "hybrid_heuristic",
    conditioning: str = "audio_text",
    seed: int = 0,
    command: str | None = None,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    reference_predictions_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_grounding_model(backend=backend, seed=seed, command=command)
    task_paths = sorted(path for path in grounding_suite_dir.glob("task_*") if path.is_dir())
    summary_tasks: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []

    for task_path in task_paths:
        data_path = task_path / "all.jsonl"
        if not data_path.exists():
            continue
        rows = _read_jsonl(data_path)
        task_name = rows[0]["task_name"] if rows else task_path.name
        task_predictions: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for row in rows:
            prediction = model.predict(task_name, row, conditioning)
            scored = _score_prediction(task_name, row, prediction)
            record = {
                "task_dir": task_path.name,
                "task_name": task_name,
                "id": row["id"],
                "prediction": prediction,
                "metrics": scored,
            }
            task_predictions.append(record)
            prediction_rows.append(record)
            metrics.append(scored)
        if not task_predictions:
            continue
        task_summary = _task_summary_with_bootstrap(
            task_name=task_name,
            records=task_predictions,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        summary_tasks[task_path.name] = task_summary
        _write_jsonl(output_dir / task_path.name / "predictions.jsonl", task_predictions)

    overall_primary = [
        float(record["metrics"]["primary_metric"])
        for record in prediction_rows
    ]
    summary = {
        "suite_dir": str(grounding_suite_dir),
        "output_dir": str(output_dir),
        "backend": backend,
        "conditioning": conditioning,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "current_metric_family": "choice_accuracy",
        "current_metric_note": (
            "The packaged evaluator reports accuracy-centered summaries together with "
            "task-appropriate retrieval metrics, bootstrap confidence intervals, and "
            "optional paired bootstrap comparisons against a reference prediction set."
        ),
        "task_count": len(summary_tasks),
        "overall_primary_metric_mean": statistics.fmean(overall_primary) if overall_primary else 0.0,
        "tasks": summary_tasks,
    }
    summary["overall_primary_metric_ci"] = _bootstrap_mean_ci(
        overall_primary,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )
    if reference_predictions_dir is not None:
        reference_prediction_rows = _load_prediction_rows(reference_predictions_dir / "predictions.jsonl")
        summary["reference_predictions_dir"] = str(reference_predictions_dir)
        summary["bootstrap_comparison"] = _compare_against_reference(
            current_prediction_rows=prediction_rows,
            reference_prediction_rows=reference_prediction_rows,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    return summary
