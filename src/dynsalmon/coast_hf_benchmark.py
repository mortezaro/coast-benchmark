import base64
import json
import math
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynsalmon.coast_grounding_eval import (
    _bootstrap_mean_ci,
    _compare_against_reference,
    _read_jsonl,
    _score_prediction,
    _task_summary_with_bootstrap,
    _write_jsonl,
)


def _lazy_import_audio() -> tuple[Any, Any]:
    import librosa  # type: ignore
    import numpy as np  # type: ignore

    return librosa, np


def _lazy_import_torch() -> Any:
    import torch  # type: ignore

    return torch


def _safe_slug(text: str) -> str:
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else "_")
    return re.sub(r"_+", "_", "".join(cleaned)).strip("_")


def _extract_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return int(match.group(0))


def _extract_first_letter(text: str, valid_letters: list[str]) -> str | None:
    upper_text = text.upper()
    for char in upper_text:
        if char in valid_letters:
            return char
    return None


def _row_text_context(row: dict[str, Any]) -> str:
    for key in ("prompt", "query_text"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _normalized_terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lexical_similarity(left: str, right: str) -> float:
    left_terms = _normalized_terms(left)
    right_terms = _normalized_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    left_counts = Counter(left_terms)
    right_counts = Counter(right_terms)
    overlap = sum(min(left_counts[token], right_counts[token]) for token in left_counts.keys() & right_counts.keys())
    denom = max(len(left_terms), len(right_terms), 1)
    return float(overlap) / float(denom)


@dataclass(slots=True)
class TierSuite:
    tier_name: str
    suite_dir: Path
    output_dir: Path


class HFAudioBenchmarkModel:
    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class _CausalLMScoringMixin:
    def _tokenizer_input_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        return list(encoded["input_ids"])

    def _avg_logprob_for_suffix(self, prefix_text: str, suffix_text: str) -> float:
        torch = self._torch
        prefix_ids = self._tokenizer_input_ids(prefix_text)
        suffix_ids = self._tokenizer_input_ids(suffix_text)
        if not suffix_ids:
            return float("-inf")

        input_ids = prefix_ids + suffix_ids
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        prefix_offset = len(prefix_ids)
        if bos_token_id is not None:
            input_ids = [int(bos_token_id)] + input_ids
            prefix_offset += 1

        max_positions = int(getattr(getattr(self.model, "config", None), "max_position_embeddings", 4096))
        max_positions = max(max_positions, len(suffix_ids) + 8)
        if len(input_ids) > max_positions:
            keep_prefix = max(0, max_positions - len(suffix_ids))
            input_ids = input_ids[-(keep_prefix + len(suffix_ids)) :]
            prefix_offset = keep_prefix

        tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(tensor)
        with torch.no_grad():
            outputs = self.model(input_ids=tensor, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = tensor[:, 1:]

        suffix_start = max(prefix_offset - 1, 0)
        if suffix_start >= targets.shape[1]:
            return float("-inf")

        suffix_logits = logits[:, suffix_start:, :]
        suffix_targets = targets[:, suffix_start:]
        logprobs = torch.log_softmax(suffix_logits, dim=-1)
        token_logprobs = logprobs.gather(-1, suffix_targets.unsqueeze(-1)).squeeze(-1)
        return float(token_logprobs.mean().item())


class _AudioCacheMixin:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._audio_cache: dict[str, Any] = {}

    def _load_audio_array(self, path: str) -> Any:
        if path not in self._audio_cache:
            librosa, np = _lazy_import_audio()
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            self._audio_cache[path] = np.asarray(audio, dtype=np.float32)
        return self._audio_cache[path]

    def _combine_audio(self, first_path: str, second_path: str, *, gap_sec: float = 0.35) -> Any:
        _, np = _lazy_import_audio()
        first = self._load_audio_array(first_path)
        second = self._load_audio_array(second_path)
        gap = np.zeros(int(self.sample_rate * gap_sec), dtype=np.float32)
        return np.concatenate([first, gap, second], axis=0)

    def _row_audio(self, row: dict[str, Any], candidate_path: str | None = None) -> Any:
        if row.get("prompt_audio_path") and candidate_path:
            return self._combine_audio(str(row["prompt_audio_path"]), candidate_path)
        if row.get("query_audio_path"):
            return self._load_audio_array(str(row["query_audio_path"]))
        if row.get("prompt_audio_path"):
            return self._load_audio_array(str(row["prompt_audio_path"]))
        if candidate_path:
            return self._load_audio_array(candidate_path)
        raise ValueError(f"Row {row.get('id')} does not contain an audio field")


class _BinaryAudioCacheMixin:
    def __init__(self) -> None:
        self._audio_blob_cache: dict[str, tuple[str, str, str]] = {}

    def _audio_blob(self, path: str) -> tuple[str, str, str]:
        if path not in self._audio_blob_cache:
            path_obj = Path(path)
            raw = path_obj.read_bytes()
            mime_type, _ = mimetypes.guess_type(path_obj.name)
            mime_type = mime_type or "audio/wav"
            fmt = path_obj.suffix.lower().lstrip(".") or "wav"
            if fmt == "mpga":
                fmt = "mp3"
            self._audio_blob_cache[path] = (
                base64.b64encode(raw).decode("ascii"),
                mime_type,
                fmt,
            )
        return self._audio_blob_cache[path]


def _extract_ordered_letters(text: str, valid_letters: list[str]) -> list[str]:
    found: list[str] = []
    valid_set = set(valid_letters)
    for char in text.upper():
        if char in valid_set and char not in found:
            found.append(char)
    return found


def _http_json_request(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 180.0,
    retries: int = 5,
    backoff_sec: float = 2.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
            return json.loads(response_body)
        except urllib.error.HTTPError as error:
            payload_text = error.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {error.code} from {url}: {payload_text[:1200]}")
            if error.code in {408, 409, 429, 500, 502, 503, 504} and attempt + 1 < retries:
                time.sleep(backoff_sec * float(2**attempt))
                continue
            raise last_error
        except Exception as error:  # pragma: no cover - network/runtime dependent
            last_error = error
            if attempt + 1 < retries:
                time.sleep(backoff_sec * float(2**attempt))
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Request failed without an exception")


class ClapBenchmarkModel(_AudioCacheMixin, HFAudioBenchmarkModel):
    def __init__(self, model_id: str, *, device: str = "cuda", sample_rate: int = 16000) -> None:
        torch = _lazy_import_torch()
        from transformers import AutoModel, AutoProcessor  # type: ignore

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        actual_sample_rate = int(getattr(self.processor.feature_extractor, "sampling_rate", sample_rate))
        super().__init__(sample_rate=actual_sample_rate)
        self._text_cache: dict[str, Any] = {}

    def _text_embedding(self, text: str) -> Any:
        if text not in self._text_cache:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = self._torch.nn.functional.normalize(features, dim=-1)
            self._text_cache[text] = features[0]
        return self._text_cache[text]

    def _audio_embedding(self, audio: Any) -> Any:
        inputs = self.processor(audios=[audio], sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self.model.get_audio_features(**inputs)
            features = self._torch.nn.functional.normalize(features, dim=-1)
        return features[0]

    def _score_audio(self, audio: Any, text: str) -> float:
        audio_embedding = self._audio_embedding(audio)
        text_embedding = self._text_embedding(text)
        score = self._torch.dot(audio_embedding, text_embedding).item()
        return float(score)

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        prompt = _row_text_context(row)
        if row.get("candidate_audio_paths"):
            scores = []
            for candidate_path in row["candidate_audio_paths"]:
                audio = self._row_audio(row, str(candidate_path))
                scores.append(self._score_audio(audio, prompt))
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            prediction = {"scores": scores, "predicted_index": predicted_index}
            if task_name == "acoustic_plausibility_ranking":
                prediction["predicted_ranking"] = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return prediction

        audio = self._row_audio(row)
        options = row.get("text_options") or row.get("label_space") or []
        scored = [self._score_audio(audio, f"{prompt}\nCandidate answer: {option}") for option in options]
        predicted_index = max(range(len(scored)), key=scored.__getitem__) if scored else 0
        return {"scores": scored, "predicted_index": predicted_index}


class Qwen2AudioBenchmarkModel(_AudioCacheMixin, HFAudioBenchmarkModel):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        sample_rate: int = 16000,
        max_new_tokens: int = 8,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        torch = _lazy_import_torch()
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration  # type: ignore

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _generate_text(self, audio: Any, instruction: str) -> str:
        prompt = f"<|AUDIO|>{instruction}"
        inputs = self.processor(text=prompt, audios=audio, sampling_rate=self.sample_rate, return_tensors="pt")
        prepared = {}
        for key, value in inputs.items():
            prepared[key] = value.to(self.device) if hasattr(value, "to") else value
        with self._torch.no_grad():
            generated = self.model.generate(
                **prepared,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = prepared["input_ids"].shape[1]
        generated = generated[:, prompt_length:]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _score_candidate_audio(self, row: dict[str, Any], candidate_path: str) -> float:
        audio = self._row_audio(row, candidate_path)
        response = self._generate_text(
            audio,
            (
                f"\nYou are judging acoustic match quality.\nContext: {_row_text_context(row)}\n"
                "Rate how well the audio matches the context on a scale from 0 to 100.\n"
                "Respond with only an integer."
            ),
        )
        parsed = _extract_first_int(response)
        return float(parsed if parsed is not None else -1)

    def _choose_text_option(self, row: dict[str, Any]) -> dict[str, Any]:
        audio = self._row_audio(row)
        options = row.get("text_options") or row.get("label_space") or []
        letters = [chr(ord("A") + index) for index in range(len(options))]
        option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
        response = self._generate_text(
            audio,
            (
                f"\nContext: {_row_text_context(row)}\n"
                "Choose the single best option for the audio.\n"
                f"{option_lines}\n"
                "Respond with only the option letter."
            ),
        )
        chosen_letter = _extract_first_letter(response, letters)
        predicted_index = letters.index(chosen_letter) if chosen_letter in letters else 0
        return {"scores": [], "predicted_index": predicted_index, "raw_response": response}

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("candidate_audio_paths"):
            scores = [self._score_candidate_audio(row, str(candidate_path)) for candidate_path in row["candidate_audio_paths"]]
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            prediction = {"scores": scores, "predicted_index": predicted_index}
            if task_name == "acoustic_plausibility_ranking":
                prediction["predicted_ranking"] = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return prediction
        return self._choose_text_option(row)


class UltravoxBenchmarkModel(_AudioCacheMixin, HFAudioBenchmarkModel):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        sample_rate: int = 16000,
        max_new_tokens: int = 8,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        import transformers  # type: ignore

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.pipeline = transformers.pipeline(
            model=model_id,
            trust_remote_code=True,
            device=device,
        )

    def _call_model(self, audio: Any, instruction: str) -> str:
        payload = {
            "audio": audio,
            "sampling_rate": self.sample_rate,
            "turns": [
                {
                    "role": "system",
                    "content": "You are a careful acoustic reasoning assistant. Follow the requested answer format exactly.",
                },
                {
                    "role": "user",
                    "content": instruction,
                },
            ],
        }
        result = self.pipeline(payload, max_new_tokens=self.max_new_tokens)
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, dict):
            for key in ("generated_text", "text", "output_text"):
                if key in result:
                    return str(result[key]).strip()
        return str(result).strip()

    def _score_candidate_audio(self, row: dict[str, Any], candidate_path: str) -> float:
        audio = self._row_audio(row, candidate_path)
        response = self._call_model(
            audio,
            (
                f"Context: {_row_text_context(row)}\n"
                "Rate how well the audio matches the context on a scale from 0 to 100.\n"
                "Respond with only an integer."
            ),
        )
        parsed = _extract_first_int(response)
        return float(parsed if parsed is not None else -1)

    def _choose_text_option(self, row: dict[str, Any]) -> dict[str, Any]:
        audio = self._row_audio(row)
        options = row.get("text_options") or row.get("label_space") or []
        letters = [chr(ord("A") + index) for index in range(len(options))]
        option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
        response = self._call_model(
            audio,
            (
                f"Context: {_row_text_context(row)}\n"
                "Choose the single best option for the audio.\n"
                f"{option_lines}\n"
                "Respond with only the option letter."
            ),
        )
        chosen_letter = _extract_first_letter(response, letters)
        predicted_index = letters.index(chosen_letter) if chosen_letter in letters else 0
        return {"scores": [], "predicted_index": predicted_index, "raw_response": response}

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("candidate_audio_paths"):
            scores = [self._score_candidate_audio(row, str(candidate_path)) for candidate_path in row["candidate_audio_paths"]]
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            prediction = {"scores": scores, "predicted_index": predicted_index}
            if task_name == "acoustic_plausibility_ranking":
                prediction["predicted_ranking"] = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return prediction
        return self._choose_text_option(row)


class SpiritLMBenchmarkModel(_AudioCacheMixin, HFAudioBenchmarkModel):
    def __init__(
        self,
        model_id: str,
        *,
        sample_rate: int = 16000,
        max_new_tokens: int = 8,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        spiritlm_code_root = os.environ.get("SPIRITLM_CODE_ROOT")
        if spiritlm_code_root and spiritlm_code_root not in sys.path:
            sys.path.insert(0, spiritlm_code_root)
        from spiritlm.model.spiritlm_model import ContentType, GenerationInput, OutputModality, Spiritlm  # type: ignore
        from transformers import GenerationConfig  # type: ignore

        self.ContentType = ContentType
        self.GenerationInput = GenerationInput
        self.OutputModality = OutputModality
        self.GenerationConfig = GenerationConfig
        self.model = Spiritlm(model_id)
        self.max_new_tokens = max_new_tokens

    def _generate_text(self, audio: Any, instruction: str) -> str:
        outputs = self.model.generate(
            output_modality=self.OutputModality.TEXT,
            interleaved_inputs=[
                self.GenerationInput(content=instruction, content_type=self.ContentType.TEXT),
                self.GenerationInput(content=audio, content_type=self.ContentType.SPEECH),
            ],
            generation_config=self.GenerationConfig(
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            ),
        )
        return " ".join(
            str(output.content).strip()
            for output in outputs
            if getattr(output, "content_type", None) == self.ContentType.TEXT
        ).strip()

    def _score_candidate_audio(self, row: dict[str, Any], candidate_path: str) -> float:
        audio = self._row_audio(row, candidate_path)
        response = self._generate_text(
            audio,
            (
                f"Context: {_row_text_context(row)}\n"
                "Rate how well this audio matches the context on a scale from 0 to 100.\n"
                "Respond with only an integer."
            ),
        )
        parsed = _extract_first_int(response)
        return float(parsed if parsed is not None else -1)

    def _choose_text_option(self, row: dict[str, Any]) -> dict[str, Any]:
        audio = self._row_audio(row)
        options = row.get("text_options") or row.get("label_space") or []
        letters = [chr(ord("A") + index) for index in range(len(options))]
        option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
        response = self._generate_text(
            audio,
            (
                f"Context: {_row_text_context(row)}\n"
                "Choose the single best option for the audio.\n"
                f"{option_lines}\n"
                "Respond with only the option letter."
            ),
        )
        chosen_letter = _extract_first_letter(response, letters)
        predicted_index = letters.index(chosen_letter) if chosen_letter in letters else 0
        return {"scores": [], "predicted_index": predicted_index, "raw_response": response}

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("candidate_audio_paths"):
            scores = [self._score_candidate_audio(row, str(candidate_path)) for candidate_path in row["candidate_audio_paths"]]
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            prediction = {"scores": scores, "predicted_index": predicted_index}
            if task_name == "acoustic_plausibility_ranking":
                prediction["predicted_ranking"] = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return prediction
        return self._choose_text_option(row)


class CastBenchmarkModel(_AudioCacheMixin, _CausalLMScoringMixin, HFAudioBenchmarkModel):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        torch = _lazy_import_torch()
        from huggingface_hub import hf_hub_download  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        wavtokenizer_code_root = os.environ.get("WAVTOKENIZER_CODE_ROOT")
        if wavtokenizer_code_root and wavtokenizer_code_root not in sys.path:
            sys.path.insert(0, wavtokenizer_code_root)
        from decoder.pretrained import WavTokenizer  # type: ignore

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        codec_repo = os.environ.get("CAST_WAVTOKENIZER_MODEL_ID", "KrauthammerLab/cast-wavtokenizer-24k-40tps")
        codec_ckpt = os.environ.get("WAVTOKENIZER_CHECKPOINT")
        codec_cfg = os.environ.get("WAVTOKENIZER_CONFIG")
        if not codec_ckpt:
            codec_ckpt = hf_hub_download(codec_repo, filename="wavtokenizer_large_unify_600_24k.ckpt")
        if not codec_cfg:
            try:
                codec_cfg = hf_hub_download(codec_repo, filename="config.yaml")
            except Exception:
                codec_cfg = None
        self.codec_sample_rate = 24000
        self.codebook_size = int(os.environ.get("CAST_CODEBOOK_SIZE", "4096"))
        self.wavtokenizer = WavTokenizer.from_pretrained0802(codec_cfg, codec_ckpt).to(self.device)

    def _audio_to_speech_string(self, audio: Any) -> str:
        librosa, _ = _lazy_import_audio()
        audio_24k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=self.codec_sample_rate)
        wav = self._torch.tensor(audio_24k, device=self.device, dtype=self._torch.float32).unsqueeze(0)
        bandwidth_id = self._torch.tensor([0], device=self.device)
        with self._torch.no_grad():
            _, discrete = self.wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        codes = discrete.reshape(-1).detach().cpu().tolist()
        pieces = [f"[Sp{int(code) + 1}]" for code in codes if 0 <= int(code) < self.codebook_size]
        return "".join(pieces)

    def _score_candidate_audio(self, row: dict[str, Any], candidate_path: str) -> float:
        context = _row_text_context(row)
        prefix_parts: list[str] = []
        if row.get("prompt_audio_path"):
            prompt_audio = self._audio_to_speech_string(self._load_audio_array(str(row["prompt_audio_path"])))
            prefix_parts.append(f"[Speech]{prompt_audio}")
        if context:
            prefix_parts.append(f"Context: {context}\nCandidate: [Speech]")
        else:
            prefix_parts.append("[Speech]")
        candidate_audio = self._audio_to_speech_string(self._load_audio_array(candidate_path))
        return self._avg_logprob_for_suffix("\n".join(prefix_parts), candidate_audio)

    def _choose_text_option(self, row: dict[str, Any]) -> dict[str, Any]:
        audio = self._row_audio(row)
        audio_tokens = self._audio_to_speech_string(audio)
        context = _row_text_context(row)
        prefix = f"[Speech]{audio_tokens}"
        if context:
            prefix = f"{prefix}\nContext: {context}\nAnswer:"
        else:
            prefix = f"{prefix}\nAnswer:"
        options = row.get("text_options") or row.get("label_space") or []
        scores = [self._avg_logprob_for_suffix(prefix, f" {option}") for option in options]
        predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
        return {"scores": scores, "predicted_index": predicted_index}

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("candidate_audio_paths"):
            scores = [self._score_candidate_audio(row, str(candidate_path)) for candidate_path in row["candidate_audio_paths"]]
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            prediction = {"scores": scores, "predicted_index": predicted_index}
            if task_name == "acoustic_plausibility_ranking":
                prediction["predicted_ranking"] = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return prediction
        return self._choose_text_option(row)


class TinyWaveBenchmarkModel(_AudioCacheMixin, _CausalLMScoringMixin, HFAudioBenchmarkModel):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        torch = _lazy_import_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        spiritlm_code_root = os.environ.get("SPIRITLM_CODE_ROOT")
        if spiritlm_code_root and spiritlm_code_root not in sys.path:
            sys.path.insert(0, spiritlm_code_root)
        from spiritlm.speech_tokenizer.hubert import spiritlm_hubert  # type: ignore
        from spiritlm.speech_tokenizer.spiritlm_tokenizer import SpiritLMTokenizer  # type: ignore

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        variant = os.environ.get("TINYWAVE_TOKENIZER_VARIANT", "").strip().lower()
        if not variant:
            if "interleaved" in model_id.lower() or "expressive" in model_id.lower():
                variant = "expressive"
            else:
                variant = "base"
        hubert_model = spiritlm_hubert()
        self.base_speech_tokenizer = SpiritLMTokenizer(hubert_model=hubert_model)
        self.allow_base_fallback = os.environ.get("TINYWAVE_ALLOW_BASE_FALLBACK", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if variant == "expressive":
            from spiritlm.speech_tokenizer.f0 import spiritlm_expressive_f0  # type: ignore
            from spiritlm.speech_tokenizer.style_encoder import spiritlm_expressive_style_encoder_w2v2  # type: ignore

            self.speech_tokenizer = SpiritLMTokenizer(
                hubert_model=hubert_model,
                pitch_model=spiritlm_expressive_f0(),
                style_model=spiritlm_expressive_style_encoder_w2v2(),
                hubert_key="hubert",
                pitch_key="pitch",
                style_key="style",
            )
        else:
            self.speech_tokenizer = self.base_speech_tokenizer

    def _audio_to_speech_string(self, audio: Any) -> str:
        tokenizer_device = getattr(getattr(self.speech_tokenizer, "hubert_model", None), "device", None) or self.device
        wav = self._torch.tensor(audio, device=tokenizer_device, dtype=self._torch.float32).view(1, 1, -1)
        with self._torch.no_grad():
            try:
                return str(self.speech_tokenizer.encode_string(wav))
            except Exception:
                if self.speech_tokenizer is not self.base_speech_tokenizer and self.allow_base_fallback:
                    return str(self.base_speech_tokenizer.encode_string(wav))
                raise

    def _score_candidate_audio(self, row: dict[str, Any], candidate_path: str) -> float:
        context = _row_text_context(row)
        prefix_parts: list[str] = []
        if row.get("prompt_audio_path"):
            prompt_audio = self._audio_to_speech_string(self._load_audio_array(str(row["prompt_audio_path"])))
            prefix_parts.append(f"[Speech]{prompt_audio}")
        if context:
            prefix_parts.append(f"{context} [Speech]")
        else:
            prefix_parts.append("[Speech]")
        candidate_audio = self._audio_to_speech_string(self._load_audio_array(candidate_path))
        return self._avg_logprob_for_suffix("\n".join(prefix_parts), candidate_audio)

    def _choose_text_option(self, row: dict[str, Any]) -> dict[str, Any]:
        audio = self._row_audio(row)
        audio_tokens = self._audio_to_speech_string(audio)
        context = _row_text_context(row)
        prefix = f"[Speech]{audio_tokens}"
        if context:
            prefix = f"{prefix}\n{context}\nAnswer:"
        else:
            prefix = f"{prefix}\nAnswer:"
        options = row.get("text_options") or row.get("label_space") or []
        scores = [self._avg_logprob_for_suffix(prefix, f" {option}") for option in options]
        predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
        return {"scores": scores, "predicted_index": predicted_index}

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("candidate_audio_paths"):
            scores = [self._score_candidate_audio(row, str(candidate_path)) for candidate_path in row["candidate_audio_paths"]]
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            prediction = {"scores": scores, "predicted_index": predicted_index}
            if task_name == "acoustic_plausibility_ranking":
                prediction["predicted_ranking"] = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return prediction
        return self._choose_text_option(row)


class Phi4MultimodalBenchmarkModel(_AudioCacheMixin, HFAudioBenchmarkModel):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        sample_rate: int = 16000,
        max_new_tokens: int = 8,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        torch = _lazy_import_torch()
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor  # type: ignore

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        setattr(config, "_attn_implementation", "eager")
        setattr(config, "attn_implementation", "eager")
        if hasattr(config, "embd_layer") and hasattr(config.embd_layer, "model"):
            try:
                setattr(config.embd_layer.model, "_attn_implementation", "eager")
                setattr(config.embd_layer.model, "attn_implementation", "eager")
            except Exception:
                pass
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            use_flash_attention_2=False,
        )
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _generate_text(self, audio: Any, instruction: str) -> str:
        prompt = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"
        inputs = self.processor(text=prompt, audios=[(audio, self.sample_rate)], return_tensors="pt")
        prepared = {}
        for key, value in inputs.items():
            prepared[key] = value.to(self.device) if hasattr(value, "to") else value
        with self._torch.no_grad():
            generated = self.model.generate(
                **prepared,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_logits_to_keep=1,
            )
        prompt_length = prepared["input_ids"].shape[1]
        generated = generated[:, prompt_length:]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _score_candidate_audio(self, row: dict[str, Any], candidate_path: str) -> float:
        audio = self._row_audio(row, candidate_path)
        response = self._generate_text(
            audio,
            (
                f"Context: {_row_text_context(row)}\n"
                "Rate how well the audio matches the context on a scale from 0 to 100.\n"
                "Respond with only an integer."
            ),
        )
        parsed = _extract_first_int(response)
        return float(parsed if parsed is not None else -1)

    def _choose_text_option(self, row: dict[str, Any]) -> dict[str, Any]:
        audio = self._row_audio(row)
        options = row.get("text_options") or row.get("label_space") or []
        letters = [chr(ord("A") + index) for index in range(len(options))]
        option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
        response = self._generate_text(
            audio,
            (
                f"Context: {_row_text_context(row)}\n"
                "Choose the single best option for the audio.\n"
                f"{option_lines}\n"
                "Respond with only the option letter."
            ),
        )
        chosen_letter = _extract_first_letter(response, letters)
        predicted_index = letters.index(chosen_letter) if chosen_letter in letters else 0
        return {"scores": [], "predicted_index": predicted_index, "raw_response": response}

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("candidate_audio_paths"):
            scores = [self._score_candidate_audio(row, str(candidate_path)) for candidate_path in row["candidate_audio_paths"]]
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            prediction = {"scores": scores, "predicted_index": predicted_index}
            if task_name == "acoustic_plausibility_ranking":
                prediction["predicted_ranking"] = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return prediction
        return self._choose_text_option(row)


class MoshiBenchmarkModel(_AudioCacheMixin, HFAudioBenchmarkModel):
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        sample_rate: int = 24000,
        max_new_tokens: int = 24,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        torch = _lazy_import_torch()
        moshi_source_root = os.environ.get("MOSHI_SOURCE_ROOT")
        if moshi_source_root and moshi_source_root not in sys.path:
            sys.path.insert(0, moshi_source_root)
        from moshi.models import LMGen, loaders  # type: ignore

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo(model_id)
        self.mimi = self.checkpoint_info.get_mimi(device=self.device)
        self.text_tokenizer = self.checkpoint_info.get_text_tokenizer()
        self.lm = self.checkpoint_info.get_moshi(device=self.device, dtype=dtype)
        self.lm.eval()
        self.LMGen = LMGen
        self.max_new_tokens = max_new_tokens
        self.sample_rate = int(getattr(self.mimi, "sample_rate", sample_rate))
        self.frame_size = int(getattr(self.mimi, "frame_size", self.sample_rate / self.mimi.frame_rate))
        self.generated_text_ignore_ids = {0, 3}

    def _transcribe_audio(self, audio: Any) -> str:
        _, np = _lazy_import_audio()
        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)
        remainder = int(len(audio) % self.frame_size)
        if remainder:
            audio = np.pad(audio, (0, self.frame_size - remainder), mode="constant")
        wav = self._torch.tensor(audio, device=self.device, dtype=self._torch.float32).view(1, 1, -1)

        text_token_ids: list[int] = []
        lm_gen = self.LMGen(
            self.lm,
            use_sampling=False,
            temp=0.0,
            temp_text=0.0,
            top_k=1,
            top_k_text=1,
        )
        first_frame = True
        with self._torch.no_grad(), lm_gen.streaming(1), self.mimi.streaming(1):
            for chunk in wav.split(self.frame_size, dim=2):
                if chunk.shape[-1] != self.frame_size:
                    continue
                codes = self.mimi.encode(chunk)
                if first_frame:
                    _ = lm_gen.step(codes)
                    first_frame = False
                tokens = lm_gen.step(codes)
                if tokens is None:
                    continue
                one_text = tokens[0, 0]
                token_id = int(one_text.item())
                if token_id in self.generated_text_ignore_ids:
                    continue
                if token_id == int(self.text_tokenizer.eos_id()):
                    break
                text_token_ids.append(token_id)

            flush_steps = max(int(getattr(self.lm, "dep_q", 0)), 1)
            silence = self._torch.zeros((1, 1, self.frame_size), device=self.device, dtype=self._torch.float32)
            for _ in range(flush_steps):
                codes = self.mimi.encode(silence)
                tokens = lm_gen.step(codes)
                if tokens is None:
                    continue
                one_text = tokens[0, 0]
                token_id = int(one_text.item())
                if token_id in self.generated_text_ignore_ids:
                    continue
                if token_id == int(self.text_tokenizer.eos_id()):
                    break
                text_token_ids.append(token_id)

        if not text_token_ids:
            return ""
        return str(self.text_tokenizer.decode(text_token_ids)).strip()

    def _score_candidate_audio(self, row: dict[str, Any], candidate_path: str) -> float:
        audio = self._row_audio(row, candidate_path)
        hypothesis = self._transcribe_audio(audio)
        context = _row_text_context(row)
        return _lexical_similarity(hypothesis, context)

    def _choose_text_option(self, row: dict[str, Any]) -> dict[str, Any]:
        audio = self._row_audio(row)
        hypothesis = self._transcribe_audio(audio)
        context = _row_text_context(row)
        options = row.get("text_options") or row.get("label_space") or []
        scores = []
        for option in options:
            option_text = str(option)
            scores.append(
                0.7 * _lexical_similarity(hypothesis, option_text)
                + 0.3 * _lexical_similarity(hypothesis, f"{context} {option_text}".strip())
            )
        predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
        return {
            "scores": scores,
            "predicted_index": predicted_index,
            "raw_response": hypothesis,
        }

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("candidate_audio_paths"):
            scores = [self._score_candidate_audio(row, str(candidate_path)) for candidate_path in row["candidate_audio_paths"]]
            predicted_index = max(range(len(scores)), key=scores.__getitem__) if scores else 0
            prediction = {"scores": scores, "predicted_index": predicted_index}
            if task_name == "acoustic_plausibility_ranking":
                prediction["predicted_ranking"] = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            return prediction
        return self._choose_text_option(row)


class _APIAudioBenchmarkModel(_BinaryAudioCacheMixin, HFAudioBenchmarkModel):
    def __init__(self, model_id: str, *, max_new_tokens: int = 64) -> None:
        super().__init__()
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def _candidate_letters(self, count: int) -> list[str]:
        return [chr(ord("A") + index) for index in range(count)]

    def _ranking_prompt(self, row: dict[str, Any], candidate_count: int) -> str:
        letters = self._candidate_letters(candidate_count)
        prompt = _row_text_context(row)
        return (
            f"{prompt}\n\n"
            "You will receive candidate audio clips labelled "
            f"{', '.join(letters)}. Rank them from best to worst for the task.\n"
            "Return only the ordered letters, separated by commas, like: A,B,C,D"
        ).strip()

    def _selection_prompt(self, row: dict[str, Any], choice_lines: list[str] | None = None) -> str:
        prompt = _row_text_context(row)
        pieces = [prompt] if prompt else []
        if choice_lines:
            pieces.append("Options:\n" + "\n".join(choice_lines))
        pieces.append("Return only the single best option letter.")
        return "\n\n".join(piece for piece in pieces if piece).strip()

    def _parse_choice(self, text: str, candidate_count: int) -> dict[str, Any]:
        letters = self._candidate_letters(candidate_count)
        ordered = _extract_ordered_letters(text, letters)
        if not ordered:
            ordered = [letters[0]]
        predicted_index = letters.index(ordered[0])
        payload = {
            "predicted_index": predicted_index,
            "raw_response": text,
        }
        if len(ordered) > 1:
            payload["predicted_ranking"] = [letters.index(letter) for letter in ordered]
        return payload

    def _predict_candidate_audio(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        candidate_paths = [str(path) for path in row["candidate_audio_paths"]]
        prompt = self._ranking_prompt(row, len(candidate_paths)) if task_name == "acoustic_plausibility_ranking" else self._selection_prompt(row)
        raw = self._respond_with_audio_prompt(
            prompt=prompt,
            query_audio_path=str(row["prompt_audio_path"]) if row.get("prompt_audio_path") else None,
            candidate_audio_paths=candidate_paths,
        )
        return self._parse_choice(raw, len(candidate_paths))

    def _predict_text_options(self, row: dict[str, Any]) -> dict[str, Any]:
        options = row.get("text_options") or row.get("label_space") or []
        letters = self._candidate_letters(len(options))
        choice_lines = [f"{letter}. {option}" for letter, option in zip(letters, options)]
        raw = self._respond_with_audio_prompt(
            prompt=self._selection_prompt(row, choice_lines),
            query_audio_path=str(row.get("query_audio_path") or row.get("prompt_audio_path") or ""),
            candidate_audio_paths=None,
        )
        return self._parse_choice(raw, len(options))

    def predict(self, task_name: str, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("candidate_audio_paths"):
            return self._predict_candidate_audio(task_name, row)
        return self._predict_text_options(row)

    def _respond_with_audio_prompt(
        self,
        *,
        prompt: str,
        query_audio_path: str | None,
        candidate_audio_paths: list[str] | None,
    ) -> str:
        raise NotImplementedError


class OpenAIAudioBenchmarkModel(_APIAudioBenchmarkModel):
    def __init__(self, model_id: str, *, max_new_tokens: int = 64) -> None:
        super().__init__(model_id=model_id, max_new_tokens=max_new_tokens)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI audio benchmark backend")
        self.api_key = api_key
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def _respond_with_audio_prompt(
        self,
        *,
        prompt: str,
        query_audio_path: str | None,
        candidate_audio_paths: list[str] | None,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if query_audio_path:
            data, _, fmt = self._audio_blob(query_audio_path)
            content.extend(
                [
                    {"type": "text", "text": "Primary audio clip:"},
                    {"type": "input_audio", "input_audio": {"data": data, "format": fmt}},
                ]
            )
        if candidate_audio_paths:
            for index, path in enumerate(candidate_audio_paths):
                letter = chr(ord("A") + index)
                data, _, fmt = self._audio_blob(path)
                content.extend(
                    [
                        {"type": "text", "text": f"Candidate {letter}:"},
                        {"type": "input_audio", "input_audio": {"data": data, "format": fmt}},
                    ]
                )
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are evaluating benchmark audio options. Follow the instruction exactly and return only the requested option letter(s).",
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "max_completion_tokens": self.max_new_tokens,
            "temperature": 0,
        }
        response = _http_json_request(
            url=f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        choice = response["choices"][0]["message"]
        message_content = choice.get("content", "")
        if isinstance(message_content, str):
            return message_content.strip()
        parts: list[str] = []
        for item in message_content or []:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(part.strip() for part in parts if str(part).strip()).strip()


class GeminiAudioBenchmarkModel(_APIAudioBenchmarkModel):
    def __init__(self, model_id: str, *, max_new_tokens: int = 64) -> None:
        super().__init__(model_id=model_id, max_new_tokens=max_new_tokens)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for the Gemini benchmark backend")
        self.api_key = api_key
        self.base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        self.min_interval_sec = float(os.environ.get("GEMINI_MIN_INTERVAL_SEC", "12.5"))
        self._last_request_ts = 0.0

    def _respond_with_audio_prompt(
        self,
        *,
        prompt: str,
        query_audio_path: str | None,
        candidate_audio_paths: list[str] | None,
    ) -> str:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if query_audio_path:
            data, mime_type, _ = self._audio_blob(query_audio_path)
            parts.extend(
                [
                    {"text": "Primary audio clip:"},
                    {"inline_data": {"mime_type": mime_type, "data": data}},
                ]
            )
        if candidate_audio_paths:
            for index, path in enumerate(candidate_audio_paths):
                letter = chr(ord("A") + index)
                data, mime_type, _ = self._audio_blob(path)
                parts.extend(
                    [
                        {"text": f"Candidate {letter}:"},
                        {"inline_data": {"mime_type": mime_type, "data": data}},
                    ]
                )
        payload = {
            "contents": [{"parts": [{"text": "You are evaluating benchmark audio options. Follow the instruction exactly and return only the requested option letter(s)."}, *parts]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.max_new_tokens,
            },
        }
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        response = _http_json_request(
            url=f"{self.base_url}/models/{self.model_id}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            payload=payload,
        )
        self._last_request_ts = time.monotonic()
        parts = response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        texts = [str(part.get("text", "")).strip() for part in parts if isinstance(part, dict) and part.get("text")]
        return "\n".join(text for text in texts if text).strip()


def build_hf_benchmark_model(
    *,
    backend: str,
    model_id: str,
    device: str,
    sample_rate: int = 16000,
    max_new_tokens: int = 8,
) -> HFAudioBenchmarkModel:
    if backend == "openai_audio":
        return OpenAIAudioBenchmarkModel(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if backend == "gemini_audio":
        return GeminiAudioBenchmarkModel(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
    if backend == "clap":
        return ClapBenchmarkModel(model_id=model_id, device=device, sample_rate=sample_rate)
    if backend == "cast_s2s":
        return CastBenchmarkModel(model_id=model_id, device=device, sample_rate=sample_rate)
    if backend == "spiritlm":
        return SpiritLMBenchmarkModel(
            model_id=model_id,
            sample_rate=sample_rate,
            max_new_tokens=max_new_tokens,
        )
    if backend == "tinywave_spirit":
        return TinyWaveBenchmarkModel(model_id=model_id, device=device, sample_rate=sample_rate)
    if backend == "moshi":
        return MoshiBenchmarkModel(
            model_id=model_id,
            device=device,
            sample_rate=sample_rate,
            max_new_tokens=max_new_tokens,
        )
    if backend == "phi4mm":
        return Phi4MultimodalBenchmarkModel(
            model_id=model_id,
            device=device,
            sample_rate=sample_rate,
            max_new_tokens=max_new_tokens,
        )
    if backend == "qwen2_audio":
        return Qwen2AudioBenchmarkModel(
            model_id=model_id,
            device=device,
            sample_rate=sample_rate,
            max_new_tokens=max_new_tokens,
        )
    if backend == "ultravox":
        return UltravoxBenchmarkModel(
            model_id=model_id,
            device=device,
            sample_rate=sample_rate,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported benchmark backend: {backend}")


def evaluate_hf_benchmark_suite(
    *,
    model: HFAudioBenchmarkModel,
    suite_dir: Path,
    output_dir: Path,
    backend: str,
    model_id: str,
    seed: int = 0,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    reference_predictions_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_paths = sorted(path for path in suite_dir.glob("task_*") if path.is_dir())
    summary_tasks: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []

    for task_path in task_paths:
        data_path = task_path / "all.jsonl"
        if not data_path.exists():
            continue
        rows = _read_jsonl(data_path)
        if not rows:
            continue
        task_name = rows[0]["task_name"]
        task_predictions: list[dict[str, Any]] = []
        for row in rows:
            prediction = model.predict(task_name, row)
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
        summary_tasks[task_path.name] = _task_summary_with_bootstrap(
            task_name=task_name,
            records=task_predictions,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        _write_jsonl(output_dir / task_path.name / "predictions.jsonl", task_predictions)

    overall_primary = [float(record["metrics"]["primary_metric"]) for record in prediction_rows]
    summary = {
        "suite_dir": str(suite_dir),
        "output_dir": str(output_dir),
        "backend": backend,
        "model_id": model_id,
        "conditioning": "audio_text",
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "current_metric_family": "choice_accuracy",
        "task_count": len(summary_tasks),
        "overall_primary_metric_mean": math.fsum(overall_primary) / float(len(overall_primary)) if overall_primary else 0.0,
        "tasks": summary_tasks,
    }
    summary["overall_primary_metric_ci"] = _bootstrap_mean_ci(
        overall_primary,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )
    if reference_predictions_dir is not None:
        reference_prediction_rows = _read_jsonl(reference_predictions_dir / "predictions.jsonl")
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


def benchmark_three_tiers(
    *,
    backend: str,
    model_id: str,
    tier_suites: list[TierSuite],
    output_root: Path,
    device: str,
    seed: int = 0,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    model = build_hf_benchmark_model(
        backend=backend,
        model_id=model_id,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    tier_results: dict[str, Any] = {}
    tier_means: list[float] = []
    for tier_suite in tier_suites:
        summary = evaluate_hf_benchmark_suite(
            model=model,
            suite_dir=tier_suite.suite_dir,
            output_dir=tier_suite.output_dir,
            backend=backend,
            model_id=model_id,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
        )
        tier_results[tier_suite.tier_name] = summary
        tier_means.append(float(summary["overall_primary_metric_mean"]))
    macro_mean = sum(tier_means) / float(len(tier_means)) if tier_means else 0.0
    root_summary = {
        "backend": backend,
        "model_id": model_id,
        "device": device,
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "tiers": {
            name: {
                "summary_path": str(Path(payload["output_dir"]) / "summary.json"),
                "overall_primary_metric_mean": payload["overall_primary_metric_mean"],
                "overall_primary_metric_ci": payload["overall_primary_metric_ci"],
                "task_count": payload["task_count"],
            }
            for name, payload in tier_results.items()
        },
        "macro_mean_across_tiers": macro_mean,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(root_summary, handle, indent=2, ensure_ascii=False)
    return root_summary
