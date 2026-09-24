"""Lazy-loaded local backends for audio/video understanding.

This module is imported by the isolated inference services, never by the main
web process.  Heavy dependencies are deliberately imported inside ``load`` so
one missing optional backend cannot prevent the service (or main_server) from
starting.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


LANGUAGE_NAMES = {
    "auto": None,
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
}


def package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def cuda_metrics() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "peak_vram_gb": 0.0}
        peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        current = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "peak_vram_gb": round(peak / 1024**3, 3),
            "allocated_vram_gb": round(current / 1024**3, 3),
            "process_vram_gb": process_vram_gb(),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc), "peak_vram_gb": 0.0}


def process_vram_gb() -> float:
    """VRAM owned by this inference process, including non-PyTorch runtimes."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        total_mib = 0.0
        for line in result.stdout.splitlines():
            pid, _, memory = line.partition(",")
            if int(pid.strip()) == os.getpid():
                total_mib += float(memory.strip())
        return round(total_mib / 1024, 3)
    except Exception:
        return 0.0


def reset_cuda_peak() -> None:
    try:
        import torch

        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    except Exception:
        pass


def release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _join_words(words: list[str], language: str) -> str:
    if not words:
        return ""
    if str(language).lower() in {"chinese", "japanese"}:
        return "".join(words)
    text = " ".join(word.strip() for word in words if word.strip())
    return re.sub(r"\s+([,.!?;:，。！？；：])", r"\1", text).strip()


def aligned_items_to_segments(items: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    """Group aligner units into subtitle-sized segments while retaining words."""
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    sentence_end = re.compile(r"[.!?。！？][\"'”’)]*$")
    for item in items:
        word = str(item.get("word") or item.get("text") or "").strip()
        if not word:
            continue
        normalized = {
            "word": word,
            "start": round(float(item.get("start", item.get("start_time", 0.0))), 3),
            "end": round(float(item.get("end", item.get("end_time", 0.0))), 3),
        }
        if current and normalized["start"] - current[-1]["end"] > 1.2:
            segments.append(_make_segment(current, language))
            current = []
        current.append(normalized)
        duration = current[-1]["end"] - current[0]["start"]
        if duration >= 8.0 or sentence_end.search(word):
            segments.append(_make_segment(current, language))
            current = []
    if current:
        segments.append(_make_segment(current, language))
    return segments


def restore_transcript_spacing(segments: list[dict[str, Any]], transcript: str) -> list[dict[str, Any]]:
    """Replace aligner-token spacing with exact slices from the ASR transcript."""
    compact_chars: list[str] = []
    source_indexes: list[int] = []
    for index, char in enumerate(transcript):
        if not char.isspace():
            compact_chars.append(char)
            source_indexes.append(index)
    compact = "".join(compact_chars)
    cursor = 0
    for segment in segments:
        token_text = "".join(item.get("word", "") for item in segment.get("words", []))
        token_text = "".join(char for char in token_text if not char.isspace())
        if not token_text:
            continue
        found = compact.find(token_text, cursor)
        if found < 0:
            continue
        end = found + len(token_text)
        start_source = source_indexes[found]
        end_source = source_indexes[end - 1] + 1
        segment["text"] = transcript[start_source:end_source].strip()
        cursor = end
    return segments


def _make_segment(words: list[dict[str, Any]], language: str) -> dict[str, Any]:
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": _join_words([item["word"] for item in words], language),
        "words": words,
    }


class ASRBackend(ABC):
    name: str
    supports_timestamps = False

    @abstractmethod
    def transcribe(self, path: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError


class AudioCaptionBackend(ABC):
    name: str

    @abstractmethod
    def caption(self, path: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError


class VideoAnalysisBackend(ABC):
    name: str

    @abstractmethod
    def analyze(self, path: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError


class LazyBackend:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.load_seconds: float | None = None
        self.load_error: str | None = None
        self.model_vram_gb: float | None = None
        self._vram_before_load = 0.0
        self.lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def status(self) -> dict[str, Any]:
        return {
            "name": getattr(self, "name", self.__class__.__name__),
            "enabled": bool(self.config.get("enabled", False)),
            "loaded": self.loaded,
            "load_seconds": self.load_seconds,
            "error": self.load_error,
            "model": self.config.get("model"),
            "model_vram_gb": self.model_vram_gb,
        }

    def _ensure_enabled(self) -> None:
        if not self.config.get("enabled", False):
            raise RuntimeError(f"{getattr(self, 'name', 'backend')} backend is disabled")

    def _begin_load(self) -> float:
        self._vram_before_load = process_vram_gb()
        return time.perf_counter()

    def _finish_load(self, started: float) -> None:
        self.load_seconds = round(time.perf_counter() - started, 3)
        self.model_vram_gb = round(max(0.0, process_vram_gb() - self._vram_before_load), 3)
        self.load_error = None

    def unload(self) -> None:
        with self.lock:
            self.model = None
            self.processor = None
            release_cuda()


class Qwen3ASRBackend(LazyBackend, ASRBackend):
    name = "qwen3-asr"
    supports_timestamps = True

    def load(self) -> None:
        self._ensure_enabled()
        if self.model is not None:
            return
        started = self._begin_load()
        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            kwargs: dict[str, Any] = {
                "dtype": torch.bfloat16,
                "device_map": "cuda:0",
                "max_inference_batch_size": int(self.config.get("batch_size", 8)),
                "max_new_tokens": int(self.config.get("max_new_tokens", 2048)),
            }
            if self.config.get("aligner_enabled", True):
                kwargs.update({
                    "forced_aligner": self.config.get("aligner_model", "Qwen/Qwen3-ForcedAligner-0.6B"),
                    "forced_aligner_kwargs": {"dtype": torch.bfloat16, "device_map": "cuda:0"},
                })
            self.model = Qwen3ASRModel.from_pretrained(self.config["model"], **kwargs)
            self._finish_load(started)
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            release_cuda()
            raise RuntimeError(f"Qwen3-ASR load failed: {self.load_error}") from exc

    def transcribe(self, path: str, options: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.load()
            reset_cuda_peak()
            started = time.perf_counter()
            language = LANGUAGE_NAMES.get(str(options.get("language", "auto")).lower(), options.get("language"))
            timestamps_requested = bool(options.get("timestamps", True))
            timestamps_used = timestamps_requested and getattr(self.model, "forced_aligner", None) is not None
            warning = None
            try:
                result = self.model.transcribe(
                    audio=path,
                    context=str(options.get("initial_prompt") or ""),
                    language=language,
                    return_time_stamps=timestamps_used,
                )[0]
            except Exception as exc:
                if not timestamps_used:
                    raise
                # The aligner is intentionally optional: ASR must still work.
                warning = f"ForcedAligner unavailable for this input: {type(exc).__name__}: {exc}"
                result = self.model.transcribe(
                    audio=path,
                    context=str(options.get("initial_prompt") or ""),
                    language=language,
                    return_time_stamps=False,
                )[0]
                timestamps_used = False

            words: list[dict[str, Any]] = []
            if timestamps_used and result.time_stamps is not None:
                raw_items = getattr(result.time_stamps, "items", result.time_stamps)
                for item in raw_items:
                    words.append({
                        "word": str(getattr(item, "text", "")),
                        "start": round(float(getattr(item, "start_time", 0.0)), 3),
                        "end": round(float(getattr(item, "end_time", 0.0)), 3),
                    })
            language_name = str(result.language or language or "unknown")
            segments = restore_transcript_spacing(
                aligned_items_to_segments(words, language_name), str(result.text).strip()
            )
            if not segments and str(result.text).strip():
                segments = [{"start": 0.0, "end": None, "text": str(result.text).strip(), "words": []}]
            return {
                "backend": self.name,
                "model": self.config["model"],
                "language": language_name,
                "text": str(result.text).strip(),
                "segments": segments,
                "word_timestamps": timestamps_used,
                "warning": warning,
                "load_seconds": self.load_seconds,
                "model_vram_gb": self.model_vram_gb,
                "processing_seconds": round(time.perf_counter() - started, 3),
                "gpu": cuda_metrics(),
            }


class WhisperBackend(LazyBackend, ASRBackend):
    name = "whisper-large-v3"
    supports_timestamps = True

    def load(self) -> None:
        self._ensure_enabled()
        if self.model is not None:
            return
        started = self._begin_load()
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.config.get("model", "large-v3"),
                device="cuda",
                compute_type=self.config.get("compute_type", "float16"),
                download_root=self.config.get("download_root") or None,
            )
            self._finish_load(started)
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            release_cuda()
            raise RuntimeError(f"Whisper load failed: {self.load_error}") from exc

    def transcribe(self, path: str, options: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.load()
            reset_cuda_peak()
            started = time.perf_counter()
            language = str(options.get("language", "auto") or "auto").lower()
            segments_iter, info = self.model.transcribe(
                path,
                language=None if language == "auto" else language,
                task=str(options.get("task", "transcribe")),
                beam_size=max(1, int(options.get("beam_size", 5))),
                vad_filter=bool(options.get("vad", True)),
                word_timestamps=bool(options.get("word_timestamps", options.get("timestamps", True))),
                condition_on_previous_text=bool(options.get("condition_on_previous_text", True)),
                initial_prompt=str(options.get("initial_prompt") or "") or None,
                chunk_length=int(options.get("chunk_size", 30)),
            )
            output_segments = []
            text_parts = []
            for segment in segments_iter:
                text = str(segment.text).strip()
                text_parts.append(text)
                words = []
                for word in getattr(segment, "words", None) or []:
                    words.append({
                        "word": str(word.word).strip(),
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "probability": round(float(word.probability), 4),
                    })
                output_segments.append({
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": text,
                    "words": words,
                    "confidence": None,
                })
            return {
                "backend": self.name,
                "model": self.config.get("model", "large-v3"),
                "language": getattr(info, "language", language),
                "language_probability": getattr(info, "language_probability", None),
                "text": " ".join(text_parts).strip(),
                "segments": output_segments,
                "word_timestamps": bool(options.get("word_timestamps", options.get("timestamps", True))),
                "load_seconds": self.load_seconds,
                "model_vram_gb": self.model_vram_gb,
                "processing_seconds": round(time.perf_counter() - started, 3),
                "gpu": cuda_metrics(),
            }


class RaonSpeechBackend(LazyBackend, ASRBackend):
    name = "raon-speech"
    supports_timestamps = False

    def load(self) -> None:
        self._ensure_enabled()
        if self.model is not None:
            return
        started = self._begin_load()
        try:
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            model_id = self.config.get("model", "KRAFTON/Raon-Speech-9B")
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            pipeline_class = get_class_from_dynamic_module(
                "modeling_raon.RaonPipeline", model_id, revision=getattr(config, "_commit_hash", None)
            )
            self.model = pipeline_class(model_id, device="cuda", dtype="bfloat16")
            self._finish_load(started)
        except Exception as exc:
            self.model = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            release_cuda()
            raise RuntimeError(f"Raon-Speech backend unavailable: {self.load_error}") from exc

    def transcribe(self, path: str, options: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.load()
            reset_cuda_peak()
            started = time.perf_counter()
            text = str(self.model.stt(path)).strip()
            return {
                "backend": self.name,
                "model": self.config.get("model"),
                "language": options.get("language", "auto"),
                "text": text,
                "segments": [{"start": 0.0, "end": None, "text": text, "words": []}],
                "word_timestamps": False,
                "warning": "Raon-Speech does not provide timestamps; transcript only.",
                "load_seconds": self.load_seconds,
                "model_vram_gb": self.model_vram_gb,
                "processing_seconds": round(time.perf_counter() - started, 3),
                "gpu": cuda_metrics(),
            }


class QwenOmniCaptionBackend(LazyBackend, AudioCaptionBackend):
    name = "qwen3-omni-captioner"

    def load(self) -> None:
        self._ensure_enabled()
        if self.model is not None:
            return
        started = self._begin_load()
        try:
            import torch
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

            model_id = self.config.get("model", "Qwen/Qwen3-Omni-30B-A3B-Captioner")
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map=self.config.get("device_map", "auto"),
                low_cpu_mem_usage=True,
            )
            # These services only return text.  Keeping the speech-generation
            # (Talker) branch wastes VRAM and slows text-only inference.
            if hasattr(self.model, "disable_talker"):
                self.model.disable_talker()
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
            self._finish_load(started)
        except Exception as exc:
            self.model = None
            self.processor = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            release_cuda()
            raise RuntimeError(f"Qwen3-Omni Captioner load failed: {self.load_error}") from exc

    def caption(self, path: str, options: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.load()
            reset_cuda_peak()
            started = time.perf_counter()
            from qwen_omni_utils import process_mm_info

            instruction = str(options.get("prompt") or self.config.get("prompt") or (
                "Describe this audio in Korean in no more than five concise sentences. "
                "Focus on speaker traits, emotion, background noise, sound effects, and music. "
                "Do not repeat a detailed transcript and do not invent timestamps or recording conditions."
            ))
            conversation = [{"role": "user", "content": [
                {"type": "audio", "audio": path}, {"type": "text", "text": instruction},
            ]}]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
            inputs = self.processor(
                text=prompt, audio=audios, return_tensors="pt", padding=True, use_audio_in_video=False
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            generated, _ = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                max_new_tokens=int(options.get("max_new_tokens", self.config.get("max_new_tokens", 160))),
            )
            text = self.processor.batch_decode(
                generated.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            return {
                "backend": self.name,
                "model": self.config.get("model"),
                "description": text,
                "timestamps_provided": False,
                "load_seconds": self.load_seconds,
                "model_vram_gb": self.model_vram_gb,
                "processing_seconds": round(time.perf_counter() - started, 3),
                "gpu": cuda_metrics(),
            }


class QwenOmniVideoBackend(LazyBackend, VideoAnalysisBackend):
    name = "qwen3-omni-video"

    def load(self) -> None:
        self._ensure_enabled()
        if self.model is not None:
            return
        started = self._begin_load()
        try:
            import torch
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

            model_id = self.config.get("model", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map=self.config.get("device_map", "auto"),
                low_cpu_mem_usage=True,
            )
            if hasattr(self.model, "disable_talker"):
                self.model.disable_talker()
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
            self._finish_load(started)
        except Exception as exc:
            self.model = None
            self.processor = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            release_cuda()
            raise RuntimeError(f"Qwen3-Omni video backend unavailable: {self.load_error}") from exc

    def analyze(self, path: str, options: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.load()
            reset_cuda_peak()
            started = time.perf_counter()
            from qwen_omni_utils import process_mm_info

            instruction = str(options.get("prompt") or (
                "Describe only the important visible actions and scene changes in Korean. "
                "Use no more than six concise sentences. Do not transcribe speech or invent timestamps."
            ))
            # qwen-omni-utils 0.0.9 is incompatible with the current
            # librosa/torchvision video-audio readers.  The main pipeline has
            # a dedicated ASR source of truth, so decode sparse visual frames
            # with ffmpeg and feed them as a video frame sequence.  This also
            # avoids running the Omni audio branch twice in Full mode.
            with tempfile.TemporaryDirectory(prefix="qwen-omni-frames-") as frame_dir:
                pattern = str(Path(frame_dir) / "frame_%05d.jpg")
                decoded = subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", path, "-an", "-vsync", "0", "-q:v", "3", pattern],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if decoded.returncode != 0:
                    raise RuntimeError(f"Video frame extraction failed: {decoded.stderr[-1000:]}")
                frames = [str(item) for item in sorted(Path(frame_dir).glob("frame_*.jpg"))]
                if not frames:
                    raise RuntimeError("Video contains no decodable frames")
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames, "sample_fps": 0.5, "raw_fps": 0.5},
                        {"type": "text", "text": instruction},
                    ],
                }]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                _, images, videos = process_mm_info(conversation, use_audio_in_video=False)
                inputs = self.processor(
                    text=prompt,
                    audio=None,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=False,
                )
                inputs = inputs.to(self.model.device).to(self.model.dtype)
                generated, _ = self.model.generate(
                    **inputs,
                    thinker_return_dict_in_generate=True,
                    max_new_tokens=int(options.get("max_new_tokens", self.config.get("max_new_tokens", 160))),
                )
                text = self.processor.batch_decode(
                    generated.sequences[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
            return {
                "backend": self.name,
                "model": self.config.get("model"),
                "description": text,
                "load_seconds": self.load_seconds,
                "model_vram_gb": self.model_vram_gb,
                "processing_seconds": round(time.perf_counter() - started, 3),
                "gpu": cuda_metrics(),
            }


def build_backends(config: dict[str, Any], role: str) -> dict[str, LazyBackend]:
    if role == "asr":
        return {
            "qwen3-asr": Qwen3ASRBackend(config.get("qwen3_asr", {})),
            "whisper-large-v3": WhisperBackend(config.get("whisper", {})),
            "raon-speech": RaonSpeechBackend(config.get("raon", {})),
        }
    if role == "caption":
        return {"qwen3-omni-captioner": QwenOmniCaptionBackend(config.get("captioner", {}))}
    if role == "video":
        return {"qwen3-omni-video": QwenOmniVideoBackend(config.get("video", {}))}
    raise ValueError(f"Unknown service role: {role}")


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}
