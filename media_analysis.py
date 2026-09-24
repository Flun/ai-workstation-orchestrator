"""Media upload, queue, cache, ffmpeg pipeline, and inference-service control."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import BASE_DIR
from gpu import get_gpus
from media import MEDIA_ALLOWED_EXTENSIONS, find_media_tool, safe_media_filename
from process_mgr import Service


ROOT = Path(BASE_DIR)
SETTINGS_FILE = ROOT / "media_analysis_settings.json"
WORK_DIR = ROOT / "media_analysis_jobs"
UPLOAD_DIR = WORK_DIR / "uploads"
TEMP_DIR = WORK_DIR / "temp"
RESULT_DIR = WORK_DIR / "results"
CACHE_DIR = ROOT / "cache" / "media"
for directory in (UPLOAD_DIR, TEMP_DIR, RESULT_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS: dict[str, Any] = {
    "python": "/home/flux/media-analysis-env/bin/python" if os.name != "nt" else str(ROOT / ".media-analysis-venv" / "Scripts" / "python.exe"),
    "hf_home": "/mnt/main-server-models/model/media-analysis/huggingface" if os.name != "nt" else str(ROOT / "models" / "huggingface"),
    "host": "127.0.0.1",
    "ports": {"asr": 8902, "caption": 8903, "video": 8904},
    "gpus": {"asr": "0", "caption": "1", "video": "0,1"},
    "auto_start": True,
    "parallel": True,
    "unload_after_processing": {"asr": False, "caption": True, "video": False},
    "max_upload_gb": 20,
    "result_retention_days": 30,
    "qwen3_asr": {
        "enabled": True,
        "model": "Qwen/Qwen3-ASR-1.7B",
        "aligner_enabled": True,
        "aligner_model": "Qwen/Qwen3-ForcedAligner-0.6B",
        "batch_size": 8,
        "max_new_tokens": 2048,
    },
    "whisper": {
        "enabled": True,
        "model": "large-v3",
        "compute_type": "float16",
        "download_root": str(Path("/mnt/main-server-models/model/whisper") if os.name != "nt" else ROOT / "models" / "whisper"),
    },
    "raon": {"enabled": False, "model": "KRAFTON/Raon-Speech-9B"},
    "captioner": {
        "enabled": False,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
        "device_map": "auto",
        "max_new_tokens": 160,
    },
    "video": {
        "enabled": False,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "device_map": "auto",
        "max_new_tokens": 160,
        "visual_only": True,
    },
}

router = APIRouter(tags=["media-analysis"])
SERVICES = {
    "asr": Service("media_analysis_asr", stop_timeout=20, kill_timeout=5),
    "caption": Service("media_analysis_caption", stop_timeout=30, kill_timeout=8),
    "video": Service("media_analysis_video", stop_timeout=30, kill_timeout=8),
}
ROLE_LOCKS = {role: threading.RLock() for role in SERVICES}
JOBS: dict[str, dict[str, Any]] = {}
JOB_CANCEL: dict[str, threading.Event] = {}
JOBS_LOCK = threading.RLock()
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="media-analysis")
CACHE_SCHEMA_VERSION = 4
JOB_LIMIT = 128
JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = value
    return output


def load_settings() -> dict[str, Any]:
    try:
        loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            loaded = {}
    except Exception:
        loaded = {}
    return _deep_merge(DEFAULT_SETTINGS, loaded)


def save_settings(value: dict[str, Any]) -> dict[str, Any]:
    merged = _deep_merge(load_settings(), value if isinstance(value, dict) else {})
    merged["host"] = "127.0.0.1"  # inference services are deliberately local-only
    for role in SERVICES:
        port = int(merged.get("ports", {}).get(role, DEFAULT_SETTINGS["ports"][role]))
        if port < 1024 or port > 65535:
            raise HTTPException(400, f"Invalid {role} service port")
        merged["ports"][role] = port
    temporary = SETTINGS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_FILE)
    return merged


if not SETTINGS_FILE.exists():
    save_settings({})


def _service_url(role: str, settings: dict[str, Any] | None = None) -> str:
    settings = settings or load_settings()
    return f"http://127.0.0.1:{int(settings['ports'][role])}"


def _service_health(role: str, timeout: float = 1.0) -> dict[str, Any]:
    service = SERVICES[role]
    info = service.info()
    if info.get("running"):
        info["running_gpus"] = _running_device(role)
    try:
        response = requests.get(f"{_service_url(role)}/health", timeout=timeout)
        response.raise_for_status()
        health = response.json()
        service.phase = "running"
        info.update({"ready": True, "health": health})
    except Exception as exc:
        info.update({"ready": False, "health_error": str(exc)})
    return info


def _normalize_gpu_list(value: Any) -> list[str]:
    """Validate a GPU selection ("0", "0,1", ["0","1"]) against live GPUs."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    parts = [str(item).strip() for item in items if str(item).strip()]
    if not parts:
        return []
    available = {str(gpu.get("index")) for gpu in get_gpus()}
    if available:
        invalid = [part for part in parts if part not in available]
        if invalid:
            raise HTTPException(400, f"존재하지 않는 GPU: {', '.join(invalid)} · 사용 가능: {', '.join(sorted(available, key=int))}")
    return sorted(dict.fromkeys(parts), key=int)


def _running_device(role: str) -> list[str]:
    """GPU indices the running service process actually owns."""
    info = SERVICES[role].info()
    device = info.get("device")
    if device:
        items = device if isinstance(device, (list, tuple)) else str(device).split(",")
        return [str(item).strip() for item in items if str(item).strip()]
    pid = info.get("pid")
    if pid and os.name != "nt":
        # Survives dashboard restarts where the in-memory Service.device is lost.
        try:
            raw = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace")
            for entry in raw.split("\0"):
                if entry.startswith("CUDA_VISIBLE_DEVICES="):
                    return [part.strip() for part in entry.split("=", 1)[1].split(",") if part.strip()]
        except Exception:
            pass
    return []


def start_service(role: str, gpus: Any = None) -> dict[str, Any]:
    if role not in SERVICES:
        raise HTTPException(404, "Unknown media analysis service")
    settings = load_settings()
    python = Path(str(settings.get("python", ""))).expanduser()
    if not python.is_file():
        raise HTTPException(503, f"Media analysis Python environment not installed: {python}. Run setup_media_analysis.sh")
    if SERVICES[role].running():
        current = _running_device(role)
        requested = _normalize_gpu_list(gpus)
        if requested and requested != current:
            # The caller picked different GPUs than the ones the process owns —
            # restart the isolated service so CUDA re-binds to the new cards.
            stop_service(role)
        else:
            return _service_health(role)
    else:
        requested = _normalize_gpu_list(gpus)
    command = [
        str(python), str(ROOT / "media_analysis_service.py"),
        "--role", role, "--config", str(SETTINGS_FILE),
        "--host", "127.0.0.1", "--port", str(settings["ports"][role]),
    ]
    if requested:
        # Remember the explicit pick so lazy auto-start reuses the same GPUs.
        configured = str(settings.get("gpus", {}).get(role, "")).strip()
        if configured != ",".join(requested):
            settings = save_settings({"gpus": {role: ",".join(requested)}})
    devices = requested or [part.strip() for part in str(settings.get("gpus", {}).get(role, "")).split(",") if part.strip()]
    env = {
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(Path(settings.get("hf_home") or DEFAULT_SETTINGS["hf_home"]).expanduser()),
    }
    # CTranslate2 loads cuBLAS/cuDNN through the dynamic loader.  When those
    # libraries are installed from PyPI they live inside this isolated venv.
    if os.name != "nt":
        probe = subprocess.run(
            [str(python), "-c", "import os,nvidia.cublas.lib,nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__)+':'+os.path.dirname(nvidia.cudnn.lib.__file__))"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if probe.returncode == 0 and probe.stdout.strip():
            env["LD_LIBRARY_PATH"] = probe.stdout.strip() + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    SERVICES[role].start(command, cwd=str(ROOT), env=env, device=devices or None)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        status = _service_health(role)
        if status.get("ready"):
            return status
        if not SERVICES[role].running():
            break
        time.sleep(0.4)
    raise HTTPException(503, f"{role} service did not become ready. Check logs/{SERVICES[role].name}.log")


def stop_service(role: str) -> dict[str, Any]:
    if role not in SERVICES:
        raise HTTPException(404, "Unknown media analysis service")
    SERVICES[role].stop()
    return _service_health(role)


def _ensure_service(role: str) -> None:
    if _service_health(role, timeout=0.4).get("ready"):
        return
    settings = load_settings()
    if not settings.get("auto_start", True):
        raise RuntimeError(f"{role} inference service is not running")
    start_service(role)


def _request_service(role: str, endpoint: str, model: str, path: Path, options: dict[str, Any]) -> dict[str, Any]:
    # Keep unload-after-processing from terminating a service while another
    # request for the same role is still inside model.generate().
    with ROLE_LOCKS[role]:
        _ensure_service(role)
        try:
            response = requests.post(
                f"{_service_url(role)}/{endpoint}",
                json={"path": str(path.resolve()), "model": model, "options": options},
                timeout=(20, 60 * 60 * 4),
            )
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith("application/json") else response.text
                raise RuntimeError(str(detail))
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"{role} inference service unavailable: {exc}") from exc


def _unload_if_configured(role: str, settings: dict[str, Any]) -> None:
    if not settings.get("unload_after_processing", {}).get(role, False):
        return
    # A split device_map can leave CUDA allocations alive through framework
    # hooks even after deleting the Python model.  The isolated process is the
    # reliable VRAM ownership boundary, so on-demand mode stops it completely.
    # The next request transparently starts it again via _ensure_service().
    with ROLE_LOCKS[role]:
        try:
            SERVICES[role].stop()
        except Exception:
            pass


def _run(command: list[str], cancel: threading.Event | None = None) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    while process.poll() is None:
        if cancel is not None and cancel.is_set():
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError("작업이 취소되었습니다.")
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed")[-4000:])
    return result


def _duration(path: Path) -> float:
    ffprobe = find_media_tool("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe가 설치되어 있지 않습니다.")
    result = _run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)])
    return max(float(result.stdout.strip()), 0.0)


def _extract_audio(source: Path, target: Path, cancel: threading.Event) -> None:
    ffmpeg = find_media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다.")
    target.parent.mkdir(parents=True, exist_ok=True)
    _run([ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)], cancel)


def _chunk_media(source: Path, directory: Path, chunk_size: int, overlap: int, *, video: bool, cancel: threading.Event) -> list[dict[str, Any]]:
    ffmpeg = find_media_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다.")
    directory.mkdir(parents=True, exist_ok=True)
    total = _duration(source)
    chunks = []
    start = 0.0
    index = 0
    extension = ".mp4" if video else ".wav"
    step = max(1, chunk_size - max(0, overlap))
    while start < total or (index == 0 and total == 0):
        remaining = max(total - start, 0.1)
        # Avoid paying for another 30B-model generation just for a tiny tail.
        # A final window may exceed the configured target by at most 10 seconds.
        tail_tolerance = min(10.0, float(chunk_size) / 3)
        duration = remaining if remaining <= float(chunk_size) + tail_tolerance else float(chunk_size)
        target = directory / f"chunk_{index:04d}{extension}"
        command = [ffmpeg, "-y", "-ss", str(start), "-i", str(source), "-t", str(duration)]
        if video:
            command += ["-vf", "fps=1/2,scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-c:a", "aac", "-b:a", "96k"]
        else:
            command += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
        command.append(str(target))
        _run(command, cancel)
        chunks.append({"path": target, "start": round(start, 3), "end": round(min(start + duration, total), 3)})
        index += 1
        start += step
        if total <= 0:
            break
    return chunks


def _timecode(seconds: float, separator: str = ",") -> str:
    millis = max(0, round(float(seconds) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _subtitle(segments: list[dict[str, Any]], fmt: str) -> str:
    lines = ["WEBVTT", ""] if fmt == "vtt" else []
    usable = [item for item in segments if item.get("end") is not None]
    for index, item in enumerate(usable, 1):
        if fmt == "srt":
            lines.append(str(index))
        separator = "." if fmt == "vtt" else ","
        lines.append(f"{_timecode(item['start'], separator)} --> {_timecode(item['end'], separator)}")
        lines.extend([str(item.get("text", "")).strip(), ""])
    return "\n".join(lines).strip() + "\n"


def _llm_context(result: dict[str, Any]) -> str:
    blocks = []
    transcript = result.get("transcription") or result.get("speech")
    if transcript:
        lines = []
        for item in transcript.get("segments", []):
            end = item.get("end")
            stamp = f"[{item.get('start', 0):.1f}" + (f" - {end:.1f}]" if end is not None else "]")
            lines.append(f"{stamp}\n{item.get('text', '')}")
        blocks.append("<video_transcription>\n" + "\n\n".join(lines) + "\n</video_transcription>")
    caption = result.get("audio_caption")
    if caption:
        lines = [f"[{item['start']:.1f} - {item['end']:.1f}]\n{item['description']}" for item in caption.get("events", [])]
        blocks.append("<audio_context>\n" + "\n\n".join(lines) + "\n</audio_context>")
    video = result.get("video_analysis")
    if video:
        lines = [f"[{item['start']:.1f} - {item['end']:.1f}]\n{item['description']}" for item in video.get("segments", [])]
        blocks.append("<video_context>\n" + "\n\n".join(lines) + "\n</video_context>")
    return "\n\n".join(blocks)


def _cache_file(file_hash: str, payload: dict[str, Any]) -> Path:
    normalized = json.dumps({"schema": CACHE_SCHEMA_VERSION, "payload": payload}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    directory = CACHE_DIR / file_hash
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def _component_cache_file(file_hash: str, component: str, config: dict[str, Any]) -> Path:
    normalized = json.dumps({"schema": CACHE_SCHEMA_VERSION, "config": config}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    directory = CACHE_DIR / file_hash
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{component}-{key}.json"


def _cached_component(file_hash: str, component: str, config: dict[str, Any], producer) -> dict[str, Any]:
    path = _component_cache_file(file_hash, component, config)
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cache_hit"] = True
        return value
    value = producer()
    value["cache_hit"] = False
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def _asr_cache_config(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "asr_model", "language", "task", "timestamps", "word_timestamps", "vad",
        "beam_size", "asr_chunk_size", "overlap", "condition_on_previous_text", "initial_prompt",
    )
    return {key: payload.get(key) for key in keys}


def _caption_cache_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {"model": payload.get("caption_model", "qwen3-omni-captioner"), "chunk_size": payload.get("chunk_size"), "overlap": payload.get("overlap")}


def _video_cache_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {"model": payload.get("video_model", "qwen3-omni-video"), "chunk_size": payload.get("video_chunk_size"), "overlap": payload.get("video_overlap")}


def _seed_component_caches(file_hash: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    mappings = (
        ("asr", _asr_cache_config(payload), result.get("transcription")),
        ("caption", _caption_cache_config(payload), result.get("audio_caption")),
        ("video", _video_cache_config(payload), result.get("video_analysis")),
    )
    for component, config, value in mappings:
        if not value:
            continue
        path = _component_cache_file(file_hash, component, config)
        if not path.exists():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_job(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)
            JOBS[job_id]["updated_at"] = time.time()


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise RuntimeError("작업이 취소되었습니다.")


def _run_asr(audio: Path, payload: dict[str, Any]) -> dict[str, Any]:
    options = {
        "language": payload.get("language", "ko"),
        "task": payload.get("task", "transcribe"),
        "timestamps": payload.get("timestamps", True),
        "word_timestamps": payload.get("word_timestamps", True),
        "vad": payload.get("vad", True),
        "beam_size": payload.get("beam_size", 5),
        "chunk_size": payload.get("asr_chunk_size", 30),
        "condition_on_previous_text": payload.get("condition_on_previous_text", True),
        "initial_prompt": payload.get("initial_prompt", ""),
    }
    return _request_service("asr", "transcribe", payload.get("asr_model", "qwen3-asr"), audio, options)


def _join_transcript_parts(parts: list[str]) -> str:
    """Join chunk transcripts while removing a token overlap at boundaries."""
    merged: list[str] = []
    for part in parts:
        words = str(part or "").strip().split()
        if not words:
            continue
        maximum = min(len(merged), len(words), 40)
        duplicate = 0
        for size in range(maximum, 0, -1):
            if [item.casefold().strip(".,!?\"'…") for item in merged[-size:]] == [item.casefold().strip(".,!?\"'…") for item in words[:size]]:
                duplicate = size
                break
        merged.extend(words[duplicate:])
    return " ".join(merged).strip()


def _run_asr_chunked(audio: Path, job_dir: Path, payload: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
    """Process the complete audio as repeated windows; chunk size is not a file limit."""
    chunk_size = int(payload.get("asr_chunk_size", payload.get("chunk_size", 30)))
    overlap = int(payload.get("overlap", 2))
    total = _duration(audio)
    if total <= chunk_size:
        result = _run_asr(audio, payload)
        result.update({"chunked": False, "chunk_count": 1, "chunk_size": chunk_size, "overlap": overlap})
        return result

    chunks = _chunk_media(audio, job_dir / "asr_chunks", chunk_size, overlap, video=False, cancel=cancel)
    responses: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    for index, chunk in enumerate(chunks):
        _check_cancel(cancel)
        _update_job(payload["job_id"], stage="transcription", progress=20 + round(35 * (index / max(len(chunks), 1))))
        response = _run_asr(chunk["path"], payload)
        responses.append(response)
        transcript_parts.append(str(response.get("text", "")))
        keep_after = float(chunk["start"]) + (overlap / 2 if index else 0)
        response_segments = response.get("segments") or []
        if not response_segments:
            response_segments = [{"start": 0.0, "end": chunk["end"] - chunk["start"], "text": response.get("text", ""), "words": []}]
        for raw in response_segments:
            item = copy.deepcopy(raw)
            local_start = float(item.get("start") or 0.0)
            local_end = item.get("end")
            absolute_start = float(chunk["start"]) + local_start
            absolute_end = float(chunk["start"]) + float(local_end) if local_end is not None else float(chunk["end"])
            if index and (absolute_start + absolute_end) / 2 < keep_after:
                continue
            item["start"] = round(absolute_start, 3)
            item["end"] = round(min(absolute_end, total), 3)
            for word in item.get("words") or []:
                if word.get("start") is not None:
                    word["start"] = round(float(chunk["start"]) + float(word["start"]), 3)
                if word.get("end") is not None:
                    word["end"] = round(float(chunk["start"]) + float(word["end"]), 3)
            segments.append(item)

    first = responses[0]
    processing = sum(float(item.get("processing_seconds") or 0) for item in responses)
    output = {
        "backend": first.get("backend"),
        "model": first.get("model"),
        "language": first.get("language", payload.get("language", "auto")),
        "text": _join_transcript_parts(transcript_parts),
        "segments": segments,
        "word_timestamps": bool(first.get("word_timestamps")),
        "load_seconds": max((float(item.get("load_seconds") or 0) for item in responses), default=0),
        "model_vram_gb": max((float(item.get("model_vram_gb") or 0) for item in responses), default=0),
        "processing_seconds": round(processing, 3),
        "gpu": responses[-1].get("gpu"),
        "chunked": True,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    warnings = list(dict.fromkeys(str(item.get("warning")) for item in responses if item.get("warning")))
    if warnings:
        output["warning"] = " ".join(warnings)
    return output


def _run_caption(audio: Path, job_dir: Path, payload: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
    # Captioner needs nearly a full GPU. A warm Video Omni process is stopped
    # only when this optional analysis is explicitly requested.
    with ROLE_LOCKS["video"]:
        if SERVICES["video"].running():
            stop_service("video")
    chunks = _chunk_media(audio, job_dir / "audio_chunks", int(payload.get("chunk_size", 30)), int(payload.get("overlap", 2)), video=False, cancel=cancel)
    events = []
    metrics = []
    previous = ""
    for index, chunk in enumerate(chunks):
        _check_cancel(cancel)
        _update_job(payload["job_id"], stage="audio_caption_model" if index == 0 else "audio_caption", progress=35 + round(30 * (index / max(len(chunks), 1))))
        response = _request_service("caption", "audio-caption", payload.get("caption_model", "qwen3-omni-captioner"), chunk["path"], {})
        description = str(response.get("description", "")).strip()
        normalized = re.sub(r"\s+", " ", description).casefold()
        if normalized and normalized != previous:
            events.append({"start": chunk["start"], "end": chunk["end"], "description": description})
            previous = normalized
        metrics.append({k: response.get(k) for k in ("load_seconds", "processing_seconds", "model_vram_gb", "gpu")})
    return {
        "backend": payload.get("caption_model", "qwen3-omni-captioner"),
        "speech_summary": None,
        "speakers": None,
        "emotion": [],
        "background_sounds": [],
        "music": None,
        "events": events,
        "description": "\n".join(item["description"] for item in events),
        "timestamps_provided_by_model": False,
        "chunk_timestamps": True,
        "chunk_count": len(chunks),
        "chunk_size": int(payload.get("chunk_size", 30)),
        "overlap": int(payload.get("overlap", 2)),
        "metrics": metrics,
    }


def _run_video(source: Path, job_dir: Path, payload: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
    # The two Omni variants cannot remain resident together on this host.
    with ROLE_LOCKS["caption"]:
        if SERVICES["caption"].running():
            stop_service("caption")
    chunks = _chunk_media(source, job_dir / "video_chunks", int(payload.get("video_chunk_size", 30)), int(payload.get("video_overlap", 2)), video=True, cancel=cancel)
    segments = []
    metrics = []
    for index, chunk in enumerate(chunks):
        _check_cancel(cancel)
        _update_job(payload["job_id"], stage="video_model" if index == 0 else "video_analysis", progress=65 + round(25 * (index / max(len(chunks), 1))))
        response = _request_service("video", "video-analyze", payload.get("video_model", "qwen3-omni-video"), chunk["path"], {})
        segments.append({
            "start": chunk["start"], "end": chunk["end"],
            "description": response.get("description", ""),
            "visual": response.get("description", ""), "audio": [],
        })
        metrics.append({key: response.get(key) for key in ("load_seconds", "processing_seconds", "model_vram_gb", "gpu")})
    return {
        "backend": payload.get("video_model", "qwen3-omni-video"), "segments": segments,
        "chunk_count": len(chunks), "chunk_size": int(payload.get("video_chunk_size", 30)),
        "overlap": int(payload.get("video_overlap", 2)),
        "metrics": metrics,
    }


def _merge_timeline(transcription: dict[str, Any] | None, caption: dict[str, Any] | None, video: dict[str, Any] | None) -> list[dict[str, Any]]:
    if transcription and transcription.get("segments"):
        anchors = transcription["segments"]
    elif video and video.get("segments"):
        anchors = video["segments"]
    elif caption and caption.get("events"):
        anchors = caption["events"]
    else:
        return []
    output = []
    for anchor in anchors:
        start = float(anchor.get("start", 0))
        end = anchor.get("end")
        end_value = float(end) if end is not None else start
        captions = [item["description"] for item in (caption or {}).get("events", []) if item["start"] < end_value and item["end"] > start]
        visuals = [item.get("visual") or item.get("description") for item in (video or {}).get("segments", []) if item["start"] < end_value and item["end"] > start]
        output.append({
            "start": start,
            "end": end,
            "speech": anchor.get("text", "") if transcription else "",
            "visual": " ".join(dict.fromkeys(filter(None, visuals))),
            "audio_events": [],
            "audio_description": " ".join(dict.fromkeys(filter(None, captions))),
            "description": " ".join(filter(None, [anchor.get("text", "") if transcription else "", *visuals, *captions])),
        })
    return output


def _process_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        payload = copy.deepcopy(job["request"])
        source = Path(job["source"])
        file_hash = job["file_hash"]
    cancel = JOB_CANCEL[job_id]
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    started = time.perf_counter()
    try:
        cache_file = _cache_file(file_hash, {key: value for key, value in payload.items() if key not in {"job_id", "filename"}})
        if cache_file.is_file():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
            result["job_id"] = job_id
            result["filename"] = payload.get("filename")
            result["cache_hit"] = True
            _seed_component_caches(file_hash, payload, result)
            _write_result(job_id, result)
            _update_job(job_id, status="completed", stage="completed", progress=100, result=result)
            return

        _update_job(job_id, status="running", stage="audio_extraction", progress=5)
        audio = job_dir / "audio.wav"
        _extract_audio(source, audio, cancel)
        _check_cancel(cancel)
        mode = payload.get("mode", "asr")
        transcription = None
        caption = None
        video = None

        if mode == "asr":
            _update_job(job_id, stage="transcription", progress=20)
            transcription = _cached_component(file_hash, "asr", _asr_cache_config(payload), lambda: _run_asr_chunked(audio, job_dir, payload, cancel))
        elif mode == "detailed" or (mode == "full" and payload.get("full_audio_caption", False)):
            _update_job(job_id, stage="analysis", progress=15)
            if settings.get("parallel", True):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    asr_future = pool.submit(_cached_component, file_hash, "asr", _asr_cache_config(payload), lambda: _run_asr_chunked(audio, job_dir, payload, cancel))
                    caption_future = pool.submit(_cached_component, file_hash, "caption", _caption_cache_config(payload), lambda: _run_caption(audio, job_dir, payload, cancel))
                    transcription = asr_future.result()
                    caption = caption_future.result()
            else:
                transcription = _cached_component(file_hash, "asr", _asr_cache_config(payload), lambda: _run_asr_chunked(audio, job_dir, payload, cancel))
                _check_cancel(cancel)
                caption = _cached_component(file_hash, "caption", _caption_cache_config(payload), lambda: _run_caption(audio, job_dir, payload, cancel))
            if mode == "full" and payload.get("is_video", False):
                # Captioner and Video Omni are both large BF16 models. Keeping
                # Captioner resident while loading Video can OOM two 64GB GPUs.
                # Caption output is complete here, so release that process first.
                with ROLE_LOCKS["caption"]:
                    if SERVICES["caption"].running():
                        stop_service("caption")
                video = _cached_component(file_hash, "video", _video_cache_config(payload), lambda: _run_video(source, job_dir, payload, cancel))
        elif mode == "full" and payload.get("is_video", False):
            # Fast Full mode: dedicated ASR supplies speech while Video Omni
            # handles sparse visual frames. Non-speech Captioner is opt-in.
            _update_job(job_id, stage="transcription", progress=15)
            transcription = _cached_component(file_hash, "asr", _asr_cache_config(payload), lambda: _run_asr_chunked(audio, job_dir, payload, cancel))
            _check_cancel(cancel)
            video = _cached_component(file_hash, "video", _video_cache_config(payload), lambda: _run_video(source, job_dir, payload, cancel))
        elif mode == "caption":
            caption = _cached_component(file_hash, "caption", _caption_cache_config(payload), lambda: _run_caption(audio, job_dir, payload, cancel))
        else:
            raise RuntimeError(f"지원하지 않는 분석 모드: {mode}")

        _check_cancel(cancel)
        duration = _duration(source)
        result = {
            "job_id": job_id,
            "mode": mode,
            "filename": payload.get("filename"),
            "duration": round(duration, 3),
            "transcription": transcription,
            "audio_caption": caption,
            "video_analysis": video,
            "segments": _merge_timeline(transcription, caption, video),
            "cache_hit": False,
            "processing_seconds": round(time.perf_counter() - started, 3),
        }
        result["llm_context"] = _llm_context(result)
        cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _seed_component_caches(file_hash, payload, result)
        _write_result(job_id, result)
        _update_job(job_id, status="completed", stage="completed", progress=100, result=result)
    except Exception as exc:
        status = "cancelled" if cancel.is_set() else "failed"
        _update_job(job_id, status=status, stage=status, error=f"{type(exc).__name__}: {exc}")
    finally:
        for role in ("asr", "caption", "video"):
            _unload_if_configured(role, settings)
        shutil.rmtree(job_dir, ignore_errors=True)
        try:
            source.unlink(missing_ok=True)
        except Exception:
            pass


def _write_result(job_id: str, result: dict[str, Any]) -> None:
    directory = RESULT_DIR / job_id
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    except Exception:
        meta = {}
    with JOBS_LOCK:
        created_at = float((JOBS.get(job_id) or {}).get("created_at") or time.time())
    meta.update({
        "job_id": job_id,
        "filename": result.get("filename"),
        "mode": result.get("mode"),
        "duration": result.get("duration"),
        "created_at": float(meta.get("created_at") or created_at),
        "completed_at": time.time(),
        "permanent": bool(meta.get("permanent", False)),
    })
    result["created_at"] = meta["created_at"]
    result["completed_at"] = meta["completed_at"]
    result["permanent"] = meta["permanent"]
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    segments = (result.get("transcription") or {}).get("segments", [])
    (directory / "transcript.txt").write_text((result.get("transcription") or {}).get("text", ""), encoding="utf-8")
    (directory / "subtitles.srt").write_text(_subtitle(segments, "srt"), encoding="utf-8")
    (directory / "subtitles.vtt").write_text(_subtitle(segments, "vtt"), encoding="utf-8")
    (directory / "llm-context.txt").write_text(result.get("llm_context", ""), encoding="utf-8")


def _result_directory(job_id: str) -> Path:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(400, "Invalid media analysis job id")
    return RESULT_DIR / job_id


def _read_result_meta(directory: Path, result: dict[str, Any]) -> dict[str, Any]:
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        stat = (directory / "result.json").stat()
        meta = {
            "job_id": directory.name,
            "filename": result.get("filename"),
            "mode": result.get("mode"),
            "duration": result.get("duration"),
            "created_at": stat.st_mtime,
            "completed_at": stat.st_mtime,
            "permanent": bool(result.get("permanent", False)),
        }
    return meta


def _cleanup_result_history(settings: dict[str, Any]) -> None:
    retention_days = int(settings.get("result_retention_days", 30))
    if retention_days <= 0:
        return
    cutoff = time.time() - retention_days * 86400
    for directory in RESULT_DIR.iterdir():
        result_path = directory / "result.json"
        if not directory.is_dir() or not JOB_ID_PATTERN.fullmatch(directory.name) or not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            meta = _read_result_meta(directory, result)
            if not meta.get("permanent") and float(meta.get("completed_at") or 0) < cutoff:
                shutil.rmtree(directory)
        except Exception:
            continue


async def _create_job(file: UploadFile, payload: dict[str, Any]) -> dict[str, Any]:
    suffix = Path(file.filename or "media").suffix.lower()
    if suffix not in MEDIA_ALLOWED_EXTENSIONS:
        raise HTTPException(400, "지원하지 않는 audio/video 파일 형식입니다.")
    is_video = (file.content_type or "").startswith("video/") or suffix in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".wmv", ".m4v"}
    if payload.get("mode") == "full" and not is_video:
        raise HTTPException(400, "Full Audio + Video 모드는 비디오 파일이 필요합니다.")
    job_id = uuid.uuid4().hex
    filename = safe_media_filename(file.filename or f"media{suffix}")
    target = UPLOAD_DIR / f"{job_id}_{filename}"
    maximum = int(float(load_settings().get("max_upload_gb", 20)) * 1024**3)
    digest = hashlib.sha256()
    size = 0
    with target.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "업로드 파일이 설정된 최대 크기를 초과했습니다.")
            digest.update(chunk)
            handle.write(chunk)
    payload.update({
        "job_id": job_id,
        "filename": filename,
        "is_video": is_video,
    })
    now = time.time()
    job = {
        "id": job_id, "status": "queued", "stage": "queued", "progress": 0,
        "filename": filename, "size": size, "source": str(target), "file_hash": digest.hexdigest(),
        "request": payload, "created_at": now, "updated_at": now, "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        JOB_CANCEL[job_id] = threading.Event()
        if len(JOBS) > JOB_LIMIT:
            removable = sorted(
                (item for item in JOBS.values() if item.get("status") in {"completed", "failed", "cancelled"}),
                key=lambda item: item.get("updated_at", 0),
            )
            for old in removable[: max(0, len(JOBS) - JOB_LIMIT)]:
                JOBS.pop(old["id"], None)
                JOB_CANCEL.pop(old["id"], None)
    EXECUTOR.submit(_process_job, job_id)
    return {"job_id": job_id, "status": "queued"}


def _payload(
    mode: str, asr_model: str, language: str, task: str, timestamps: bool,
    word_timestamps: bool, vad: bool, beam_size: int, chunk_size: int, overlap: int,
    initial_prompt: str, condition_on_previous_text: bool,
    caption_model: str = "qwen3-omni-captioner", video_model: str = "qwen3-omni-video",
    full_audio_caption: bool = False,
) -> dict[str, Any]:
    if mode not in {"asr", "caption", "detailed", "full"}:
        raise HTTPException(400, "Invalid analysis mode")
    chunk_size = min(max(int(chunk_size), 15), 60)
    return {
        "mode": mode, "asr_model": asr_model, "language": language, "task": task,
        "caption_model": caption_model, "video_model": video_model,
        "full_audio_caption": bool(full_audio_caption),
        "timestamps": timestamps, "word_timestamps": word_timestamps, "vad": vad,
        "beam_size": min(max(int(beam_size), 1), 20),
        "chunk_size": chunk_size, "overlap": min(max(int(overlap), 0), 10),
        "video_chunk_size": chunk_size, "video_overlap": min(max(int(overlap), 0), 10),
        "asr_chunk_size": chunk_size, "initial_prompt": initial_prompt,
        "condition_on_previous_text": condition_on_previous_text,
    }


@router.get("/av-text")
def media_analysis_page():
    return FileResponse(str(ROOT / "media_analysis.html"), headers={"Cache-Control": "no-store"})


@router.get("/api/media-analysis/status")
def media_analysis_status():
    settings = load_settings()
    _cleanup_result_history(settings)
    runtime_models = {}
    for role in SERVICES:
        try:
            response = requests.get(f"{_service_url(role, settings)}/models", timeout=0.6)
            runtime_models[role] = response.json().get("models", {}) if response.ok else {}
        except Exception:
            runtime_models[role] = {}
    return {
        "settings": settings,
        "ffmpeg": bool(find_media_tool("ffmpeg")),
        "services": {role: _service_health(role) for role in SERVICES},
        "available_gpus": [
            {
                "index": gpu.get("index"), "name": gpu.get("name"),
                "vram_total": gpu.get("vram_total"), "vram_free": gpu.get("vram_free"),
                "vram_used": gpu.get("vram_used"),
            }
            for gpu in get_gpus()
        ],
        "models": {
            "qwen3-asr": settings["qwen3_asr"], "whisper-large-v3": settings["whisper"],
            "raon-speech": settings["raon"], "qwen3-omni-captioner": settings["captioner"],
            "qwen3-omni-video": settings["video"],
        },
        "runtime_models": runtime_models,
    }


@router.post("/api/media-analysis/settings")
def media_analysis_save_settings(payload: dict[str, Any]):
    return {"settings": save_settings(payload)}


@router.get("/api/media-analysis/history")
def media_analysis_history(limit: int = 100):
    _cleanup_result_history(load_settings())
    records = []
    for directory in RESULT_DIR.iterdir():
        result_path = directory / "result.json"
        if not directory.is_dir() or not JOB_ID_PATTERN.fullmatch(directory.name) or not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            meta = _read_result_meta(directory, result)
            transcription = result.get("transcription") or {}
            caption = result.get("audio_caption") or {}
            video = result.get("video_analysis") or {}
            preview = transcription.get("text") or caption.get("description") or " ".join(
                str(item.get("description", "")) for item in video.get("segments", [])
            )
            records.append({
                **meta,
                "job_id": directory.name,
                "preview": str(preview).strip()[:240],
                "backend": transcription.get("backend") or caption.get("backend") or video.get("backend"),
                "has_transcription": bool(transcription),
                "has_audio_caption": bool(caption),
                "has_video_analysis": bool(video),
                "size_bytes": sum(path.stat().st_size for path in directory.iterdir() if path.is_file()),
            })
        except Exception:
            continue
    records.sort(key=lambda item: float(item.get("completed_at") or 0), reverse=True)
    return {"items": records[: min(max(limit, 1), 500)], "retention_days": int(load_settings().get("result_retention_days", 30))}


@router.post("/api/media/jobs/{job_id}/permanent")
def media_analysis_set_permanent(job_id: str, payload: dict[str, Any]):
    directory = _result_directory(job_id)
    result_path = directory / "result.json"
    if not result_path.is_file():
        raise HTTPException(404, "분석 결과를 찾지 못했습니다.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    meta = _read_result_meta(directory, result)
    permanent = bool(payload.get("permanent", True))
    meta["permanent"] = permanent
    result["permanent"] = permanent
    (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with JOBS_LOCK:
        if job_id in JOBS and JOBS[job_id].get("result"):
            JOBS[job_id]["result"]["permanent"] = permanent
    return {"ok": True, "job_id": job_id, "permanent": permanent}


@router.delete("/api/media/jobs/{job_id}")
def media_analysis_delete_result(job_id: str):
    directory = _result_directory(job_id)
    with JOBS_LOCK:
        current = JOBS.get(job_id)
        if current and current.get("status") not in {"completed", "failed", "cancelled"}:
            raise HTTPException(409, "실행 중인 작업은 먼저 취소해 주세요.")
        JOBS.pop(job_id, None)
        JOB_CANCEL.pop(job_id, None)
    if not directory.is_dir():
        raise HTTPException(404, "분석 결과를 찾지 못했습니다.")
    shutil.rmtree(directory)
    return {"ok": True, "job_id": job_id}


@router.post("/api/media-analysis/services/{role}/start")
def media_analysis_start_service(role: str, payload: dict[str, Any] | None = None):
    return start_service(role, (payload or {}).get("gpus"))


@router.post("/api/media-analysis/services/{role}/stop")
def media_analysis_stop_service(role: str):
    return stop_service(role)


@router.post("/api/media-analysis/services/{role}/unload")
def media_analysis_unload_service(role: str, payload: dict[str, Any] | None = None):
    if role not in SERVICES:
        raise HTTPException(404, "Unknown media analysis service")
    try:
        response = requests.post(f"{_service_url(role)}/unload", json=payload or {}, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(503, f"{role} service unavailable: {exc}") from exc


@router.post("/api/media-analysis/services/{role}/load")
def media_analysis_load_service(role: str, payload: dict[str, Any] | None = None):
    if role not in SERVICES:
        raise HTTPException(404, "Unknown media analysis service")
    peer = "video" if role == "caption" else "caption" if role == "video" else None
    if peer:
        with ROLE_LOCKS[peer]:
            if SERVICES[peer].running():
                stop_service(peer)
    requested_gpus = (payload or {}).get("gpus")
    if requested_gpus is not None:
        # Honour the GPU the user picked on the card, restarting the service
        # if it currently sits on different cards.
        start_service(role, requested_gpus)
    else:
        _ensure_service(role)
    try:
        response = requests.post(
            f"{_service_url(role)}/load", json=payload or {}, timeout=(20, 60 * 30)
        )
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            raise HTTPException(response.status_code, str(detail))
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(503, f"{role} service unavailable: {exc}") from exc


@router.post("/api/media/transcribe")
async def media_transcribe(
    file: UploadFile = File(...), asr_model: str = Form("qwen3-asr"), language: str = Form("ko"),
    task: str = Form("transcribe"), timestamps: bool = Form(True), word_timestamps: bool = Form(True),
    vad: bool = Form(True), beam_size: int = Form(5), chunk_size: int = Form(30), overlap: int = Form(2),
    initial_prompt: str = Form(""), condition_on_previous_text: bool = Form(True),
):
    return await _create_job(file, _payload("asr", asr_model, language, task, timestamps, word_timestamps, vad, beam_size, chunk_size, overlap, initial_prompt, condition_on_previous_text))


@router.post("/api/media/audio-caption")
async def media_audio_caption(file: UploadFile = File(...), chunk_size: int = Form(30), overlap: int = Form(2)):
    return await _create_job(file, _payload("caption", "qwen3-asr", "auto", "transcribe", False, False, True, 5, chunk_size, overlap, "", True))


@router.post("/api/media/analyze")
async def media_analyze(
    file: UploadFile = File(...), mode: str = Form("detailed"), asr_model: str = Form("qwen3-asr"),
    language: str = Form("ko"), task: str = Form("transcribe"), timestamps: bool = Form(True),
    word_timestamps: bool = Form(True), vad: bool = Form(True), beam_size: int = Form(5),
    chunk_size: int = Form(30), overlap: int = Form(2), initial_prompt: str = Form(""),
    condition_on_previous_text: bool = Form(True), caption_model: str = Form("qwen3-omni-captioner"),
    video_model: str = Form("qwen3-omni-video"), full_audio_caption: bool = Form(False),
):
    return await _create_job(file, _payload(mode, asr_model, language, task, timestamps, word_timestamps, vad, beam_size, chunk_size, overlap, initial_prompt, condition_on_previous_text, caption_model, video_model, full_audio_caption))


@router.get("/api/media/jobs/{job_id}")
def media_analysis_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            result_path = RESULT_DIR / job_id / "result.json"
            if result_path.is_file():
                return {"id": job_id, "status": "completed", "stage": "completed", "progress": 100, "result": json.loads(result_path.read_text(encoding="utf-8"))}
            raise HTTPException(404, "분석 작업을 찾지 못했습니다.")
        public = {key: copy.deepcopy(value) for key, value in job.items() if key not in {"source", "file_hash"}}
    return public


@router.post("/api/media/jobs/{job_id}/cancel")
def media_analysis_cancel(job_id: str):
    event = JOB_CANCEL.get(job_id)
    if event is None:
        raise HTTPException(404, "분석 작업을 찾지 못했습니다.")
    event.set()
    _update_job(job_id, status="cancelling", stage="cancelling")
    return {"ok": True, "job_id": job_id}


@router.get("/api/media/jobs/{job_id}/download/{format_name}")
def media_analysis_download(job_id: str, format_name: str):
    files = {
        "json": ("result.json", "application/json"), "txt": ("transcript.txt", "text/plain"),
        "srt": ("subtitles.srt", "application/x-subrip"), "vtt": ("subtitles.vtt", "text/vtt"),
        "context": ("llm-context.txt", "text/plain"),
    }
    if format_name not in files:
        raise HTTPException(404, "Unknown result format")
    filename, media_type = files[format_name]
    path = RESULT_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(404, "결과 파일이 아직 생성되지 않았습니다.")
    return FileResponse(str(path), filename=f"{job_id}-{filename}", media_type=media_type)
