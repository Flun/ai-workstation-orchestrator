#!/usr/bin/env python3
"""Benchmark enabled ASR backends through main_server's public job API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from pathlib import Path

import requests


def run(base_url: str, path: Path, model: str, language: str) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    started = time.perf_counter()
    with path.open("rb") as handle:
        response = requests.post(
            f"{base_url}/api/media/transcribe",
            files={"file": (path.name, handle, mime)},
            data={"asr_model": model, "language": language, "timestamps": "true", "word_timestamps": "true"},
            timeout=3600,
        )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    while True:
        job = requests.get(f"{base_url}/api/media/jobs/{job_id}", timeout=30).json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(1)
    wall = time.perf_counter() - started
    if job["status"] != "completed":
        return {"model": model, "error": job.get("error"), "wall_seconds": wall}
    result = job["result"]
    asr = result["transcription"]
    duration = float(result.get("duration") or 0)
    return {
        "model": model,
        "load_seconds": asr.get("load_seconds"),
        "inference_seconds": asr.get("processing_seconds"),
        "wall_seconds": round(wall, 3),
        "audio_duration": duration,
        "rtf": round(float(asr.get("processing_seconds") or wall) / duration, 4) if duration else None,
        "model_vram_gb": asr.get("model_vram_gb"),
        "process_vram_gb": (asr.get("gpu") or {}).get("process_vram_gb"),
        "language": asr.get("language"),
        "transcript": asr.get("text"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8999")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--models", nargs="+", default=["qwen3-asr", "whisper-large-v3", "raon-speech"])
    args = parser.parse_args()
    if not args.sample.is_file():
        parser.error(f"sample not found: {args.sample}")
    results = [run(args.url.rstrip("/"), args.sample, model, args.language) for model in args.models]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
