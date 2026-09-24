"""Isolated inference HTTP service for media analysis backends."""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from media_analysis_backends import build_backends, cuda_metrics, load_config, package_available


class InferenceRequest(BaseModel):
    path: str
    model: str
    options: dict[str, Any] = Field(default_factory=dict)


def create_app(role: str, config_path: str) -> FastAPI:
    config = load_config(config_path)
    backends = build_backends(config, role)
    app = FastAPI(title=f"Media Analysis {role.title()} Service", docs_url=None, redoc_url=None)
    request_lock = threading.RLock()

    def resolve(request: InferenceRequest):
        backend = backends.get(request.model)
        if backend is None:
            raise HTTPException(400, f"Unknown {role} model: {request.model}")
        path = Path(request.path).resolve()
        if not path.is_file():
            raise HTTPException(404, "Media input does not exist")
        return backend, path

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "role": role,
            "pid": os.getpid(),
            "cuda": cuda_metrics(),
            "packages": {
                "qwen_asr": package_available("qwen_asr"),
                "faster_whisper": package_available("faster_whisper"),
                "transformers": package_available("transformers"),
                "qwen_omni_utils": package_available("qwen_omni_utils"),
            },
        }

    @app.get("/models")
    def models():
        return {"role": role, "models": {name: backend.status() for name, backend in backends.items()}}

    @app.post("/transcribe")
    def transcribe(request: InferenceRequest):
        if role != "asr":
            raise HTTPException(404, "This service does not provide ASR")
        backend, path = resolve(request)
        try:
            with request_lock:
                return backend.transcribe(str(path), request.options)
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/audio-caption")
    def audio_caption(request: InferenceRequest):
        if role != "caption":
            raise HTTPException(404, "This service does not provide audio captioning")
        backend, path = resolve(request)
        try:
            with request_lock:
                return backend.caption(str(path), request.options)
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/video-analyze")
    def video_analyze(request: InferenceRequest):
        if role != "video":
            raise HTTPException(404, "This service does not provide video analysis")
        backend, path = resolve(request)
        try:
            with request_lock:
                return backend.analyze(str(path), request.options)
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/unload")
    def unload(payload: dict[str, Any] | None = None):
        target = str((payload or {}).get("model") or "")
        selected = backends.items() if not target else [(target, backends.get(target))]
        unloaded = []
        for name, backend in selected:
            if backend is not None:
                backend.unload()
                unloaded.append(name)
        return {"ok": True, "unloaded": unloaded}

    @app.post("/load")
    def load_model(payload: dict[str, Any]):
        target = str((payload or {}).get("model") or "")
        backend = backends.get(target)
        if backend is None:
            raise HTTPException(400, f"Unknown {role} model: {target}")
        try:
            with request_lock:
                backend.load()
                return {"ok": True, "model": target, "status": backend.status()}
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local media analysis inference service")
    parser.add_argument("--role", choices=("asr", "caption", "video"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(create_app(args.role, args.config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
