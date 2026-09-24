#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="${MEDIA_ANALYSIS_ENV:-/home/flux/media-analysis-env}"
MODEL_ROOT="${MEDIA_ANALYSIS_MODEL_ROOT:-/mnt/main-server-models/model/media-analysis}"

python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/media_analysis_requirements.txt"
# Raon currently declares torchaudio while the newest torch wheel can be ahead
# of torchaudio's package metadata. The cu130 wheel imports correctly here;
# avoid letting pip downgrade the shared torch runtime.
"$ENV_DIR/bin/python" -m pip install --no-deps torchaudio==2.11.0
mkdir -p "$MODEL_ROOT/huggingface" "$MODEL_ROOT/whisper"

echo "Media analysis environment ready: $ENV_DIR"
echo "Model cache: $MODEL_ROOT"
echo "Restart main_server, open /av-text, then start the ASR service."
