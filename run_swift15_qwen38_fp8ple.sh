#!/usr/bin/env bash
set -e

sudo docker run -d --rm \
  --name swift15-qwen38-fp8ple \
  --entrypoint vllm \
  --runtime nvidia \
  --gpus '"device=0,1"' \
  --ipc host \
  --cap-add SYS_PTRACE \
  -p 8000:8000 \
  -v /mnt/main-server-models/model/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ-FP8PLE:/model:ro \
  -v qwen38-cmp64-cache:/root/.cache/vllm \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e TRITON_CACHE_DIR=/root/.cache/vllm/triton \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PP_LAYER_PARTITION=24,24 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_FORCE_FP8=1 \
  -e VLLM_GDN_DECODE_KERNEL=triton \
  -e VLLM_BT_POOL=1 \
  -e VLLM_QWEN_SMOOTHIE=1 \
  -e VLLM_QWEN_SMOOTHIE_MIN_SCALE=0.5 \
  -e VLLM_QWEN_SMOOTHIE_SMOOTHNESS=10.0 \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e CUDA_CACHE_PATH=/root/.cache/vllm/cuda \
  -e VLLM_ENABLE_STARTUP_PLAN=0 \
  qwen38-pp2-int4-fp8kv:site42-smoothie-qwen \
  serve /model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --distributed-timeout-seconds 3600 \
  --cpu-distributed-timeout-seconds 3600 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --prefix-match-unit 32 \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --limit-mm-per-prompt '{"video":1,"image":100}' \
  --mamba-cache-mode align \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
