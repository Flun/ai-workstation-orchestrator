#!/usr/bin/env python3
"""Convert Swift 1.5's BF16 PLE n-gram table to FP8 e4m3.

Source : ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ.
Output : identical checkpoint, except the 128 PLE shard tensors are rewritten
         BF16 -> FP8 e4m3 with a single global scale, matching the PLE layout
         of Qwen/Qwen3.8-Flash-Next-FP8 that wtdcode/vllm-backport's
         PleOffloadWorker already knows how to load:

    ...layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight -> float8_e4m3fn
    ...layers.1.ple.ple_embedding.ngram_embedding.weight_scale          -> bf16, shape (1,)

Serve with VLLM_PLE_CPU_OFFLOAD=1 and VLLM_PLE_FORCE_FP8=1 in the
qwen38-pp2-int4-fp8kv image.

Idempotent: every file is written to <name>.tmp then atomically renamed, so a
re-run skips completed outputs and never deletes anything (stale .tmp files
from a killed run are simply overwritten).

Run: python convert_fp8_ple.py
"""

import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_SNAP = os.environ.get(
    "SRC_SNAP", "/mnt/main-server-models/model/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ"
)
OUT_DIR = os.environ.get(
    "OUT_DIR", "/mnt/main-server-models/model/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ-FP8PLE"
)
PLE_SUBSTR = "ngram_embedding.shard_"
SHARD0_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
SCALE_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
E4M3_MAX = 448.0


def free_gib(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def main() -> None:
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    st_files = sorted(f for f in os.listdir(SRC_SNAP) if f.endswith(".safetensors"))
    src_bytes = sum(os.path.getsize(os.path.join(SRC_SNAP, f)) for f in st_files)
    # Hard-link unchanged model files. Rewritten PLE files need their own space.
    with open(os.path.join(SRC_SNAP, "model.safetensors.index.json")) as fh:
        src_index = json.load(fh)
    ple_files = {
        f for k, f in src_index["weight_map"].items() if PLE_SUBSTR in k
    }
    need_gib = sum(
        os.path.getsize(os.path.join(SRC_SNAP, f)) for f in ple_files
    ) / 2**30 + 5.0
    avail = free_gib(OUT_DIR)
    print(f"source: {len(st_files)} shards, {src_bytes / 2**30:.2f} GiB", flush=True)
    print(f"free at output: {avail:.1f} GiB (need at most ~{need_gib:.1f})", flush=True)
    if avail < need_gib:
        sys.exit(f"DISK FULL STOP: {avail:.1f} GiB free < {need_gib:.1f} GiB needed")

    # Safety: refuse to touch an output dir that contains anything unexpected.
    unexpected = [
        f
        for f in os.listdir(OUT_DIR)
        if f not in st_files + ["model.safetensors.index.json", "conversion_report.json"]
        and not f.endswith(".tmp")
    ]
    if unexpected:
        sys.exit(f"output dir contains unexpected files, aborting: {unexpected[:5]}")

    # Scale placement: same file as shard_0 (matches weight_map in source).
    wm_src = src_index["weight_map"]
    assert SHARD0_KEY in wm_src, "shard_0 missing from source index"
    scale_file = wm_src[SHARD0_KEY]
    print(f"weight_scale will be added to: {scale_file}", flush=True)

    # Pass A: global amax over all PLE shard tensors (deterministic).
    amax = 0.0
    n_ple = 0
    for f in st_files:
        with safe_open(os.path.join(SRC_SNAP, f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if PLE_SUBSTR in k:
                    t = sf.get_tensor(k)
                    amax = max(amax, t.abs().max().item())
                    del t
                    n_ple += 1
    assert n_ple == 128, f"expected 128 PLE shards, found {n_ple}"
    assert amax > 0, "degenerate amax"
    scale = amax / E4M3_MAX
    # Quantize with the bf16-rounded scale that gets stored, so dequant
    # (q * stored_scale) reproduces the exact quantization used here.
    scale = torch.tensor(scale, dtype=torch.bfloat16).item()
    print(f"amax={amax:.6f}  scale={scale:.8e}  (bf16-rounded: "
          f"{torch.tensor(scale, dtype=torch.bfloat16).item():.8e})", flush=True)

    # Pass B: rewrite PLE-bearing files (and the scale file), copy the rest.
    inv_scale = 1.0 / scale
    for i, f in enumerate(st_files):
        dst = os.path.join(OUT_DIR, f)
        if os.path.exists(dst):
            print(f"[{i + 1}/{len(st_files)}] {f}: exists, skipping", flush=True)
            continue
        src_p = os.path.join(SRC_SNAP, f)
        with safe_open(src_p, framework="pt", device="cpu") as sf:
            md = sf.metadata()
            names = list(sf.keys())
            has_ple = any(PLE_SUBSTR in k for k in names)
            if not has_ple and f != scale_file:
                os.link(src_p, dst)
                print(f"[{i + 1}/{len(st_files)}] {f}: hard-linked unchanged", flush=True)
                continue
            tensors = {}
            n_conv = 0
            for k in names:
                t = sf.get_tensor(k)
                if PLE_SUBSTR in k:
                    t = (t.to(torch.float32) * inv_scale).clamp(
                        -E4M3_MAX, E4M3_MAX
                    ).to(torch.float8_e4m3fn)
                    n_conv += 1
                tensors[k] = t
            if f == scale_file:
                tensors[SCALE_KEY] = torch.tensor([scale], dtype=torch.bfloat16)
            tmp = dst + ".tmp"
            save_file(tensors, tmp, metadata=md)
            os.replace(tmp, dst)
            print(f"[{i + 1}/{len(st_files)}] {f}: {n_conv} PLE tensors -> fp8 "
                  f"({os.path.getsize(dst) / 2**30:.2f} GiB)", flush=True)
            del tensors

    # Pass C: copy all non-safetensors files.
    for f in sorted(os.listdir(SRC_SNAP)):
        if f.endswith(".safetensors"):
            continue
        if f in {"model.safetensors.index.json", "FILE_MANIFEST.json", "VALIDATION.json"}:
            continue
        src_p = os.path.join(SRC_SNAP, f)
        if os.path.isdir(src_p):
            continue
        dst = os.path.join(OUT_DIR, f)
        if not os.path.exists(dst):
            shutil.copyfile(src_p, dst)

    # Pass D: rebuild the index from the ACTUAL output state.
    total_size = 0
    total_params = 0
    ple_fp8 = 0
    out_wm = {}
    for f in sorted(os.listdir(OUT_DIR)):
        if not f.endswith(".safetensors") or f.endswith(".tmp"):
            continue
        with safe_open(os.path.join(OUT_DIR, f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                sl = sf.get_slice(k)
                shape, dt = sl.get_shape(), sl.get_dtype()
                n = 1
                for d in shape:
                    n *= d
                out_wm[k] = f
                total_params += n
                itemsize = {"F64": 8, "U64": 8, "I64": 8, "F32": 4, "U32": 4,
                            "I32": 4, "BF16": 2, "F16": 2, "U16": 2, "I16": 2,
                            "U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1,
                            "F8_E5M2": 1}.get(dt)
                if itemsize is None:
                    sys.exit(f"unhandled dtype {dt} for {k}")
                total_size += int(n * itemsize)
                if PLE_SUBSTR in k:
                    if dt != "F8_E4M3":
                        sys.exit(f"{k}: dtype {dt}, expected F8_E4M3")
                    ple_fp8 += 1
    missing_scale = SCALE_KEY not in out_wm
    if missing_scale:
        sys.exit("weight_scale missing from output — rewrite incomplete")
    idx = {
        "metadata": {"total_parameters": total_params, "total_size": total_size},
        "weight_map": out_wm,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump(idx, fh, indent=2)
        fh.write("\n")

    out_bytes = sum(
        os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)
    )
    report = {
        "source_repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ",
        "source_snapshot": SRC_SNAP,
        "output_dir": OUT_DIR,
        "conversion": "PLE n-gram table (128 shards) BF16 -> FP8 e4m3, single global scale",
        "amax": amax,
        "scale_bf16_stored": torch.tensor([scale], dtype=torch.bfloat16).item(),
        "scale_key": SCALE_KEY,
        "scale_file": out_wm[SCALE_KEY],
        "ple_fp8_tensors": ple_fp8,
        "expected_ple_fp8_tensors": 128,
        "src_bytes": src_bytes,
        "out_bytes": out_bytes,
        "saved_bytes": src_bytes - out_bytes,
        "tool": "convert_swift_ple_fp8.py (adapted from Jon-Nielsen's converter)",
        "ple_fp8_layout_reference": "Qwen/Qwen3.8-Flash-Next-FP8 as loaded by wtdcode/vllm-backport PleOffloadWorker",
        "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(OUT_DIR, "conversion_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    print("CONVERSION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
