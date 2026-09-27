"""Check the local Swift FP8-PLE conversion without generating tokens."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


SOURCE = Path("/mnt/main-server-models/model/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ")
OUTPUT = Path("/mnt/main-server-models/model/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ-FP8PLE")
PLE = "ngram_embedding.shard_"
SCALE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
SHARD0 = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"


def indexed_headers(folder: Path) -> dict[str, tuple[str, list[int], str]]:
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    expected = index["weight_map"]
    actual = {}
    for filename in sorted(set(expected.values())):
        with safe_open(folder / filename, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                part = reader.get_slice(key)
                actual[key] = (filename, part.get_shape(), part.get_dtype())
    assert set(actual) == set(expected), (set(expected) - set(actual), set(actual) - set(expected))
    assert all(actual[key][0] == filename for key, filename in expected.items())
    return actual


source = indexed_headers(SOURCE)
output = indexed_headers(OUTPUT)
assert set(output) == set(source) | {SCALE}
ple_count = 0
for key, (src_file, src_shape, src_dtype) in source.items():
    out_file, out_shape, out_dtype = output[key]
    assert src_file == out_file and src_shape == out_shape, key
    if PLE in key:
        assert src_dtype == "BF16" and out_dtype == "F8_E4M3", key
        ple_count += 1
    else:
        assert src_dtype == out_dtype, key
assert ple_count == 128
assert output[SCALE][1:] == ([1], "BF16")

with safe_open(OUTPUT / output[SCALE][0], framework="pt", device="cpu") as reader:
    scale = reader.get_tensor(SCALE).float().item()
with safe_open(SOURCE / source[SHARD0][0], framework="pt", device="cpu") as reader:
    original = reader.get_slice(SHARD0)[:10000].float()
with safe_open(OUTPUT / output[SHARD0][0], framework="pt", device="cpu") as reader:
    restored = reader.get_slice(SHARD0)[:10000].float() * scale
noise = (original - restored).square().mean().sqrt()
signal = original.square().mean().sqrt()
snr_db = 20 * torch.log10(signal / noise).item()
print(f"verified: {len(output)} tensors, {ple_count} FP8 PLE shards, scale={scale}, shard0 sample SNR={snr_db:.2f} dB")
