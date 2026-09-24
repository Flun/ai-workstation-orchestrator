"""Serving Stats — 실행 중인 서빙 서비스(vLLM/llama.cpp/ComfyUI)의 실시간 지표 요약.

각 서비스의 Prometheus/상태 엔드포인트를 폴링해서 "지금 속도가 얼마나 나오나"를
보는 읽기 전용 대시보드용 API다. 상태를 변경하지 않고, 폴링 실패는 해당 서버만
offline으로 표시할 뿐 요청 전체를 실패시키지 않는다.

속도(t/s)와 Cache HIT%는 카운터 2회 샘플의 델타에서 계산하므로 첫 폴링에는
속도 값이 없고 게이지(대기열·KV 사용량)만 채워진다.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
from fastapi import APIRouter

router = APIRouter()

POLL_TIMEOUT = 1.5          # 단일 서비스 HTTP 요청 타임아웃(초)
MAX_WINDOW = 300.0          # 델타 윈도우 상한(초). 이보다 오래된 이전 샘플은 폐기

_llm_discovery = lambda: []       # app.py가 주입: [{key,label,base_url,engine}]
_comfy_discovery = lambda: {}     # app.py가 주입: {instance: {key,label,service,port}}

_lock = threading.Lock()
_prev: dict[str, dict] = {}       # key -> {"ts": float, "counters": {...}, "buckets": {...}}


def set_llm_discovery(fn):
    global _llm_discovery
    _llm_discovery = fn


def set_comfy_discovery(fn):
    global _comfy_discovery
    _comfy_discovery = fn


# ---------- Prometheus 텍스트 파서 (최소 구현) ----------

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+(\S+)$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> list[tuple[str, dict, float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, _, raw_labels, value = m.groups()
        labels = dict(_LABEL.findall(raw_labels or ""))
        try:
            samples.append((name, labels, float(value)))
        except ValueError:
            continue
    return samples


def _value(samples, name, **match) -> float | None:
    """이름+라벨 부분 일치 중 첫 번째 값."""
    for sname, labels, value in samples:
        if sname != name:
            continue
        if all(str(labels.get(k)) == str(v) for k, v in match.items()):
            return value
    return None


def _sum_value(samples, name, **match) -> float | None:
    total, found = 0.0, False
    for sname, labels, value in samples:
        if sname != name:
            continue
        if all(str(labels.get(k)) == str(v) for k, v in match.items()):
            total += value
            found = True
    return total if found else None


def _histogram(samples, name) -> tuple[list[float], list[float]] | None:
    """_bucket 시리즈를 (정렬된 상한, 누적 개수)로. _sum/_count는 버리지 말 것."""
    buckets: dict[float, float] = {}
    for sname, labels, value in samples:
        if sname != name + "_bucket":
            continue
        le = labels.get("le")
        if le is None:
            continue
        bound = float("inf") if le == "+Inf" else float(le)
        buckets[bound] = buckets.get(bound, 0.0) + value
    if not buckets:
        return None
    bounds = sorted(buckets)
    return bounds, [buckets[b] for b in bounds]


def _percentile(bounds, cumulative, quant) -> float | None:
    """누적 히스토그램에서 분위수(선형 보간). +Inf 버킷이면 직전 상한 반환."""
    total = cumulative[-1] if cumulative else 0.0
    if total <= 0:
        return None
    target = total * quant
    prev_bound, prev_cum = 0.0, 0.0
    for bound, cum in zip(bounds, cumulative):
        if cum >= target:
            if bound == float("inf"):
                return prev_bound
            if cum == prev_cum:
                return bound
            frac = (target - prev_cum) / (cum - prev_cum)
            return prev_bound + (bound - prev_bound) * frac
        prev_bound, prev_cum = bound, cum
    return None


# ---------- 델타 계산 ----------

def _rates(key: str, counters: dict, now: float) -> dict:
    """이전 샘플 대비 초당 증가율. 리스타트(카운터 감소)·갱신 없으면 None."""
    out = {}
    with _lock:
        prev = _prev.get(key)
        _prev[key] = {"ts": now, "counters": counters}
    if not prev:
        return out
    dt = now - prev["ts"]
    if dt <= 0.05 or dt > MAX_WINDOW:
        return out
    for name, value in counters.items():
        old = prev["counters"].get(name)
        if old is None:
            continue
        if value < old:      # 서버 재시작 — 이번 윈도우는 버림
            continue
        out[name] = (value - old) / dt
    return out


# ---------- vLLM ----------

def _poll_vllm(server: dict, now: float) -> dict:
    base = (server.get("base_url") or "").removesuffix("/v1").rstrip("/")
    result = {
        "key": server["key"], "engine": "vLLM", "label": server.get("label") or "vLLM",
        "url": base, "online": False, "model": None, "metrics": {},
    }
    try:
        resp = requests.get(f"{base}/metrics", timeout=POLL_TIMEOUT)
        resp.raise_for_status()
    except Exception as error:
        result["error"] = str(error)
        return result
    samples = parse_prometheus(resp.text)
    if not samples:
        result["error"] = "메트릭 없음 (--disable-logging?)"
        return result
    result["online"] = True

    model = None
    for sname, labels, _ in samples:
        if sname.startswith("vllm:") and labels.get("model_name"):
            model = labels["model_name"]
            break
    result["model"] = model

    counters = {}
    gauge = {}
    def pick(metric, out_key, agg=_sum_value):
        v = agg(samples, metric)
        if v is not None:
            counters[out_key] = v

    pick("vllm:prompt_tokens_total", "prompt_tokens")
    pick("vllm:generation_tokens_total", "gen_tokens")
    pick("vllm:prefix_cache_hits_total", "prefix_hits")
    pick("vllm:prefix_cache_queries_total", "prefix_queries")
    pick("vllm:request_success_total", "requests")
    pick("vllm:num_preemptions_total", "preemptions")
    pick("vllm:request_prefill_time_seconds_sum", "prefill_time")
    pick("vllm:request_decode_time_seconds_sum", "decode_time")
    pick("vllm:spec_decode_num_draft_tokens_total", "draft_tokens")
    pick("vllm:spec_decode_num_accepted_tokens_total", "accepted_tokens")
    pick("vllm:request_prompt_tokens_sum", "req_prompt_sum")
    pick("vllm:request_generation_tokens_sum", "req_gen_sum")

    for metric, out_key in (
        ("vllm:num_requests_running", "running"),
        ("vllm:num_requests_waiting", "waiting"),
        ("vllm:kv_cache_usage_perc", "kv_usage"),
    ):
        v = _value(samples, metric)
        if v is not None:
            gauge[out_key] = v

    rates = _rates(server["key"], counters, now)
    m = dict(gauge)
    m["preemptions"] = counters.get("preemptions", 0.0)
    if rates:
        # Prefill: request_*_sum은 요청 완료 시점에만 갱되므로, 완료 이벤트가
        # 포함된 윈도우에서만 토큰÷시간 비율이 성립한다(0.5s 미만은 정렬 어긋남).
        # Decode: generation_tokens는 스트리밍 중 계속 갱되므로 wall-clock 델타
        # 자체가 실시간 디코딩 속도다 (idle 시 0).
        prefill_t = rates.get("prefill_time")
        if rates.get("prompt_tokens") is not None and prefill_t is not None and prefill_t > 0.5:
            m["prefill_tps"] = rates["prompt_tokens"] / prefill_t
        if rates.get("gen_tokens") is not None:
            m["decode_tps"] = rates["gen_tokens"]
        queries = rates.get("prefix_queries")
        if queries and queries > 1e-9:
            m["cache_hit_pct"] = 100.0 * rates.get("prefix_hits", 0.0) / queries
        draft = rates.get("draft_tokens")
        if draft and draft > 1e-9:
            m["spec_accept_pct"] = 100.0 * rates.get("accepted_tokens", 0.0) / draft
        req_rate = rates.get("requests")
        if req_rate is not None:
            m["req_per_min"] = req_rate * 60.0
    # 히스토그램 분위수는 매 폴링 최신 스냅샷 기준(누적 전체) + 윈도우 델타 둘 다 유용하니
    # 델타 분포가 있으면 델타, 없으면 누적 분포를 쓴다.
    for hist, key_p50, key_p95 in (
        ("vllm:time_to_first_token_seconds", "ttft_p50_s", "ttft_p95_s"),
        ("vllm:request_decode_time_seconds", "req_decode_p50_s", "req_decode_p95_s"),
    ):
        hist_now = _histogram(samples, hist)
        if not hist_now:
            continue
        bounds, cum = hist_now
        hist_key = server["key"] + ":hist:" + hist
        with _lock:
            prev_entry = _prev.get(hist_key) or {}
            _prev[hist_key] = {"cum": list(cum), "bounds": bounds}
        prev_cum, prev_bounds = prev_entry.get("cum"), prev_entry.get("bounds")
        delta_cum = None
        if prev_cum and prev_bounds == bounds:
            delta_cum = [c - p for c, p in zip(cum, prev_cum) if c - p >= 0]
            if len(delta_cum) != len(bounds):
                delta_cum = None
        use_cum = delta_cum if delta_cum and delta_cum[-1] >= 2 else cum
        m[key_p50] = _percentile(bounds, use_cum, 0.5)
        m[key_p95] = _percentile(bounds, use_cum, 0.95)
    result["metrics"] = m
    return result


# ---------- llama.cpp ----------

def _poll_llama(server: dict, now: float) -> dict:
    base = (server.get("base_url") or "").removesuffix("/v1").rstrip("/")
    result = {
        "key": server["key"], "engine": "llama.cpp", "label": server.get("label") or "llama.cpp",
        "url": base, "online": False, "model": None, "metrics": {},
    }
    try:
        props = requests.get(f"{base}/props", timeout=POLL_TIMEOUT).json()
        result["online"] = True
        result["model"] = props.get("model_path") or props.get("model_alias")
        n_ctx = props.get("n_ctx") or props.get("ctx_size")
        if n_ctx:
            result["metrics"]["n_ctx"] = n_ctx
    except Exception as error:
        result["error"] = str(error)
        return result
    try:
        resp = requests.get(f"{base}/metrics", timeout=POLL_TIMEOUT)
        if resp.status_code != 200:
            result["metrics_note"] = "--metrics 없음 — 속도 지표 미제공"
            return result
    except Exception:
        result["metrics_note"] = "--metrics 없음 — 속도 지표 미제공"
        return result
    samples = parse_prometheus(resp.text)
    counters = {}
    gauge = {}
    # 버전별로 llamacpp: / llama: 접두어가 달라서 접미어 매칭으로 흡수한다.
    counter_map = {
        "prompt_tokens_total": "prompt_tokens",
        "tokens_predicted_total": "gen_tokens",
        "predicted_tokens_total": "gen_tokens",
        "prompt_seconds_total": "prefill_time",
        "predicted_seconds_total": "decode_time",
        "tokens_predicted_seconds_total": "decode_time",
        "spec_decode_num_draft_tokens_total": "draft_tokens",
        "spec_decode_num_accepted_tokens_total": "accepted_tokens",
    }
    gauge_map = {
        "requests_processing": "running",
        "requests_deferred": "waiting",
        "kv_cache_usage": "kv_usage",
        "cache_hit_rate": "cache_hit_ratio",
        "cache_hit_ratio": "cache_hit_ratio",
    }
    for sname, _, value in samples:
        body = sname.split(":", 1)[-1]
        out_key = counter_map.get(body) or gauge_map.get(body)
        if out_key is None:
            continue
        bucket = gauge if out_key in gauge_map.values() else counters
        bucket[out_key] = bucket.get(out_key, 0.0) + value
    rates = _rates(server["key"], counters, now) if counters else {}
    m = {k: v for k, v in gauge.items() if k != "cache_hit_ratio"}
    if gauge.get("cache_hit_ratio") is not None:
        m["cache_hit_pct"] = gauge["cache_hit_ratio"] * 100.0
    if rates.get("prompt_tokens") is not None and rates.get("prefill_time", 0) > 0.5:
        m["prefill_tps"] = rates["prompt_tokens"] / rates["prefill_time"]
    if rates.get("gen_tokens") is not None:
        m["decode_tps"] = rates["gen_tokens"]
    draft = rates.get("draft_tokens")
    if draft and draft > 1e-9:
        m["spec_accept_pct"] = 100.0 * rates.get("accepted_tokens", 0.0) / draft
    result["metrics"] = m
    return result


# ---------- ComfyUI ----------

def _poll_comfy(key: str, label: str, port: int, now: float) -> dict:
    result = {
        "key": key, "engine": "ComfyUI", "label": label,
        "url": f"http://127.0.0.1:{port}", "online": False, "model": None, "metrics": {},
    }
    try:
        stats = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=POLL_TIMEOUT).json()
        result["online"] = True
        system = stats.get("system") or {}
        result["model"] = f"v{system.get('ComfyUI_version', '?')}"
        devices = []
        for dev in stats.get("devices") or []:
            total = dev.get("torch_vram_total") or dev.get("vram_total") or 0
            free = dev.get("torch_vram_free") if dev.get("torch_vram_free") is not None else dev.get("vram_free")
            if total and free is not None:
                devices.append({
                    "name": dev.get("name") or dev.get("type") or "GPU",
                    "vram_used_mb": round((total - free) / (1024 * 1024)),
                    "vram_total_mb": round(total / (1024 * 1024)),
                })
        if devices:
            result["metrics"]["devices"] = devices
        result["metrics"]["uptime_s"] = system.get("start_time") and round(time.time() - system["start_time"])
    except Exception as error:
        result["error"] = str(error)
        return result
    try:
        queue = requests.get(f"http://127.0.0.1:{port}/queue", timeout=POLL_TIMEOUT).json()
        result["metrics"]["queue_running"] = len(queue.get("queue_running") or [])
        result["metrics"]["queue_pending"] = len(queue.get("queue_pending") or [])
    except Exception:
        pass
    return result


# ---------- 집계 API ----------

@router.get("/api/serving-stats")
def serving_stats():
    now = time.time()
    llm = list(_llm_discovery() or [])
    comfy = dict(_comfy_discovery() or {})

    tasks = []
    for server in llm:
        if (server.get("engine") or "").lower() == "vllm":
            tasks.append(lambda s=server: _poll_vllm(s, now))
        else:
            tasks.append(lambda s=server: _poll_llama(s, now))
    # ComfyUI는 실행 중인 인스턴스만 폴링 (꺼진 포트에 매번 연결 시도하지 않음)
    for instance, info in comfy.items():
        port = info.get("port")
        if not isinstance(port, int):
            continue
        running = info.get("running", True)
        if not running:
            continue
        tasks.append(lambda k=f"comfy-{instance}", l=info.get("label", instance), p=port:
                     _poll_comfy(k, l, p, now))

    servers = []
    if tasks:
        with ThreadPoolExecutor(max_workers=min(6, len(tasks))) as pool:
            servers = list(pool.map(lambda fn: fn(), tasks))
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "poll_interval_s": 2,
        "servers": servers,
    }
