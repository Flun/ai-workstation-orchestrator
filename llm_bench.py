"""LLM Benchmark — OpenAI 호환 LLM 서버의 prefill(pp)/decode 속도를 측정한다.

카드(슬롯) 단위로 벤치마크 대상 API를 등록하고, 짧은 테스트 / 64k 단계 상세 /
150k·200k·252k 장문 벤치마크를 실행한다. 결과는 llm_bench.json에 메타데이터
(테스트 시각, 서빙 엔진, 모델, 서버 최대 컨텍스트, API 주소)와 함께 누적된다.

측정 방식 (POST /v1/completions 기준, 단계당 요청 1회):
- pp·TTFT·decode : 유니크 프롬프트(선두 랜덤 태그 → prefix cache 절대 미스)로
  max_tokens=128 스트리밍을 1회 보내 첫 토큰까지의 시간(TTFT)을 프리필 시간으로
  보고 pp = prompt_tokens/TTFT, 나머지 구간에서 decode tok/s를 계산한다
  (llama.cpp pp 로그와 같은 "실제 프리필 처리량" 개념, 서버 로그는 참조하지 않음).
- cold : 서버 /metrics의 생애 누적 요청/토큰 카운터가 0이면 그 실행의 pp가
  진짜 첫 가동(cold) 프리필이다(server_cold=true로 기록). 그 외에는 전부 일반
  프리필이며, 과거처럼 동일 단계를 반복 측정하지 않는다.
- N명 동시(users>1): 같은 길이의 유니크 프롬프트 N개를 동시에 스트리밍해
  사용자별 평균 decode/pp와 총합 decode tok/s(서버 전체 처리량)를 별도 측정한다.
프롬프트 토큰 수는 서버의 /tokenize 엔드포인트(llama.cpp/vLLM 모두 제공)로 보정하고,
없으면 글자수 근사치를 쓴다. 보고되는 속도는 서버가 응답한 usage.prompt_tokens 기준.
"""

import json
import os
import random
import re
import threading
import time
import uuid
from datetime import datetime

import requests
from fastapi import APIRouter, HTTPException

from config import BASE_DIR

router = APIRouter(prefix="/api/llm-bench", tags=["llm-bench"])

STORE_FILE = os.path.join(BASE_DIR, "llm_bench.json")
MAX_RESULTS = 300

# 벤치마크 종류별 측정 단계(프롬프트 토큰 수).
KIND_STAGES = {
    "quick": [2048],
    "detailed": [1024, 2048, 4096, 8192, 16384, 32768, 65536],
    "long150k": [150_000],
    "long200k": [200_000],
    "long252k": [252_000],
}
KIND_LABELS = {
    "quick": "짧은 벤치마크",
    "detailed": "상세 벤치마크 (1k→64k)",
    "long150k": "150k 장문",
    "long200k": "200k 장문",
    "long252k": "252k 장문",
}
GEN_TOKENS = 128  # decode 측정에 쓰는 생성 토큰 수

_store_lock = threading.Lock()   # llm_bench.json 읽기/쓰기
_run_lock = threading.Lock()     # 동시에 벤치마크 1개만 (결과의 신뢰성)
_jobs = {}
_jobs_lock = threading.Lock()

# app.py가 main_server에서 직접 실행한 llama/vllm 서비스 목록을 알려주는 콜백.
_service_discovery = None


def set_service_discovery(fn):
    global _service_discovery
    _service_discovery = fn


# ---------- 저장소 ----------

def _load_store():
    if not os.path.isfile(STORE_FILE):
        return {"cards": [], "results": []}
    try:
        with open(STORE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {"cards": [], "results": []}
    if not isinstance(data, dict):
        return {"cards": [], "results": []}
    data.setdefault("cards", [])
    data.setdefault("results", [])
    return data


def _save_store(data):
    tmp = STORE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, STORE_FILE)


def _now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------- OpenAI 호환 서버 유틸 ----------

def normalize_base(api_url):
    """http://127.0.0.1:8000 같은 입력을 http://127.0.0.1:8000/v1 형태로 정규화."""
    url = str(api_url or "").strip().rstrip("/")
    if not url:
        raise HTTPException(400, "API 주소를 입력하세요")
    if "://" not in url:
        url = "http://" + url
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _origin(base):
    scheme, _, rest = base.partition("://")
    host = rest.split("/", 1)[0]
    return f"{scheme}://{host}"


_FILLER = (
    "The quick brown fox jumps over the lazy dog while the inference engine "
    "streams tokens across a long context window and measures throughput. "
)


def _count_tokens(origin, model, text):
    """서버 /tokenize로 토큰 수 측정. vLLM/llama.cpp 페이로드 둘 다 시도."""
    for payload in ({"model": model, "prompt": text}, {"content": text}):
        try:
            r = requests.post(origin + "/tokenize", json=payload, timeout=60)
            if r.status_code == 200:
                j = r.json()
                n = j.get("token_length") or j.get("count")
                if isinstance(n, int) and n > 0:
                    return n
                toks = j.get("tokens")
                if isinstance(toks, list) and toks:
                    return len(toks)
                # 키가 안 맞아서 빈 결과가 나온 경우(예: llama.cpp에 vLLM 페이로드) 다음 시도.
                continue
        except (requests.RequestException, ValueError):
            continue
    return None


def _build_prompt(origin, model, target_tokens, rng):
    """target_tokens 분량의 유니크 프롬프트 생성. (text, 예상 토큰수)"""
    tag = f"[bench-{rng.randrange(1 << 48):012x}] "
    unit = tag + _FILLER
    chars_per_tok = 3.7
    n = max(1, int(target_tokens * chars_per_tok / len(unit)))
    text = unit * n
    approx = _count_tokens(origin, model, text)
    if approx:
        for _ in range(3):
            if abs(approx - target_tokens) <= max(32, target_tokens * 0.02):
                break
            n = max(1, int(round(n * target_tokens / float(approx))))
            text = unit * n
            approx = _count_tokens(origin, model, text) or approx
    else:
        approx = int(len(text) / chars_per_tok)
    return text, approx


def _stop_params(engine, gen_tokens):
    """엔진별로 early-stop(EOS)을 막아 gen_tokens를 채우게 하는 파라미터."""
    if engine == "llama.cpp":
        return {"min_tokens": gen_tokens}
    return {"ignore_eos": True}


def _completions(base, model, prompt, max_tokens, engine, stream=False, extra=None, timeout=3600.0):
    """POST {base}/completions. (응답본문 스트리밍이면 제너레이터, 시간을 되돌려줌)"""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    body.update(_stop_params(engine, max_tokens))
    if extra:
        body.update(extra)
    url = base + "/completions"
    kwargs = {"json": body, "timeout": (15, timeout)}
    r = requests.post(url, stream=stream, **kwargs)
    if r.status_code >= 400 and engine != "unknown":
        # 엔진별 파라미터를 거부하는 서버면 빠뜨리고 1회 재시도.
        body.pop("min_tokens", None)
        body.pop("ignore_eos", None)
        r = requests.post(url, stream=stream, **kwargs)
    r.raise_for_status()
    return r


def _measure_plain(base, model, prompt, max_tokens, engine):
    """비스트리밍 1회 요청 → (elapsed_s, prompt_tokens, completion_tokens)."""
    t0 = time.perf_counter()
    r = _completions(base, model, prompt, max_tokens, engine, stream=False)
    elapsed = time.perf_counter() - t0
    j = r.json()
    usage = j.get("usage") or {}
    p_tok = usage.get("prompt_tokens")
    c_tok = usage.get("completion_tokens")
    return elapsed, p_tok, c_tok


def _measure_stream(base, model, prompt, gen_tokens, engine, cancel_event=None, t0=None):
    """스트리밍 1회 요청 → dict(ttft, total, prompt_tokens, completion_tokens).

    chunk 하나 = 대략 1토큰으로 보고 first-token 시각과 전체 시각을 잰다.
    t0를 외부에서 넘기면(동시 배치) 모든 요청이 같은 시작 시각 기준으로 측정된다.
    """
    if t0 is None:
        t0 = time.perf_counter()
    r = _completions(
        base, model, prompt, gen_tokens, engine, stream=True,
        extra={"stream_options": {"include_usage": True}},
    )
    ttft = None
    chunks = 0
    usage = {}
    for line in r.iter_lines(decode_unicode=True):
        if cancel_event is not None and cancel_event.is_set():
            r.close()
            raise TimeoutError("취소됨")
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            j = json.loads(payload)
        except ValueError:
            continue
        if isinstance(j.get("usage"), dict):
            usage.update(j["usage"])
        for ch in j.get("choices") or []:
            piece = ch.get("text")
            if piece is None:
                delta = ch.get("delta") or {}
                piece = delta.get("content")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or chunks
    return {
        "ttft": ttft,
        "total": total,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "chunks": chunks,
    }


def probe_server(api_url, model_hint=None):
    """서버 연결/엔진/모델/최대 컨텍스트 감지. HTTPException 대신 dict를 돌려준다."""
    base = normalize_base(api_url)
    origin = _origin(base)
    out = {
        "ok": False,
        "api_url": base,
        "engine": "unknown",
        "models": [],
        "model": None,
        "max_context": None,
        "max_context_source": None,
        "server_header": None,
        "error": None,
    }
    try:
        r = requests.get(base + "/models", timeout=10)
        r.raise_for_status()
        out["server_header"] = r.headers.get("server")
        data = (r.json() or {}).get("data") or []
        models = []
        max_ctx = None
        for item in data:
            mid = item.get("id")
            if mid:
                models.append(mid)
            for key in ("max_model_len", "n_ctx", "max_context"):
                value = item.get(key)
                if isinstance(value, int) and value > 0:
                    max_ctx = value
                    out["max_context_source"] = "/v1/models." + key
                    break
        out["models"] = models
        out["model"] = model_hint if model_hint in models else (models[0] if models else None)
        out["max_context"] = max_ctx
        out["ok"] = True
    except requests.RequestException as exc:
        out["error"] = f"/v1/models 요청 실패: {exc.__class__.__name__}: {exc}"
        return out

    # llama.cpp 전용 /props — n_ctx(슬롯 컨텍스트)와 모델 경로를 준다.
    try:
        pr = requests.get(origin + "/props", timeout=5)
        if pr.status_code == 200:
            props = pr.json() or {}
            dgs = props.get("default_generation_settings") or {}
            if isinstance(dgs, dict) and ("n_ctx" in dgs or "n_predict" in dgs):
                out["engine"] = "llama.cpp"
                n_ctx = dgs.get("n_ctx")
                if isinstance(n_ctx, int) and n_ctx > 0:
                    out["max_context"] = n_ctx
                    out["max_context_source"] = "/props.default_generation_settings.n_ctx"
    except requests.RequestException:
        pass

    if out["engine"] == "unknown":
        # vLLM /load, SGLang /get_server_info로 추가 판별.
        try:
            lr = requests.get(origin + "/load", timeout=5)
            if lr.status_code == 200:
                try:
                    lj = lr.json() or {}
                except ValueError:
                    lj = {}
                if isinstance(lj, dict) and any(
                    k in lj for k in ("num_requests_running", "num_waiting", "server_load", "dispatcher_stats")
                ):
                    out["engine"] = "vLLM"
        except requests.RequestException:
            pass
        if out["engine"] == "unknown":
            try:
                mr = requests.get(origin + "/metrics", timeout=5)
                if mr.status_code == 200 and "vllm" in mr.text[:4000].lower():
                    out["engine"] = "vLLM"
            except requests.RequestException:
                pass
        if out["engine"] == "unknown":
            try:
                sr = requests.get(origin + "/get_server_info", timeout=5)
                if sr.status_code == 200:
                    sj = sr.json() or {}
                    out["engine"] = "SGLang"
                    for key in ("context_length", "max_total_num_tokens", "max_prefill_tokens"):
                        value = sj.get(key)
                        if isinstance(value, int) and value > 0:
                            out["max_context"] = value
                            out["max_context_source"] = "/get_server_info." + key
                            break
            except requests.RequestException:
                pass
        if out["engine"] == "unknown":
            header = (out["server_header"] or "").lower()
            if "llama" in header:
                out["engine"] = "llama.cpp"
            elif "vllm" in header:
                out["engine"] = "vLLM"
            elif out["ok"]:
                out["engine"] = "OpenAI 호환"
    return out


# ---------- 카드 CRUD ----------

@router.get("")
def bench_get():
    with _store_lock:
        store = _load_store()
    with _jobs_lock:
        jobs = [
            {k: v for k, v in job.items() if k not in ("cancel_event",)}
            for job in _jobs.values()
        ]
    return {"cards": store["cards"], "results": store["results"], "jobs": jobs}


@router.post("/cards")
def card_create(data: dict):
    data = data or {}
    title = str(data.get("title") or "").strip() or f"새 벤치마크 {datetime.now().strftime('%m-%d %H:%M')}"
    card = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "created_at": _now_iso(),
        "api_url": "",
        "source": "manual",
        "source_label": "",
        "engine": None,
        "model": None,
        "max_context": None,
        "probed_at": None,
        "conc": False,
        "users": 4,
    }
    with _store_lock:
        store = _load_store()
        store["cards"].insert(0, card)
        _save_store(store)
    return {"card": card}


@router.patch("/cards/{card_id}")
def card_update(card_id: str, data: dict):
    data = data or {}
    with _store_lock:
        store = _load_store()
        card = next((c for c in store["cards"] if c["id"] == card_id), None)
        if not card:
            raise HTTPException(404, "카드를 찾지 못했습니다")
        for field in ("title", "api_url", "source", "source_label", "model"):
            if field in data:
                value = str(data[field] if data[field] is not None else "").strip()
                if field == "api_url" and value:
                    try:
                        value = normalize_base(value)
                    except HTTPException:
                        raise
                card[field] = value
        if "users" in data:
            try:
                card["users"] = min(64, max(1, int(data["users"])))
            except (TypeError, ValueError):
                pass
        if "conc" in data:
            card["conc"] = bool(data["conc"])
        _save_store(store)
    return {"card": card}


@router.delete("/cards/{card_id}")
def card_delete(card_id: str):
    with _store_lock:
        store = _load_store()
        before = len(store["cards"])
        store["cards"] = [c for c in store["cards"] if c["id"] != card_id]
        store["results"] = [r for r in store["results"] if r.get("card_id") != card_id]
        removed = before - len(store["cards"])
        _save_store(store)
    if not removed:
        raise HTTPException(404, "카드를 찾지 못했습니다")
    return {"ok": True}


@router.post("/cards/{card_id}/probe")
def card_probe(card_id: str, data: dict):
    data = data or {}
    with _store_lock:
        store = _load_store()
        card = next((c for c in store["cards"] if c["id"] == card_id), None)
        if not card:
            raise HTTPException(404, "카드를 찾지 못했습니다")
        api_url = data.get("api_url") or card.get("api_url")
        info = probe_server(api_url, model_hint=card.get("model"))
        if info.get("ok"):
            card["api_url"] = info["api_url"]
            card["engine"] = info["engine"]
            card["model"] = data.get("model") or info["model"]
            card["max_context"] = info["max_context"]
            card["models"] = info.get("models") or []
            card["probed_at"] = _now_iso()
            if info.get("max_context_source"):
                card["max_context_source"] = info["max_context_source"]
        _save_store(store)
    return {"card": card, "probe": info}


@router.get("/services")
def services_detect():
    """main_server가 직접 실행 중인 LLM 서비스(vLLM/llama.cpp) 목록."""
    if _service_discovery is None:
        return {"services": []}
    try:
        found = _service_discovery() or []
    except Exception as exc:  # 상태 조회 실패로 벤치마크 페이지가 죽으면 안 됨
        return {"services": [], "error": str(exc)}
    return {"services": found}


# ---------- 벤치마크 실행 ----------

def _new_job(card_id, kind):
    job = {
        "id": uuid.uuid4().hex[:12],
        "card_id": card_id,
        "kind": kind,
        "status": "running",
        "started_at": _now_iso(),
        "started_mono": time.monotonic(),
        "finished_at": None,
        "error": None,
        "cancelled": False,
        "log": [],
        "stages": [],
        "cancel_event": threading.Event(),
    }
    with _jobs_lock:
        _jobs[job["id"]] = job
        finished = [j for j in _jobs.values() if j["status"] != "running"]
        for old in finished[:-20]:
            _jobs.pop(old["id"], None)
    return job


def _job_log(job, message):
    job["log"].append({"t": _now_iso(), "message": message})


# 이 manager 세션에서 벤치를 돌린 적 있는 서버 URL — /metrics 없는 서버의 cold 판정 보조용.
_benched_urls = set()


def _server_cold_state(origin):
    """vLLM/llama.cpp /metrics의 생애 누적 요청·토큰 카운터로 첫 가동 여부를 안다.

    True  = 카운터가 0 (이 서버는 이 프로세스 생애에서 아직 completions을 안 받음 → 첫 pp는 cold)
    False = 카운터가 있음 (이미 트래픽 있음 → 일반 프리필)
    None  = /metrics 없음/파싱 불가 (판정 불가)
    """
    try:
        r = requests.get(origin + "/metrics", timeout=5)
        if r.status_code != 200:
            return None
        pattern = re.compile(
            r"^(?:vllm:(?:prompt_tokens_total|num_requests_total|request_success_total)"
            r"|llama_n_prompt_tokens_total|llama:prompt_tokens_total)\b\S*\s+([0-9eE.+-]+)"
        )
        total = None
        for line in r.text.splitlines():
            m = pattern.match(line)
            if m:
                try:
                    total = (total or 0.0) + float(m.group(1))
                except ValueError:
                    pass
        if total is None:
            return None
        return total <= 0.0
    except requests.RequestException:
        return None


def _run_concurrent(base, model, prompts, engine, gen_tokens, cancel_event):
    """N개 유니크 프롬프트를 동시에 스트리밍 요청해 사용자별/총합 속도를 잰다.

    - 사용자별 decode: 각 요청의 (토큰수-1)/(종료-첫토큰) 평균
    - 총합 decode: 전체 생성 토큰 / (마지막 종료 - 첫 번째 토큰) 윈도우
    - 사용자별 pp(체감): prompt_tokens/TTFT 평균 — 동시 요청이면 큐잉이 포함돼
      단일 사용자 pp보다 낮게 나오는 것이 정상이다.
    """
    n = len(prompts)
    metrics = [None] * n
    errors = []
    t0 = time.perf_counter()

    def one(idx):
        try:
            metrics[idx] = _measure_stream(base, model, prompts[idx], gen_tokens, engine,
                                           cancel_event=cancel_event, t0=t0)
        except Exception as exc:
            errors.append(f"#{idx + 1}: {exc.__class__.__name__}")

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    done = [m for m in metrics if m]
    if not done:
        return {"users": n, "ok": False, "error": "; ".join(errors[:3]) or "전원 실패"}
    per_user_decode, per_user_pp = [], []
    for m in done:
        gen = m["completion_tokens"] or m["chunks"]
        window = max(m["total"] - m["ttft"], 1e-6)
        per_user_decode.append(max(gen - 1, 1) / window)
        if m.get("prompt_tokens"):
            per_user_pp.append(m["prompt_tokens"] / max(m["ttft"], 1e-6))
    total_gen = sum(max((m["completion_tokens"] or m["chunks"]) - 1, 1) for m in done)
    decode_window = max(max(m["total"] for m in done) - min(m["ttft"] for m in done), 1e-6)
    ptoks = [m["prompt_tokens"] for m in done if m.get("prompt_tokens")]
    return {
        "users": n,
        "ok": True,
        "completed": len(done),
        "errors": errors[:5],
        "prompt_tokens_avg": round(sum(ptoks) / len(ptoks)) if ptoks else None,
        "avg_ttft_s": round(sum(m["ttft"] for m in done) / len(done), 3),
        "max_ttft_s": round(max(m["ttft"] for m in done), 3),
        "per_user_decode_tok_s": round(sum(per_user_decode) / len(per_user_decode), 1),
        "total_decode_tok_s": round(total_gen / decode_window, 1),
        "per_user_pp_tok_s": round(sum(per_user_pp) / len(per_user_pp), 1) if per_user_pp else None,
    }


def _run_benchmark(job, card, api_url, model, engine, max_context, serving_label, users=1):
    """단계당 유니크 프롬프트 스트리밍 1회로 pp·TTFT·decode를 한꺼번에 잰다.

    - pp     : prompt_tokens / TTFT. 프롬프트는 매 단계 새로 만들고 선두에 랜덤
      태그를 붙여 vLLM 자동 prefix cache에 절대 맞지 않는다 (llama.cpp pp 로그와
      같은 "실제 프리필 처리량" 개념).
    - decode : (총 시간 - TTFT) 구간에서 생성 토큰당 속도.
    - cold   : 서버 /metrics의 생애 누적 카운터가 0이면 이 실행의 첫 프리필이
      진짜 첫 가동(cold)이다. 그 외에는 모두 일반 프리필로 표기 — 과거처럼
      같은 단계를 2회 반복 측정하지 않는다.
    - N명 동시(users>1): 같은 길이 유니크 프롬프트 N개를 동시에 스트리밍해
      사용자별/총합 속도를 별도 측정한다.
    """
    kind = job["kind"]
    stages = KIND_STAGES[kind]
    rng = random.Random(uuid.uuid4().int)
    base = normalize_base(api_url)
    origin = _origin(base)
    stage_results = []

    server_cold = _server_cold_state(origin)
    if server_cold is None:
        server_cold = False if base in _benched_urls else None
    meta = {
        "api_url": base,
        "engine": engine,
        "serving": serving_label,
        "model": model,
        "max_context": max_context,
        "gen_tokens": GEN_TOKENS,
        "users": users,
        "server_cold": server_cold,
    }
    if server_cold:
        _job_log(job, "서버 생애 첫 벤치 감지(/metrics 카운터 0) — 이 실행의 pp는 cold로 표기")

    for target in stages:
        if job["cancel_event"].is_set():
            job["cancelled"] = True
            break
        guard = max_context
        if guard and target + GEN_TOKENS + 64 > guard:
            stage = {
                "target_tokens": target, "skipped": True,
                "note": f"서버 최대 컨텍스트({guard:,}) 대비 프롬프트+생성이 커서 건너뜀",
            }
            stage_results.append(stage)
            job["stages"] = list(stage_results)
            _job_log(job, f"{target:,} tok 건너뜀 (ctx {guard:,} 부족)")
            continue

        stage = {"target_tokens": target, "skipped": False, "note": ""}

        # 단계당 요청 1회: 유니크 프롬프트로 gen 스트리밍 → TTFT=pp, 나머지=decode
        try:
            prompt, approx = _build_prompt(origin, model, target, rng)
            stage["prompt_tokens"] = approx
            decode = _measure_stream(base, model, prompt, GEN_TOKENS, engine,
                                     cancel_event=job["cancel_event"])
            tok = decode.get("prompt_tokens") or approx
            stage["prompt_tokens"] = tok
            gen = decode["completion_tokens"] or decode["chunks"]
            stage["ttft_s"] = round(decode["ttft"], 3)
            stage["pp_tok_s"] = round(tok / max(decode["ttft"], 1e-6), 1)
            decode_window = max(decode["total"] - decode["ttft"], 1e-6)
            stage["decode_tok_s"] = round(max(gen - 1, 1) / decode_window, 1)
            stage["decode_tokens"] = gen
            stage["total_s"] = round(decode["total"], 3)
            if gen < GEN_TOKENS * 0.9:
                stage["note"] = (stage["note"] + " " if stage["note"] else "") + \
                    f"EOS로 {gen}/{GEN_TOKENS}토큰만 생성됨 (decode 속도는 유효)"
        except TimeoutError:
            job["cancelled"] = True
        except requests.RequestException as exc:
            stage["skipped"] = True
            stage["note"] = f"측정 요청 실패: {exc.__class__.__name__}"
            stage_results.append(stage)
            job["stages"] = list(stage_results)
            _job_log(job, f"{target:,} tok 측정 실패: {exc.__class__.__name__}")
            continue

        # N명 동시 배치 — 같은 길이 유니크 프롬프트 N개를 동시에 스트리밍.
        # 방금 단일 측정에서 쓴 프롬프트는 KV 캐시에 남아있으니 재사용하지 않고
        # 전부 새로 생성한다(재사용하면 그 요청만 캐시 히트로 체감 pp가 부풀려짐).
        if users > 1 and not job["cancelled"]:
            _job_log(job, f"{target:,} tok × {users}명 동시 배치 실행 중…")
            prompts = []
            build_error = None
            for _ in range(users):
                if job["cancel_event"].is_set():
                    job["cancelled"] = True
                    break
                try:
                    p_text, _ = _build_prompt(origin, model, target, rng)
                    prompts.append(p_text)
                except Exception as exc:
                    build_error = f"동시 프롬프트 생성 실패: {exc}"
                    break
            if len(prompts) == users:
                conc = _run_concurrent(base, model, prompts, engine, GEN_TOKENS,
                                       job["cancel_event"])
                if conc and conc.get("ok"):
                    stage["concurrent"] = conc
                    _job_log(job, (
                        f"{target:,} tok × {users}명: 사용자별 decode "
                        f"{conc['per_user_decode_tok_s']:,} tok/s · 총 decode "
                        f"{conc['total_decode_tok_s']:,} tok/s · 평균 TTFT {conc['avg_ttft_s']}s"
                    ))
                elif conc:
                    stage["note"] = (stage["note"] + " " if stage["note"] else "") + \
                        "동시 측정 실패: " + str(conc.get("error") or "원인 없음")
            elif build_error:
                stage["note"] = (stage["note"] + " " if stage["note"] else "") + build_error

        stage_results.append(stage)
        job["stages"] = list(stage_results)
        _job_log(job, (
            f"{target:,} tok: pp {stage.get('pp_tok_s') or '-'} tok/s"
            f" · decode {stage.get('decode_tok_s') or '-'} tok/s · TTFT {stage.get('ttft_s') or '-'}s"
        ))

    _benched_urls.add(base)
    finished = _now_iso()
    job["finished_at"] = finished
    valid = [s for s in stage_results if not s.get("skipped")]
    if job["cancelled"]:
        job["status"] = "cancelled"
    elif not valid:
        job["status"] = "error"
        job["error"] = "모든 단계가 실패했습니다. 서버/모델/컨텍스트 설정을 확인하세요."
    else:
        job["status"] = "done"

    if valid:
        first = valid[0]
        best_pp = max((s.get("pp_tok_s") or 0) for s in valid)
        conc_src = next((s["concurrent"] for s in reversed(valid)
                         if s.get("concurrent") and s["concurrent"].get("ok")), None)
        summary = {
            "pp_tok_s": first.get("pp_tok_s"),
            "decode_tok_s": (valid[-1] if kind == "detailed" else first).get("decode_tok_s"),
            "best_pp_tok_s": best_pp or None,
        }
        if conc_src:
            summary["concurrent"] = {
                "users": conc_src["users"],
                "per_user_decode_tok_s": conc_src["per_user_decode_tok_s"],
                "total_decode_tok_s": conc_src["total_decode_tok_s"],
                "avg_ttft_s": conc_src["avg_ttft_s"],
            }
        result = {
            "id": uuid.uuid4().hex[:12],
            "card_id": card["id"],
            "card_title": card.get("title"),
            "kind": kind,
            "kind_label": KIND_LABELS.get(kind, kind),
            "started_at": job["started_at"],
            "finished_at": finished,
            "duration_s": round(time.monotonic() - job["started_mono"], 1),
            "partial": bool(job["cancelled"]),
            "meta": meta,
            "stages": stage_results,
            "summary": summary,
        }
        with _store_lock:
            store = _load_store()
            store["results"].insert(0, result)
            store["results"] = store["results"][:MAX_RESULTS]
            _save_store(store)
        job["result_id"] = result["id"]


@router.post("/run")
def bench_run(data: dict):
    data = data or {}
    card_id = str(data.get("card_id") or "").strip()
    kind = str(data.get("kind") or "quick").strip()
    if kind not in KIND_STAGES:
        raise HTTPException(400, f"알 수 없는 벤치마크 종류: {kind}")
    with _store_lock:
        store = _load_store()
        card = next((c for c in store["cards"] if c["id"] == card_id), None)
    if not card:
        raise HTTPException(404, "카드를 찾지 못했습니다")
    api_url = data.get("api_url") or card.get("api_url")
    if not api_url:
        raise HTTPException(400, "API 주소를 먼저 지정하세요")
    try:
        users = int(data.get("users") or 1)
    except (TypeError, ValueError):
        users = 1
    users = min(32, max(1, users))
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 벤치마크가 이미 실행 중입니다. 끝나고 나서 다시 시도하세요")

    job = _new_job(card_id, kind)
    job["users"] = users

    def worker():
        try:
            # 실행 직전 서버를 다시_probe해 엔진/모델/최대 컨텍스트 메타데이터를 갱신.
            info = probe_server(api_url, model_hint=data.get("model") or card.get("model"))
            if not info.get("ok"):
                job["status"] = "error"
                job["error"] = info.get("error") or "서버에 연결할 수 없습니다"
                job["finished_at"] = _now_iso()
                return
            model = data.get("model") or card.get("model") or info["model"]
            if not model:
                job["status"] = "error"
                job["error"] = "모델을 찾지 못했습니다 (/v1/models가 비어 있음)"
                job["finished_at"] = _now_iso()
                return
            engine = info["engine"]
            max_context = info["max_context"]
            serving_label = card.get("source_label") or (
                f"{engine} @ {info['api_url']}" if engine else info["api_url"]
            )
            if card.get("source") != "auto":
                serving_label = f"수동 API · {serving_label}"
            # 카드에도 최신 탐지 정보를 반영.
            with _store_lock:
                fresh = _load_store()
                target = next((c for c in fresh["cards"] if c["id"] == card_id), None)
                if target:
                    target["api_url"] = info["api_url"]
                    target["engine"] = engine
                    target["model"] = model
                    target["max_context"] = max_context
                    target["probed_at"] = _now_iso()
                    if info.get("max_context_source"):
                        target["max_context_source"] = info["max_context_source"]
                    _save_store(fresh)
            job["meta"] = {"engine": engine, "model": model, "max_context": max_context,
                           "serving": serving_label}
            if max_context:
                _job_log(job, f"서버 연결됨: {serving_label} · 모델 {model} · max ctx {max_context:,}")
            else:
                _job_log(job, f"서버 연결됨: {serving_label} · 모델 {model}")
            _run_benchmark(job, card, info["api_url"], model, engine, max_context,
                           serving_label, users=users)
        except Exception as exc:  # 워커 예외가 스레드에서 조용히 죽지 않도록 job에 기록
            job["status"] = "error"
            job["error"] = f"{exc.__class__.__name__}: {exc}"
            job["finished_at"] = _now_iso()
        finally:
            _run_lock.release()

    threading.Thread(target=worker, name=f"llm-bench-{job['id']}", daemon=True).start()
    return {"job_id": job["id"]}


@router.get("/jobs/{job_id}")
def job_get(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "작업을 찾지 못했습니다")
    return {k: v for k, v in job.items() if k != "cancel_event"}


@router.get("/jobs/{job_id}/cancel")
@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "작업을 찾지 못했습니다")
    job["cancel_event"].set()
    return {"ok": True}


@router.delete("/results/{result_id}")
def result_delete(result_id: str):
    with _store_lock:
        store = _load_store()
        before = len(store["results"])
        store["results"] = [r for r in store["results"] if r.get("id") != result_id]
        removed = before - len(store["results"])
        _save_store(store)
    if not removed:
        raise HTTPException(404, "결과를 찾지 못했습니다")
    return {"ok": True}
