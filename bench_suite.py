"""External Bench — EleutherAI/lm-evaluation-harness와 local-inference-lab/llm-inference-bench를
메인서버에서 실행 중인 vLLM/llama.cpp 서버에 물어 벤치하는 페이지의 백엔드.

기존 llm_bench.py(직접 계측하는 pp/decode 벤치)와 달리, 이 페이지는 외부 벤치 도구를
서브프로세스로 실행하고 그 결과(JSON)를 파싱해 카드에 누적 기록한다.

- 벤치 대상 API는 메인서버가 켜둔 서비스만 허용한다(화이트리스트). 임의 URL은 400.
  · llm-inference-bench는 /v1/chat/completions 기반이라 vLLM 전용(SGLang/vLLM 최적화).
  · lm-eval-harness는 llama.cpp를 llama.cpp 전용 백엔드(gguf), vLLM을 OpenAI
    completions 백엔드(local-completions + 원격/tokenizer)로 평가한다.
- 도구 설치는 수동: bench_tools/setup_bench_tools.sh 가 클론+venv를 준비한다.
  상태는 GET /api/bench-suite/tool-status 로 노출, 없으면 UI가 설치 명령을 보여준다.
- 결과는 bench_suite.json에 카드별 누적(상한 200), 실행 로그는 bench_suite_logs/에 남긴다.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from config import BASE_DIR

router = APIRouter(prefix="/api/bench-suite", tags=["bench-suite"])

BENCH_DIR = os.path.join(BASE_DIR, "bench_tools")
VENV_PY = os.path.join(BENCH_DIR, ".venv", "bin", "python")
LIB_SCRIPT = os.path.join(BENCH_DIR, "llm-inference-bench", "llm_decode_bench.py")
LMEVAL_CLI = os.path.join(BENCH_DIR, ".venv", "bin", "lm-eval")
STORE_FILE = os.path.join(BASE_DIR, "bench_suite.json")
LOG_DIR = os.path.join(BASE_DIR, "bench_suite_logs")
RAW_DIR = os.path.join(BENCH_DIR, "runs")
SETUP_SCRIPT = os.path.join(BENCH_DIR, "setup_bench_tools.sh")
MAX_RESULTS = 200
LOG_TAIL_CHARS = 16000

# 프리필 스카우트 컨텍스트는 서버 컨텍스트 한도에 맞춰 잘라낸다.
PREFILL_CANDIDATES = [8192, 16384, 32768, 65536, 131072]

_store_lock = threading.Lock()
_run_lock = threading.Lock()   # GPU/서버를 잡는 벤치는 한 번에 하나
_jobs = {}
_jobs_lock = threading.Lock()
_tasks_cache = {"ts": 0.0, "data": None}

_service_discovery = None


def set_service_discovery(fn):
    global _service_discovery
    _service_discovery = fn


def _now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


# ---------- 대상 서버(메인서버 실행 서비스만) ----------

def _services():
    if _service_discovery is None:
        return []
    try:
        return _service_discovery() or []
    except Exception:
        return []


def _resolve_target(service_key):
    """메인서버가 실행 중인 LLM 서비스 하나를 카드용 dict로 정규화."""
    svc = next((s for s in _services() if s.get("key") == service_key), None)
    if not svc:
        raise HTTPException(400, "메인서버에서 실행 중인 LLM 서비스가 아닙니다. "
                                 "vLLM/llama.cpp를 먼저 켜세요")
    base = str(svc.get("base_url") or "")
    origin = re.sub(r"/v1/?$", "", base)
    engine = str(svc.get("engine") or "")
    return {
        "service_key": svc["key"],
        "service_label": svc.get("label") or svc["key"],
        "origin": origin,          # http://127.0.0.1:8000 (접두어 비교용)
        "api_url": base,           # .../v1
        "engine": engine,
        "engine_key": "llama" if "llama" in engine.lower() else ("vllm" if "vllm" in engine.lower() else "other"),
    }


def _check_job_target(job):
    """실행 직전 화이트리스트 재검증: 서비스가 내려갔으면 409."""
    live = _resolve_target(job["service_key"])
    if live["origin"] != job["origin"]:
        raise HTTPException(409, "대상 서버가 실행 중이 아닙니다 — 메인서버에서 해당 LLM 서비스를 켜세요")
    return live


# ---------- 도구 상태 ----------

def _tool_status():
    venv_ok = os.path.isfile(VENV_PY)
    return {
        "venv_ready": venv_ok,
        "lib_ready": venv_ok and os.path.isfile(LIB_SCRIPT),
        "lmeval_ready": venv_ok and os.path.isfile(LMEVAL_CLI),
        "bench_dir": BENCH_DIR,
        "setup_script": os.path.relpath(SETUP_SCRIPT, BASE_DIR),
        "python": VENV_PY,
    }


def _require(tool):
    st = _tool_status()
    if not st[tool]:
        raise HTTPException(409, f"{tool} 준비되지 않음 — bench_tools/setup_bench_tools.sh 실행 후 새로고침")


@router.get("/tool-status")
def tool_status():
    return _tool_status()


@router.get("/tasks")
def lmeval_tasks():
    """lm-eval이 아는 태스크/그룹 목록(10분 캐시)."""
    now = time.time()
    if _tasks_cache["data"] is not None and now - _tasks_cache["ts"] < 600:
        return _tasks_cache["data"]
    _require("lmeval_ready")
    try:
        proc = subprocess.run([LMEVAL_CLI, "ls", "tasks"], capture_output=True, text=True,
                              timeout=180, cwd=BENCH_DIR)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(500, f"태스크 목록 조회 실패: {exc}")
    if proc.returncode != 0:
        raise HTTPException(500, f"lm-eval ls tasks 실패: {proc.stderr[-400:]}")
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        name, loc = cells[0], cells[1]
        if not name or name == "Group" or set(name) <= set("-"):
            continue
        rows.append({"name": name, "group": "/_groups/" in loc, "path": loc})
    payload = {"tasks": rows, "names": [r["name"] for r in rows]}
    _tasks_cache.update({"ts": now, "data": payload})
    return payload


# ---------- 실행 명령 구성 ----------

def _lib_command(card, preset):
    target = _resolve_target(card.get("service_key"))
    if target["engine_key"] != "vllm":
        raise HTTPException(400, "llm-inference-bench는 vLLM/SGLang 서버 전용입니다 "
                                 f"(이 카드 엔진: {target['engine'] or '?'})")
    concurrency = [int(x) for x in str(preset.get("concurrency") or "").replace(" ", "").split(",") if x]
    contexts = [int(x) for x in str(preset.get("contexts") or "").replace(" ", "").split(",") if x]
    if not concurrency or not contexts:
        raise HTTPException(400, "동시성/컨텍스트 목록을 채우세요")
    duration = int(preset.get("duration") or 30)
    if not 5 <= duration <= 3600:
        raise HTTPException(400, "셀당 시간은 5~3600초 사이여야 합니다")
    max_tokens = int(preset.get("max_tokens") or 2048)
    if not 16 <= max_tokens <= 131072:
        raise HTTPException(400, "max_tokens는 16~131072 사이여야 합니다")
    output = os.path.join(RAW_DIR, f"lib_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json")
    command = [VENV_PY, LIB_SCRIPT,
               "--host", target["origin"],
               "--model", card.get("model") or "",
               "--concurrency", ",".join(str(c) for c in concurrency[:12]),
               "--contexts", ",".join(str(c) for c in contexts[:8]),
               "--duration", str(duration),
               "--max-tokens", str(max_tokens),
               "--prefill-contexts",
               ",".join(f"{c // 1024}k" for c in PREFILL_CANDIDATES
                        if c <= int(card.get("max_context") or 0) or c == PREFILL_CANDIDATES[0]),
               "--display-mode", "plain", "--no-hw-monitor", "--no-resume",
               "--output", output]
    return command, output


_LME_PAT = re.compile(r"^[A-Za-z0-9_,.\- ]+$")


def _lme_command(card, preset):
    target = _resolve_target(card.get("service_key"))
    engine_key = target["engine_key"]
    if engine_key == "vllm":
        model_type = "local-completions"
        base_url = target["api_url"] + "/completions"
    elif engine_key == "llama":
        model_type = "gguf"                      # llama.cpp 전용 백엔드(/props 슬롯 병렬, id_slot)
        base_url = target["origin"]
    else:
        raise HTTPException(400, f"지원하지 않는 엔진: {target['engine']}")
    if card.get("use_chat_template"):
        model_type = {"local-completions": "local-chat-completions", "gguf": "gguf"}[model_type]
        base_url = target["origin"] + ("/v1/chat/completions" if model_type == "local-chat-completions" else "")

    tasks = str(preset.get("tasks") or "").strip()
    if not tasks or not _LME_PAT.match(tasks):
        raise HTTPException(400, "태스크 목록은 쉼표로 구분된 이름만 가능합니다")
    # 쉼표/공백으로 나눠 공백을 제거 — lm-eval은 쉼표만 나눠 ' gsm8k'를 모르는 태스크로 본다.
    task_list = [t for t in re.split(r"[,\s]+", tasks) if t]
    known = (_tasks_cache["data"] or {}).get("names")
    if known:
        missing = [t for t in task_list if t not in known]
        if missing:
            raise HTTPException(400, "lm-eval이 모르는 태스크: " + ", ".join(missing)
                                 + " — 자동완성 목록에서 선택하세요")
    tasks = ",".join(task_list)
    limit = preset.get("limit")
    try:
        limit = int(limit) if limit not in (None, "") else None
    except (TypeError, ValueError):
        limit = None
    if limit is not None and not 1 <= limit <= 100000:
        raise HTTPException(400, "limit은 1~100000")
    num_concurrent = min(64, max(1, int(preset.get("num_concurrent") or 8)))
    fewshot = preset.get("num_fewshot")
    tokenizer = str(card.get("tokenizer") or "").strip()
    if engine_key == "vllm":
        # llama.cpp와 달리 vLLM은 /tokenizer_info를 제공하지 않아 HF 토크나이저가 필요하다.
        if not tokenizer or not os.path.isdir(tokenizer):
            raise HTTPException(400, "vLLM 평가에는 서버가 올린 모델의 토크나이저 디렉터리가 필요합니다 — "
                                     "토크나이저 경로를 채우세요 (docker run의 -v ...:/model:ro 소스 경로)")

    out_dir = os.path.join(RAW_DIR, f"lme_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    model_args = {
        "base_url": base_url,
        "model": card.get("model") or "",
        "num_concurrent": str(num_concurrent),
        "max_gen_toks": str(int(preset.get("max_gen_toks") or 256)),
    }
    if engine_key == "vllm":
        model_args["tokenizer_backend"] = "huggingface"
        model_args["tokenizer"] = tokenizer
    if engine_key == "llama" and card.get("parallel"):
        model_args["parallel"] = str(int(card["parallel"]))
    command = [LMEVAL_CLI, "run", "--model", model_type,
               "--model_args", ",".join(f"{k}={v}" for k, v in model_args.items()),
               "--tasks", tasks, "--batch_size", "1", "--output_path", out_dir]
    if limit is not None:
        command += ["--limit", str(limit)]
    if fewshot not in (None, ""):
        command += ["--num_fewshot", str(int(fewshot))]
    if card.get("use_chat_template") and engine_key == "vllm":
        command += ["--apply_chat_template"]
    return command, out_dir


def _lib_preset_args(preset_id, custom):
    if preset_id == "lib_quick":
        return {"concurrency": "1,2", "contexts": "0", "duration": 15, "max_tokens": 512}
    if preset_id == "lib_standard":
        return {"concurrency": "1,2,4,8", "contexts": "0,16384", "duration": 30, "max_tokens": 2048}
    if preset_id == "lib_full":
        return {"concurrency": "1,2,4,8,16", "contexts": "0,16384,65536", "duration": 30, "max_tokens": 2048}
    if preset_id == "lib_custom":
        return custom or {}
    raise HTTPException(400, f"알 수 없는 프리셋: {preset_id}")


def _lme_preset_args(preset_id, custom):
    if preset_id == "lme_quick":
        return {"tasks": "luka_monster,piqa,arc_easy", "limit": 200, "num_fewshot": 0}
    if preset_id == "lme_standard":
        return {"tasks": "arc_easy,arc_challenge,truthfulqa_mc2,hellaswag,gsm8k", "limit": None, "num_fewshot": None}
    if preset_id == "lme_custom":
        return custom or {}
    raise HTTPException(400, f"알 수 없는 프리셋: {preset_id}")


# ---------- 잡 실행 ----------

def _new_job(card_id, suite, preset_id, command, artifact):
    job = {
        "id": uuid.uuid4().hex[:12],
        "card_id": card_id,
        "suite": suite,
        "preset": preset_id,
        "status": "running",
        "started_at": _now_iso(),
        "started_mono": time.monotonic(),
        "finished_at": None,
        "error": None,
        "log": [],
        "cancel_event": threading.Event(),
        "log_file": os.path.join(LOG_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.log"),
        "artifact": artifact,
        "command": [str(c) for c in command],
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    with _jobs_lock:
        _jobs[job["id"]] = job
        finished = [j for j in _jobs.values() if j["status"] != "running"]
        for old in sorted(finished, key=lambda j: j["started_at"])[:-20]:
            _jobs.pop(old["id"], None)
    return job


def _job_log(job, message):
    job["log"].append({"t": _now_iso(), "message": message})


def _read_log_tail(path):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - LOG_TAIL_CHARS))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _sanitize(cmd_str):
    return re.sub(r"(--api-key[ =])\S+", r"\1***", cmd_str)


def _parse_lib_result(path, started_at):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("metadata") or {}
    cells = []
    for r in data.get("results") or []:
        cells.append({
            "concurrency": r.get("concurrency"),
            "context_tokens": r.get("context_tokens"),
            "aggregate_tps": round(r.get("aggregate_tps") or 0, 1) if r.get("aggregate_tps", -1) >= 0 else None,
            "ttft_avg_s": round(r.get("ttft_avg") or 0, 3) or None,
            "itl_avg_ms": round((r.get("inter_token_latency_avg") or 0) * 1000, 1) or None,
            "output_tokens": r.get("client_output_tokens"),
            "failed": bool(r.get("failure_reason")),
            "failure": (r.get("failure_reason") or "")[:160] or None,
        })
    prefill = {}
    for ctx, p in (data.get("prefill") or {}).items():
        prefill[ctx] = {"tok_per_sec": p.get("tok_per_sec"),
                        "ttft_s": p.get("client_ttft_seconds") or p.get("ttft_seconds"),
                        "prompt_tokens": p.get("prompt_tokens")}
    summary = {
        "engine": meta.get("engine"),
        "model": meta.get("model"),
        "concurrency_levels": meta.get("concurrency_levels"),
        "context_lengths": meta.get("context_lengths"),
        "duration_per_test": meta.get("duration_per_test"),
        "best_total_tps": None, "best_single_tps": None,
    }
    vals = [c for c in cells if c["aggregate_tps"]]
    if vals:
        summary["best_total_tps"] = max(c["aggregate_tps"] for c in vals)
        summary["best_single_tps"] = max((c["aggregate_tps"] / max(c["concurrency"] or 1, 1)) for c in vals)
    return {
        "started_at": started_at,
        "tool_version": meta.get("version"),
        "meta": {"engine": summary["engine"], "model": summary["model"],
                 "concurrency": summary["concurrency_levels"], "contexts": summary["context_lengths"]},
        "summary": summary,
        "cells": cells,
        "prefill": prefill,
        "summary_table": data.get("summary_table") or {},
    }


def _parse_lme_result(out_dir):
    """lm-eval이 써놓은 results_*.json을 읽어 태스크별 점수표로 축약."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(out_dir, "**", "results_*.json"), recursive=True))
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as fh:
        data = json.load(fh)
    results = data.get("results") or {}
    n_samples = data.get("n-samples") or {}
    tasks = []
    for name, res in results.items():
        metrics = []
        for key, value in res.items():
            if "," not in key or not isinstance(value, (int, float)):
                continue
            metric, filt = key.split(",", 1)
            if metric.endswith("_stderr"):
                continue      # stderr는 각 metric 항목에 병합해 저장
            stderr = res.get(f"{metric}_stderr,{filt}")
            metrics.append({"metric": metric, "filter": filt,
                            "value": round(float(value), 5),
                            "stderr": round(float(stderr), 5) if isinstance(stderr, (int, float)) else None})
        if metrics:
            ns = n_samples.get(name) or {}
            tasks.append({"task": name, "alias": res.get("alias") or name,
                          "samples": ns.get("effective") or ns.get("original") or res.get("sample_len"),
                          "metrics": metrics})
    cfg = data.get("config") or {}
    return {
        "result_file": os.path.relpath(files[-1], BASE_DIR),
        "meta": {"model_args": cfg.get("model_args"), "num_fewshot": cfg.get("num_fewshot"),
                 "limit": cfg.get("limit"), "config_model": cfg.get("model")},
        "tasks": tasks,
        "groups": {k: v for k, v in (data.get("group_subtasks") or {}).items() if v},
        "date": data.get("date"),
    }


def _finalize(job, card, target, status, exit_code):
    finished = _now_iso()
    job["finished_at"] = finished
    job["status"] = status
    if status == "error" and not job.get("error"):
        job["error"] = f"도구 종료 코드 {exit_code} — 로그를 확인하세요"
    if status not in ("error", "cancelled"):
        status = "done"
    parsed = None
    try:
        if job["suite"] == "lib":
            if os.path.isfile(job["artifact"]):
                parsed = _parse_lib_result(job["artifact"], job["started_at"])
        else:
            parsed = _parse_lme_result(job["artifact"])
    except Exception as exc:
        status = "error"
        job["error"] = f"결과 파싱 실패: {exc.__class__.__name__}: {exc}"
    if parsed is None and status == "done":
        status = "error"
        job["error"] = "결과 파일을 찾지 못했습니다 (실행은 끝났지만 산출물이 없음)"
    if status == "error":
        tail = _read_log_tail(job["log_file"])
        if "gated dataset" in tail:
            job["error"] = (job.get("error") or "") + \
                " · 게이트된 HF 데이터셋(gpqa 등)입니다 — bench_tools/.venv/bin/hf auth login 후 재실행"
        elif "DatasetNotFoundError" in tail:
            job["error"] = (job.get("error") or "") + " · 데이터셋을 받지 못했습니다 (네트워크/이름 확인)"

    duration = round(time.monotonic() - job["started_mono"], 1)
    with _store_lock:
        store = _load_store()
        live_card = next((c for c in store["cards"] if c["id"] == card["id"]), card)
        result = {
            "id": uuid.uuid4().hex[:12],
            "card_id": card["id"],
            "card_title": live_card.get("title"),
            "suite": job["suite"],
            "preset": job["preset"],
            "started_at": job["started_at"],
            "finished_at": finished,
            "duration_s": duration,
            "exit_code": exit_code,
            "status": status,
            "error": job.get("error"),
            "log_file": os.path.relpath(job["log_file"], BASE_DIR),
            "command": _sanitize(" ".join(job["command"])),
            "requested": job.get("requested"),
            "meta": {"engine": target["engine"], "model": card.get("model"),
                     "serving": target["service_label"], "api_url": target["api_url"]},
        }
        if parsed:
            parsed_meta = parsed.pop("meta", None)
            result.update(parsed)
            if isinstance(parsed_meta, dict):
                result["meta"].update(parsed_meta)
        store["results"].insert(0, result)
        store["results"] = store["results"][:MAX_RESULTS]
        _save_store(store)
    job["result_id"] = result["id"]
    job["status"] = status


def _terminate(process, cancel_event):
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except OSError:
        try:
            process.terminate()
        except OSError:
            pass
    deadline = time.time() + 8
    while time.time() < deadline and process.poll() is None:
        time.sleep(0.3)
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            pass


def _worker(job, card, target, preset_id):
    try:
        _job_log(job, "실행: " + _sanitize(" ".join(job["command"])))
        env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1", NO_COLOR="1", PYTHONUNBUFFERED="1")
        with open(job["log_file"], "wb") as log_fh:
            process = subprocess.Popen(
                job["command"], cwd=BENCH_DIR, env=env,
                stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
            job["pid"] = process.pid
            while process.poll() is None:
                if job["cancel_event"].is_set():
                    _terminate(process, job["cancel_event"])
                    _finalize(job, card, target, "cancelled", process.poll())
                    return
                time.sleep(1.0)
            exit_code = process.poll()
        if job["cancel_event"].is_set():
            _finalize(job, card, target, "cancelled", exit_code)
        elif exit_code != 0:
            tail = _read_log_tail(job["log_file"])[-800:]
            job["error"] = f"종료 코드 {exit_code}: {tail}"
            _finalize(job, card, target, "error", exit_code)
        else:
            _finalize(job, card, target, "done", exit_code)
    except Exception as exc:
        job["error"] = f"{exc.__class__.__name__}: {exc}"
        try:
            _finalize(job, card, target, "error", -1)
        except Exception:
            job["status"] = "error"
            job["finished_at"] = _now_iso()
    finally:
        _run_lock.release()
        job["_lock_held"] = False


# ---------- 라우터 ----------

@router.get("")
def suite_get():
    with _store_lock:
        store = _load_store()
    with _jobs_lock:
        jobs = [{k: v for k, v in j.items() if k not in ("cancel_event",)} for j in _jobs.values()]
    return {"cards": store["cards"], "results": store["results"], "jobs": jobs}


@router.post("/cards")
def card_create(data: dict):
    data = data or {}
    title = str(data.get("title") or "").strip() or f"외부 벤치 {datetime.now().strftime('%m-%d %H:%M')}"
    card = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "created_at": _now_iso(),
        "suite": "lib" if data.get("suite") == "lib" else "lme",
        "service_key": "",
        "api_url": "",
        "source_label": "",
        "engine": None,
        "model": None,
        "max_context": None,
        "tokenizer": str(data.get("tokenizer") or ""),
        "use_chat_template": bool(data.get("use_chat_template")),
        "lib": {"concurrency": "1,2,4,8", "contexts": "0,16384", "duration": 30, "max_tokens": 2048},
        "lme": {"tasks": "arc_easy", "limit": 200, "num_concurrent": 8, "max_gen_toks": 256, "num_fewshot": None},
    }
    with _store_lock:
        store = _load_store()
        store["cards"].insert(0, card)
        _save_store(store)
    return {"card": card}


_ALLOWED_FIELDS = {"title", "suite", "service_key", "model", "tokenizer", "use_chat_template",
                   "lib", "lme", "parallel"}


@router.patch("/cards/{card_id}")
def card_update(card_id: str, data: dict):
    data = data or {}
    with _store_lock:
        store = _load_store()
        card = next((c for c in store["cards"] if c["id"] == card_id), None)
        if not card:
            raise HTTPException(404, "카드를 찾지 못했습니다")
        for field, value in data.items():
            if field not in _ALLOWED_FIELDS:
                continue
            if field == "suite":
                card["suite"] = "lib" if value == "lib" else "lme"
            elif field in ("lib", "lme") and isinstance(value, dict):
                merged = dict(card.get(field) or {})
                merged.update({k: v for k, v in value.items()
                               if k in ("concurrency", "contexts", "duration", "max_tokens",
                                        "tasks", "limit", "num_concurrent", "max_gen_toks", "num_fewshot")})
                card[field] = merged
            elif field == "use_chat_template":
                card[field] = bool(value)
            elif field == "parallel":
                try:
                    card["parallel"] = min(64, max(1, int(value)))
                except (TypeError, ValueError):
                    pass
            else:
                card[field] = str(value if value is not None else "").strip()
        # 서비스 재지정 시 카드의 api_url/engine/서버 메타를 화이트리스트에서 갱신.
        if card.get("service_key"):
            try:
                target = _resolve_target(card["service_key"])
                card["api_url"] = target["api_url"]
                card["engine"] = target["engine"]
                card["source_label"] = target["service_label"]
            except HTTPException:
                card["service_key"] = ""
                card["source_label"] = ""
        _save_store(store)
    # 최대 컨텍스트는 잠금 밖에서 프로브
    if card.get("service_key"):
        try:
            import llm_bench
            info = llm_bench.probe_server(card["api_url"], model_hint=card.get("model"))
            if info.get("ok"):
                with _store_lock:
                    store = _load_store()
                    fresh = next((c for c in store["cards"] if c["id"] == card_id), None)
                    if fresh:
                        fresh["engine"] = info["engine"] or fresh.get("engine")
                        fresh["model"] = card.get("model") or info["model"]
                        fresh["max_context"] = info["max_context"]
                        fresh["models"] = info.get("models") or []
                        _save_store(store)
        except Exception:
            pass
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


@router.post("/run")
def suite_run(data: dict):
    data = data or {}
    card_id = str(data.get("card_id") or "").strip()
    suite = str(data.get("suite") or "").strip()
    preset_id = str(data.get("preset") or "").strip()
    with _store_lock:
        store = _load_store()
        card = next((c for c in store["cards"] if c["id"] == card_id), None)
    if not card:
        raise HTTPException(404, "카드를 찾지 못했습니다")
    suite = suite or card.get("suite")
    if not card.get("service_key"):
        raise HTTPException(400, "벤치할 서버(메인서버 서비스)를 먼저 선택하세요")
    target = _resolve_target(card["service_key"])

    if not card.get("model"):
        raise HTTPException(400, "모델을 먼저 조회하세요 (서버 선택 시 자동 채움)")

    if suite == "lib":
        _require("lib_ready")
        args = _lib_preset_args(preset_id, data.get("custom") or card.get("lib"))
        command, artifact = _lib_command(card, args)
    elif suite == "lme":
        _require("lmeval_ready")
        args = _lme_preset_args(preset_id, data.get("custom") or card.get("lme"))
        command, artifact = _lme_command(card, args)
    else:
        raise HTTPException(400, f"알 수 없는 벤치 종류: {suite}")

    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 외부 벤치가 이미 실행 중입니다. 끝나고 나서 다시 시도하세요")
    job = _new_job(card_id, suite, preset_id, command, artifact)
    if suite == "lme":
        job["requested"] = [t for t in re.split(r"[,\s]+", str(args.get("tasks") or "")) if t]
    job["_lock_held"] = True
    threading.Thread(target=_worker, args=(job, card, target, preset_id),
                     name=f"bench-suite-{job['id']}", daemon=True).start()
    return {"job_id": job["id"]}


@router.get("/jobs/{job_id}")
def job_get(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "작업을 찾지 못했습니다")
    out = {k: v for k, v in job.items() if k not in ("cancel_event", "_lock_held")}
    if job["status"] == "running":
        out["log_tail"] = _read_log_tail(job["log_file"])[-4000:]
    return out


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "작업을 찾지 못했습니다")
    job["cancel_event"].set()
    return {"ok": True}


@router.get("/logs")
def logs_get(path: str):
    """결과에 기록된 로그 파일(상대경로) 읽기 — bench_suite_logs/ 안만 허용."""
    real = os.path.realpath(os.path.join(BASE_DIR, path))
    if not real.startswith(os.path.realpath(LOG_DIR) + os.sep) or not os.path.isfile(real):
        raise HTTPException(404, "로그를 찾지 못했습니다")
    with open(real, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 200000))
        return {"path": path, "content": fh.read().decode("utf-8", "replace")}


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
