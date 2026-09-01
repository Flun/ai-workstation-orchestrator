"""CMP 170HX 전용 170tune 클럭 제어 브릿지.

170tune (https://github.com/cachenetics/170tune) 는 CMP 170HX(GA100)의
SM clock/언더볼트(GPC VF offset)와 HBM 메모리 클럭(NDIV)을 userspace에서
live로 조율하는 도구다. main_server는 비권한(flux)으로 실행되므로 root 전용
헬퍼 /usr/local/sbin/main-server-170tune 를 `sudo -n` 으로 호출해
170tune를 조종한다 (sudoers.d/main-server-170tune).

nvidia-smi 방식(-pl/-lgc)과의 관계: 두 방식은 같은 nvidia-smi 레지스터를
공유하므로 마지막 적용이 이긴다. 각 적용은 상대방의 lock을 먼저 해제하고,
GPU별 "부팅 방식"(METHOD)을 저장 파일에 기록해 부팅 재적용을 결정한다
(170tune-apply.service / gpu-tune.service).

설치: git clone https://github.com/cachenetics/170tune /opt/170tune &&
      cd /opt/170tune && sudo ./install.sh   (helpers -> /usr/local/bin)
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from fastapi import HTTPException

HELPER = "/usr/local/sbin/main-server-170tune"
TUNE_BIN = "/usr/local/bin/170tune"
CONFIG_DIR = Path("/etc/main-server/170tune.d")
STOCK_NDIV = 64
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 170hx-oc의 검증된 프로파일 테이블 (레포 tools/170hx-oc 헤더 주석과 동일).
# serving-qualified: dense, match. BENCH-ONLY: eff, balanced, perf, max
# (+300/+350 offset은 실서빙 soak에서 Xid 13 — UI에 경고 표시).
PROFILES = {
    "stock":    {"label": "stock (기본)",    "offset": 0,   "clk": 0,    "pl": 250, "bench": False,
                 "note": "공식 스톡: offset 0, 클럭 잠금 해제, 250W"},
    "dense":    {"label": "dense (서빙)",    "offset": 250, "clk": 1200, "pl": 300, "bench": False,
                 "note": "서빙 검증: -13% 성능 / -40% 전력"},
    "match":    {"label": "match (서빙 기본)", "offset": 250, "clk": 1400, "pl": 300, "bench": False,
                 "note": "서빙 검증 기본값: 스톡 성능 / -29% 전력"},
    "balanced": {"label": "balanced (벤치)", "offset": 300, "clk": 1470, "pl": 300, "bench": True,
                 "note": "벤치 전용: +6% 성능 / -25% 전력 (실서빙 soak 실패 이력)"},
    "perf":     {"label": "perf (벤치)",     "offset": 350, "clk": 1590, "pl": 300, "bench": True,
                 "note": "벤치 전용: +15% 성능 / -9% 전력 (실서빙 soak 실패 이력)"},
    "max":      {"label": "max (벤치)",      "offset": 350, "clk": 1650, "pl": 300, "bench": True,
                 "note": "벤치 전용: +17% 성능 (스톡 전력, 실서빙 soak 실패 이력)"},
    "custom":   {"label": "custom (직접)",   "offset": None, "clk": None, "pl": None, "bench": True,
                 "note": "offset/ceiling/PL 직접 입력 (+450 초과 거부)"},
}
VALID_PROFILES = tuple(PROFILES)


def installed() -> bool:
    return os.path.isfile(TUNE_BIN) and os.path.isfile(HELPER)


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HTTPException(503, f"170tune 실행 실패: {error}") from error


def _helper(args: list[str], timeout: int = 120, allow_fail: bool = False) -> subprocess.CompletedProcess:
    if not installed():
        raise HTTPException(
            503, "170tune가 설치되어 있지 않습니다 (git clone https://github.com/cachenetics/170tune /opt/170tune && sudo ./install.sh)"
        )
    result = _run(["sudo", "-n", HELPER] + args, timeout=timeout)
    if result.returncode and not allow_fail:
        detail = (result.stderr or result.stdout or "").strip()
        raise HTTPException(400, detail or "170tune 실행 실패")
    return result


def _selector(value) -> str:
    text = str(value or "").strip()
    if not (text.isdigit() or re.fullmatch(r"GPU-[A-Za-z0-9-]+", text)):
        raise HTTPException(400, "GPU 식별자가 올바르지 않습니다")
    return text


def read_config(uuid_key: str) -> dict | None:
    """UUID별 저장된 170tune 프로필 (root 0644 — flux가 읽기 가능)."""
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid_key):
        return None
    path = CONFIG_DIR / f"{uuid_key}.conf"
    if not path.is_file():
        return None
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    except OSError:
        return None
    if not values:
        return None
    out = {
        "uuid": uuid_key,
        "method": values.get("METHOD", "170tune"),
        "enabled": values.get("TUNE_ENABLED", "1") != "0",
        "profile": values.get("PROFILE", "") or None,
    }
    for key in ("OFFSET", "CLK", "PL", "NDIV"):
        try:
            out[key.lower()] = int(values.get(key, 0) or 0)
        except ValueError:
            out[key.lower()] = 0
    return out


def parse_status(text: str) -> dict:
    """`170tune status` 출력(key: value + nvidia-smi CSV)을 구조화."""
    out: dict = {
        "ok": True, "warnings": [],
        "serial": None, "mem_clock_mhz": None, "mem_source": None, "offset_mhz": None,
        "clk_max_sm": None, "clk_current_sm": None, "mem_clock_current": None,
        "power_limit": None, "power_draw": None, "temp_gpu": None, "temp_mem": None,
        "mem_total": None, "raw": text,
    }
    csv_line = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^serial\s*:\s*(\S+)", stripped)
        if m:
            out["serial"] = m.group(1)
            continue
        m = re.match(r"^memory clock\s*:\s*(\d+)\s*MHz\s*\[?(\w+)?\]?", stripped)
        if m:
            out["mem_clock_mhz"] = int(m.group(1))
            out["mem_source"] = m.group(2)
            continue
        m = re.match(r"^offset\s*:\s*([+-]?\d+)\s*MHz", stripped)
        if m:
            out["offset_mhz"] = int(m.group(1))
            continue
        if "WEDGED" in stripped or "TAKES NO CUDA CONTEXT" in stripped:
            out["ok"] = False
            out["warnings"].append(stripped)
            continue
        if re.match(r"^\d+(\.\d+)?\s*(MHz|W|MiB)", stripped):
            csv_line = stripped
    if csv_line:
        # clocks.max.sm, clocks.current.sm, clocks.current.memory, power.limit,
        # power.draw, temperature.gpu, temperature.memory, memory.total
        fields = [f.strip().split()[0] for f in csv_line.split(",")]
        keys = ["clk_max_sm", "clk_current_sm", "mem_clock_current",
                "power_limit", "power_draw", "temp_gpu", "temp_mem", "mem_total"]
        for key, raw in zip(keys, fields):
            try:
                value = float(raw)
                out[key] = int(value) if value == int(value) else value
            except ValueError:
                pass
    return out


def parse_preflight(text: str, returncode: int) -> dict:
    """`170tune preflight` 체크리스트를 [{ok, item, detail}] 로 구조화."""
    items: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^\[(ok|FAIL)\]\s+(.*)$", stripped)
        if m:
            items.append({"ok": m.group(1) == "ok", "item": m.group(2), "detail": ""})
            continue
        if stripped.startswith("info") and items:
            items[-1]["detail"] = (items[-1]["detail"] + " " + stripped).strip()
            continue
        if items and line[:1] in (" ", "\t") and not line.strip().startswith(("ok", "FAIL", "[", "info")):
            items[-1]["detail"] = (items[-1]["detail"] + "\n" + stripped).strip()
            continue
    return {"ok": returncode == 0 and all(item["ok"] for item in items), "items": items, "raw": text}


def status(selector) -> dict:
    result = _helper(["status", _selector(selector)], timeout=60)
    return parse_status(result.stdout)


def preflight(selector) -> dict:
    result = _helper(["preflight", _selector(selector)], timeout=120, allow_fail=True)
    return parse_preflight(result.stdout, result.returncode)


def snapshot_stock(selector) -> dict:
    """이 카드의 현재(=stock) 값을 revert baseline으로 기록 — 한 번만 필요."""
    result = _helper(["snapshot-stock", _selector(selector)], timeout=60)
    return {"ok": True, "message": (result.stdout or "").strip()}


def recover(selector) -> dict:
    """카드를 stock으로 돌려놓는 복구 경로 (wedge/컨텍트 고사 시)."""
    result = _helper(["recover", _selector(selector)], timeout=300)
    return {"ok": True, "message": (result.stdout or result.stderr or "").strip()}


def set_profile(selector, profile: str, offset: int, clk: int, pl: int, ndiv: int) -> dict:
    """170tune 프로필을 live 적용 + 부팅 저장. named=테이블 값, custom=입력값."""
    if profile not in VALID_PROFILES:
        raise HTTPException(400, f"지원하지 않는 프로파일: {profile}")
    if profile != "custom":
        table = PROFILES[profile]
        offset = table["offset"] or 0
        clk = table["clk"] or 0
        pl = table["pl"] or 0
    offset = int(offset)
    clk = int(clk)
    pl = int(pl)
    ndiv = int(ndiv)
    if not 0 <= offset <= 450:
        raise HTTPException(400, "SM offset은 0~450MHz여야 합니다 (+450 초과가 하드크래시킴)")
    if clk not in (0,) and not 210 <= clk <= 5000:
        raise HTTPException(400, "클럭 상한은 210~5000MHz여야 합니다 (0=없음)")
    if not 0 <= pl <= 500:
        raise HTTPException(400, "power limit은 0~500W여야 합니다 (0=170tune 기본 300W)")
    if not 0 <= ndiv <= 80:
        raise HTTPException(400, f"NDIV는 0(변경 안 함)~80여야 합니다 (stock {STOCK_NDIV})")
    result = _helper(
        ["set", _selector(selector), profile, str(offset), str(clk), str(pl), str(ndiv)],
        timeout=180,
    )
    return {"ok": True, "message": (result.stdout or "").strip()}


def reset(selector) -> dict:
    """stock SM 포인트로 리셋 (offset 0, 클럭 잠금 해제, 250W) + best-effort HBM 리셋."""
    result = _helper(["reset", _selector(selector)], timeout=180)
    return {"ok": True, "message": (result.stdout or result.stderr or "").strip()}


def method(selector, choice: str) -> dict:
    """GPU별 부팅 방식 선택 (nvidia-smi | 170tune) — 현재 실행값은 변경 안 함."""
    if choice not in ("nvidia-smi", "170tune"):
        raise HTTPException(400, "method는 nvidia-smi 또는 170tune이어야 합니다")
    result = _helper(["method", _selector(selector), choice], timeout=30)
    return {"ok": True, "message": (result.stdout or "").strip()}


def profiles() -> dict:
    return {
        "stock_ndiv": STOCK_NDIV,
        "installed": installed(),
        "profiles": {
            name: {
                "label": info["label"], "offset": info["offset"], "clk": info["clk"],
                "pl": info["pl"], "bench": info["bench"], "note": info["note"],
            } for name, info in PROFILES.items()
        },
    }
