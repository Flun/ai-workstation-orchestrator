"""GPU VRAM Arbiter — main_service_new 내부 기능.

정책 요약 (명세: GPU VRAM Arbiter 구현 명세)
  * 사용자는 GPU 단위로만 ON/OFF를 고른다. ON인 GPU의 ComfyUI/llama.cpp는 자동 관리 대상.
  * idle을 이유로 unload하지 않는다. RESIDENT는 VRAM 캐시로 계속 유지한다.
  * 실제 부족이 발생할 때만 RESIDENT를 eviction하고, ACTIVE 작업은 강제 중단하지 않는다.
  * 판단의 최종 기준은 NVML이다. 미관리 CUDA 프로세스에는 개입하지 않지만 사용량은 반영된다.
  * GPU마다 독립 lock/queue를 쓰며 전역 lock은 존재하지 않는다.

설치 위치: /home/flux/main_server_new/vram_arbiter.py (독립 데몬 아님)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request
from starlette.responses import JSONResponse, StreamingResponse

import subprocess

from config import BASE_DIR, IS_WINDOWS

ARBITER_SETTINGS_FILE = os.path.join(BASE_DIR, "vram_arbiter_settings.json")
ARBITER_EVENTS_FILE = os.path.join(BASE_DIR, "gpu_arbiter_events.json")

UNLOADED, RESIDENT, ACTIVE = "UNLOADED", "RESIDENT", "ACTIVE"
LOADING, UNLOADING, ERROR = "LOADING", "UNLOADING", "ERROR"
SUPPORTED_TYPES = ("comfy", "llama")
STATE_LABELS = {
    UNLOADED: "내려감", RESIDENT: "상주(캐시)", ACTIVE: "작업 중",
    LOADING: "올리는 중", UNLOADING: "내리는 중", ERROR: "오류",
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "enabled": True,                     # 마스터 토글 (꺼지면 모든 GPU 우회)
    # GPU 도메인별 설정. 각 GPU 는 서로 완전히 독립적인 스케줄링 단위이다 — GPU0 의
    # 요청/eviction 은 GPU0 에 속한 managed 서비스만 대상으로 하고 GPU1 은 상태·free·lock·
    # victim 을 전혀 참조하지 않는다.
    # evict_threshold_percent: **이번 요청(expected) 의 크기가 이 GPU 총 VRAM 의 몇 % 인지**
    #   기준. 예) 24GB GPU + 50 → 예상 12GB 이상인 요청만 "큰 요청"으로 보고 resident eviction
    #   을 허용한다. free 가 충분하면 threshold 와 무관하게 eviction 하지 않는다.
    #   0 = 부족하면 항상 eviction (전통 동작, 안전 기본값). 높게 잡으면 상주 모델을 더 지켜주고
    #   그 대신 백엔드 자체 offload/OOM 에 맡긴다.
    "gpus": {},                          # {"GPU-xxx": {enabled, safety_margin_mb, evict_threshold_percent, index, name}}
    "default_evict_threshold_percent": 0,
    "default_safety_margin_mb": 2048,
    "admission_timeout_sec": 900,        # ACTIVE 작업 완료까지 기다리는 상한 (§11)
    "unload_timeout_sec": 60,            # unload 후 NVML free 증가 대기 (§27)
    "llama_unload_mode": "protect",      # protect | router_only | restart
    "active_util_percent": 5,            # NVML util 임계값 이상이면 보수적으로 ACTIVE 취급
    "expected_vram_mb": {},              # 선택적 오버라이드만. 미설정이어야 정상 (§15)
    "learned_peak_mb": {},               # 실측 peak 자동 학습 (reconcile 이 갱신)
    # §15: 사용자가 값을 채워야 동작하는 구조를 금지한다. 우선순위:
    #   config 오버라이드 → learned(최근 실측 peak) → llama 는 GGUF 파일크기 → UNKNOWN.
    # UNKNOWN 은 비율 기반 추정치를 쓰지 않는다. 대형 Comfy 워크플로우에서 35% 같은 값은
    # 과소추정 위험이 크므로, 학습값이 없는 첫 실행에서는 **보수적으로 resident victim 을
    # 모두 내려놓고** 실제 peak 를 학습시킨다. 이후 admission 부터는 learned 값으로
    # 공존 여부를 최적화한다 (§12).
    "unknown_clear_residents": True,
    # 훅이 통과한 뒤 Comfy 가 큐잉→queue_running 에 잡히기 전까지의 짧은 공백을 보호한다.
    # 그 구간에는 ACTIVE 로 안 잡혀 victim 이 될 수 있어, 통과 시점부터 이 시간만 홀드를 연장한다.
    "post_admit_grace_sec": 30,
    # victim 이 하나도 없어 내릴 게 없는 경우에만 쓰는 최종 하한 (부탁 불가능한 요구를 막는다).
    "unknown_expected_min_mb": 4096,
    # ---- paired llama handoff (§7 실제 사용 시나리오) ------------------------------------
    # Comfy workflow 안의 JH llama 노드가 같은 GPU 의 llama 를 호출한다. on_prompt 시점에
    # llama 를 내리면 안 된다 — b10685 라우터는 unload 된 모델도 status.value="unloaded" 로
    # /v1/models 에 남기 때문에 JH 의 model="auto" 해석은 그대로 성공하고, 곧 이은
    # POST /v1/chat/completions 에서 라우터가 autoload 로 같은 모델을 다시 올린다
    # (server-models.cpp proxy_post → ensure_model_ready). 즉 on_prompt eviction 은 헛수고이고,
    # 그 경로에는 pending 이 없어 handoff 가 no-pending 으로 끝나 llama 가 RESIDENT 로 돌아가
    # 다음 Comfy 모델 로드와 다시 contention 한다.
    # 그래서 on_prompt 는 "수요 등록" 만 하고, JH 노드가 llama 응답을 다 받은 직후 (return 직전)
    # handoff 가 실제 unload 를 수행한다.
    "paired_handoff_enabled": True,
    # pending demand 의 유통명령. 워크플로우가 handoff 없이 끝나도 상주가 영구 잠기지 않게 한다.
    "pending_demand_ttl_sec": 900,
    "exclude_services": [],              # §24 고급 예외 (기본 미사용)

    # ---- llama.cpp Router mode (A안) -------------------------------------------
    # 라우터 모드는 라우터 프로세스를 유지한 채 모델 VRAM만 반환하는 유일한 경로다.
    # 단 첫 버전은 모델을 1개만 등록한다 — /v1/models data[0] 에 의존하는 클라이언트
    # (ComfyUI JH 노드의 model="auto") 와 section name 순서가 보장되지 않기 때문이다.
    "llama_router_enabled": True,        # ON: 다음 llama.cpp 시작부터 라우터 모드로 실행
    "llama_router_preset_path": "",      # 빈 값이면 main_server_new/llama_router_presets.ini
    "llama_router_autoload": True,       # §3: --no-models-autoload 를 붙이지 않는다
    "llama_router_model_name": "",       # INI section name = canonical 모델명. 비면 GGUF 파일명에서 유도
    # 다중 alias(alias = a, b, c) 는 런타임 검증 전까지 쓰지 않는다. 기본 OFF.
    "llama_router_use_alias": False,
    "llama_router_aliases": "",
    # 미머지 PR #19855 기능이라 미지원 바이너리에서는 사용하지 않는다. 머지된 빌드로
    # 업그레이드되면 --help 검사로 자동 감지해 INI 에 쓴다.
    "llama_inject_missing_model": False,  # 게이트가 model 필드를 대신 채울지 (A안 기본 OFF)
}

# llama_unload_mode 의미
#   protect     : 프로세스를 죽이지 않는다. router 모드(/models/unload)면 모델만 내리고,
#                 아니면 unload 불가로 표시한다 (권장 기본값, §17/§29-15).
#   router_only : protect와 동일하지만 unload 불가 시 요청도 즉시 실패시킨다.
#   restart     : 반환이 필요하면 llama-server를 재시작한다 (명시적 옵트인).

ROUTER_PRESET_FILENAME = "llama_router_presets.ini"
# default-model 은 PR #19855(llama.cpp)에서 제안된 기능으로 아직 머지되지 않았고,
# 이 머신의 b10665 바이너리 strings 와 /opt/llama/.src/b10665 소스에 존재하지 않는다.
# 미지원 바이너리의 INI 에 쓰면 common/preset.cpp 의 unknown key throw 로 기동 자체가
# 실패하므로, --help 검사로 지원이 확인된 빌드에서만 사용한다.
DEFAULT_MODEL_ENV = "LLAMA_ARG_DEFAULT_MODEL"

_llama_feature_cache: Dict[str, Dict[str, bool]] = {}

router = APIRouter(prefix="/api/gpu-arbiter", tags=["gpu-arbiter"])
proxy_router = APIRouter(prefix="/arbiter", tags=["gpu-arbiter-proxy"])


class VramNotAvailable(RuntimeError):
    """명세 §21의 VRAM_NOT_AVAILABLE."""

    def __init__(self, message: str, *, gpu_uuids: List[str], free_mb: int, required_mb: int,
                 blocked_by: Optional[List[str]] = None, deficits: Optional[Dict[str, int]] = None):
        super().__init__(message)
        self.gpu_uuids = gpu_uuids
        self.free_mb = free_mb
        self.required_mb = required_mb
        self.blocked_by = blocked_by or []
        self.deficits = deficits or {}   # GPU 별 부족량 (§3 placement 판단 결과)


# --------------------------------------------------------------------- NVML 계층 (§14)

_NVML_LOCK = threading.Lock()
_NVML_STATE: Dict[str, Any] = {"lib": None, "failed": False}


def _nvml():
    if _NVML_STATE["failed"]:
        return None
    try:
        import pynvml  # nvidia-ml-py — main_service_new venv에만 설치되어 있다
    except Exception:
        _NVML_STATE["failed"] = True
        return None
    if _NVML_STATE["lib"] is None:
        with _NVML_LOCK:
            if _NVML_STATE["lib"] is None:
                try:
                    pynvml.nvmlInit()
                    _NVML_STATE["lib"] = pynvml
                except Exception as error:  # 드라이버 미준비/권한
                    _NVML_STATE["failed"] = True
                    print(f"[arbiter] NVML init 실패: {error}")
                    return None
    return _NVML_STATE["lib"]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def llama_binary_features(binary_path: str) -> Dict[str, bool]:
    """llama-server 바이너리의 기능 지원 여부를 --help 로 검사해 캐시한다.

    별도 프로세스로 --help 만 실행하고 즉시 끝나므로 GPU 를 건드리지 않고,
    실행 중인 llama-server 에 요청을 보내지 않는다. 결과는 경로별로 캐시한다.

    default-model(PR #19855) 은 아직 머지되지 않았고 이 머신의 b10665 에도 없다.
    미지원 바이너리의 INI 에 default-model = true 를 넣으면 preset 로더가
    "option 'default-model' not recognized" 로 기동을 abort 하기 때문에,
    반드시 지원 여부 확인 후에만 파일에 쓴다.
    """
    # subprocess 는 함수 지역 import 없이 모듈 상단에서 사용한다.
    key = str(binary_path or "")
    if not key:
        return {"router": False, "default_model": False}
    cached = _llama_feature_cache.get(key)
    if cached is not None:
        return cached
    features = {"router": False, "default_model": False}
    try:
        result = subprocess.run([key, "--help"], capture_output=True, text=True, timeout=20,
                                creationflags=(subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0))
        help_text = (result.stdout or "") + (result.stderr or "")
        features["router"] = "--models-dir" in help_text or "router server" in help_text
        features["default_model"] = "default-model" in help_text
    except Exception as error:
        print(f"[arbiter] llama 바이너리 기능 검사 실패 ({key}): {error}")
    _llama_feature_cache[key] = features
    return features


def router_preset_path() -> str:
    configured = str(arbiter.setting("llama_router_preset_path") or "").strip()
    if configured:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
    return os.path.join(BASE_DIR, ROUTER_PRESET_FILENAME)


def sanitize_section_name(value: str) -> str:
    """INI section name 을 안전한 문자로만 만든다 (라우터 canonical 모델명과 동일하게 쓰인다)."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return name or "main"


def router_section_name(model_path: str) -> str:
    """preset INI section name = 라우터의 canonical 모델 이름.

    A안: 다중 alias 는 런타임 검증 전까지 쓰지 않고, 기존 클라이언트가 실제로 보내는
    model 값에 section name 자체를 맞춘다. 예) "model":"local-model" → [local-model]
    지정이 없으면 GGUF 파일명에서 유도한다 — ComfyUI JH 노드처럼 /v1/models 의
    data[0].id 를 읽는 클라이언트는 이름과 무관하게 자동으로 맞는다.
    """
    configured = str(arbiter.setting("llama_router_model_name") or "").strip()
    if configured:
        return sanitize_section_name(configured)
    base = os.path.splitext(os.path.basename(str(model_path or "")))[0]
    return sanitize_section_name(base)


def write_router_preset(entries: List[Dict[str, Any]], *, native_default_model: bool,
                        common_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """라우터 preset INI 를 생성한다 (A안: 첫 버전은 모델 1개).

      version = 1

      [local-model]
      model = /path/to/model.gguf

    원칙:
      * section name 이 곧 canonical 모델명이다 → 별칭 해석 없이 클라이언트 값과 일치시킨다.
      * alias 는 llama_router_use_alias=True 일 때만 쓴다 (기본 OFF, 런타임 검증 전).
      * default-model 은 미지원 바이너리에 절대 쓰지 않는다 — unknown key 로 기동 실패한다.
        지원 빌드에서도 개별 model preset 에만 넣는다 ([*] global 금지).
      * [*] common_args 는 기본 생성하지 않는다. ctx/flash/ngl 같은 값은 라우터 CLI 로
        넘기면 모든 자식 모델에 overlay 되므로(server-models.cpp preset.merge(base_preset)),
        INI 를 이중으로 적어 서로 덮는 상황을 피한다.
    """
    path = router_preset_path()
    lines = ["version = 1", ""]
    if common_args:
        lines.append("[*]")
        for arg in common_args:
            text = str(arg).strip()
            if not text.startswith("-") or text.startswith("--models"):
                continue
            key = text.lstrip("-").replace("_", "-")
            if "=" in key:
                key, value = key.split("=", 1)
                lines.append(f"{key} = {value}")
            else:
                lines.append(key)
        lines.append("")
    default_name = ""
    written = 0
    use_alias = bool(arbiter.setting("llama_router_use_alias", False))
    for entry in entries:
        name = sanitize_section_name(entry.get("name") or "")
        model = str(entry.get("model") or "").strip()
        if not model:
            continue
        lines.append(f"[{name}]")
        lines.append(f"model = {model}")
        mmproj = str(entry.get("mmproj") or "").strip()
        if mmproj and os.path.isfile(mmproj):
            lines.append(f"mmproj = {mmproj}")
        alias = str(entry.get("alias") or "").strip() if use_alias else ""
        if alias:
            # 다중 alias: 소스상 comma-split 지원(server-models.cpp add_model) 하지만
            # 이 머신에서 아직 런타임 확인 전이므로 기본으로 쓰지 않는다.
            lines.append(f"alias = {alias}")
        if entry.get("default") and native_default_model:
            lines.append("default-model = true")   # 개별 preset 에만. [*] 에 넣으면 기동 실패
        if entry.get("default"):
            default_name = name
        lines.append("")
        written += 1
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
    except OSError as error:
        raise RuntimeError(f"라우터 preset 파일 생성 실패: {error}") from error
    return {"path": path, "default_model": default_name,
            "native_default_model": bool(native_default_model), "entries": written}


def read_gpu_snapshot() -> Dict[str, Dict[str, Any]]:
    """{uuid: {index,name,pci_bus_id,total_mb,used_mb,free_mb}} — NVML 우선, nvidia-smi 폴백."""
    pynvml = _nvml()
    result: Dict[str, Dict[str, Any]] = {}
    if pynvml is not None:
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                uuid = _text(pynvml.nvmlDeviceGetUUID(handle))
                try:
                    pci = _text(pynvml.nvmlDeviceGetPciInfo(handle).busId)
                except Exception:
                    pci = ""
                try:
                    util = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                except Exception:
                    util = None
                result[uuid] = {
                    "index": index,
                    "name": _text(pynvml.nvmlDeviceGetName(handle)),
                    "pci_bus_id": pci,
                    "total_mb": int(memory.total // 1048576),
                    "used_mb": int(memory.used // 1048576),
                    "free_mb": int(memory.free // 1048576),
                    "util": util,
                }
            if result:
                return result
        except Exception as error:
            print(f"[arbiter] NVML 스냅샷 실패, nvidia-smi 폴백: {error}")
    from gpu import get_gpus  # 이미 있는 NVML/nvidia-smi 이중 백엔드 재사용

    for gpu in get_gpus() or []:
        uuid = str(gpu.get("uuid") or "")
        if not uuid:
            continue
        result[uuid] = {
            "index": gpu.get("index"),
            "name": gpu.get("name") or "",
            "pci_bus_id": gpu.get("pci_bus_id") or "",
            "total_mb": int(gpu.get("vram_total") or 0),
            "used_mb": int(gpu.get("vram_used") or 0),
            "free_mb": int(gpu.get("vram_free") or 0),
            "util": gpu.get("util"),
        }
    return result


def read_process_map() -> Dict[str, Dict[int, int]]:
    """{gpu_uuid: {pid: used_mb}} — NVML compute apps. PID→GPU 매핑의 정답 (§6)."""
    pynvml = _nvml()
    result: Dict[str, Dict[int, int]] = {}
    if pynvml is not None:
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                uuid = _text(pynvml.nvmlDeviceGetUUID(handle))
                procs: Dict[int, int] = {}
                infos: List[Any] = []
                for name in ("nvmlDeviceGetComputeRunningProcesses_v3",
                             "nvmlDeviceGetComputeRunningProcesses_v2",
                             "nvmlDeviceGetComputeRunningProcesses"):
                    func = getattr(pynvml, name, None)
                    if func is None:
                        continue
                    try:
                        infos = func(handle) or []
                        break
                    except Exception:
                        continue
                for info in infos:
                    pid = int(getattr(info, "pid", 0) or 0)
                    used = int((getattr(info, "usedGpuMemory", 0) or 0) // 1048576)
                    if pid:
                        procs[pid] = max(procs.get(pid, 0), used)
                result[uuid] = procs
            return result
        except Exception as error:
            print(f"[arbiter] NVML 프로세스 목록 실패, nvidia-smi 폴백: {error}")
    from gpu import scan_vram_processes

    for proc in scan_vram_processes(force=True) or []:
        uuid, pid, used = proc.get("gpu_uuid"), proc.get("pid"), proc.get("used_mb")
        if uuid and pid and used is not None:
            result.setdefault(uuid, {})[int(pid)] = int(used)
    return result





# --------------------------------------------------------------------- 데이터 구조 (§20)

@dataclass
class GpuPool:
    """GPU별 독립 Resource Pool. GPU 0 contention이 GPU 1에 영향 주지 않는다 (§7)."""

    uuid: str
    index: Optional[int]
    name: str
    pci_bus_id: str = ""
    enabled: bool = False
    safety_margin_mb: int = 2048
    # 이번 요청 expected / 이 GPU total * 100 이 이 값 이상일 때만 resident eviction 허용.
    # (요청 크기 기준 — GPU 사용률이 아니다.) 0 = free 부족 시 항상 eviction.
    evict_threshold_percent: int = 0
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    unmanaged_mb: int = 0
    util: Optional[int] = None   # ACTIVE 보수 판정용 (§11), read_gpu_snapshot에서 함께 읽는다
    observed_at: float = 0.0
    waiting_requests: int = 0
    # RLock를 쓰는 이유: 같은 스레드가 GPU lock을 쥔 채 Condition.notify()를 호출한다.
    # 비재귀 Lock이면 __exit__의 `with pool.cond`이 자기 자신의 lock을 다시 획득하려 해
    # 데드락이 난다. threading.Condition는 RLock와 짝을 이룰 때만 안전하게 동작한다.
    lock: Any = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        self.cond = threading.Condition(self.lock)

    def apply(self, info: Dict[str, Any]) -> None:
        self.index = info.get("index", self.index)
        self.name = info.get("name") or self.name
        self.pci_bus_id = info.get("pci_bus_id") or self.pci_bus_id
        self.total_mb = int(info.get("total_mb") or 0)
        self.used_mb = int(info.get("used_mb") or 0)
        self.free_mb = int(info.get("free_mb") or 0)
        if "util" in info:
            self.util = info.get("util")
        self.observed_at = time.time()




@dataclass
class ServiceState:
    id: str
    type: str                       # comfy | llama
    label: str = ""
    gpu_uuids: List[str] = field(default_factory=list)
    backend: str = ""
    port: Optional[int] = None
    pid: Optional[int] = None
    state: str = UNLOADED
    active_requests: int = 0
    expected_vram_mb: int = 0
    current_vram_mb: int = 0
    last_used_at: float = field(default_factory=time.time)
    last_error: str = ""
    model: str = ""
    router_capable: Optional[bool] = None
    unload_supported: bool = True
    estimate_source: str = "none"
    busy_reported: bool = False          # llama /slots is_processing
    model_loaded_reported: bool = True   # llama /models status
    slots_present: bool = False
    # Router mode 상세 (A안)
    router_models: List[Dict[str, Any]] = field(default_factory=list)  # [{id,status}]
    default_model: str = ""              # INI section name (= 라우터 canonical 모델명)
    native_default_model: bool = False   # 바이너리가 default-model 을 지원하는지
    autoload_enabled: bool = True        # 라우터가 요청 시 자동 로드하는 상태
    gpu_pids: List[int] = field(default_factory=list)  # 라우터 부모 + 모델 자식 PID (VRAM 합산)
    # GPU 도메인 (§7 단순화): main_server 가 서비스를 시작할 때 이미 아는 type/port/GPU 로
    # 정하는 "소속 GPU". scheduling 은 이 도메인 안에서만 이루어진다.
    home_gpu: str = ""
    multi_gpu: bool = False              # 여러 GPU 에 걸친 인스턴스 (llama 만 해당)
    managed_evict: bool = True           # False 면 eviction 관리 제외 (victim · requester 모두)
    domain_note: str = ""                # 상태 API 로 내보내는 도메인 비고 (제외 사유 등)
    # in-process 게이트 홀드: 훅이 블로킹 대기 중이거나 요청이 진행 중임을 나타낸다.
    # 이 기간에는 reconcile 이 이 서비스를 RESIDENT 로 격하하지 않는다 (victim 제외).
    holds: int = 0
    hold_until: float = 0.0

    @property
    def managed(self) -> bool:
        return self.type in SUPPORTED_TYPES


# --------------------------------------------------------------------- Arbiter 본체

class VramArbiter:
    def __init__(self) -> None:
        self.pools: Dict[str, GpuPool] = {}
        self.services: Dict[str, ServiceState] = {}
        self.settings: Dict[str, Any] = json.loads(json.dumps(DEFAULT_SETTINGS))
        self.events: List[Dict[str, Any]] = []
        self._settings_lock = threading.Lock()
        self._events_lock = threading.Lock()
        self.gpu_services: Dict[str, List[str]] = {}
        self._pid_to_gpus: Dict[int, List[str]] = {}
        # Comfy 인스턴스별 upcoming VRAM demand ({comfy_service_id: {registered_at, expires_at,
        # llama_ports, planned, handoffs, expected_mb, estimate_source, prompt_id}}).
        self.pending: Dict[str, Dict[str, Any]] = {}
        self._pending_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started_at = 0.0
        self._last_reconcile = 0.0
        # app.py가 주입하는 훅 (참조 방향은 app -> arbiter 단방향으로 유지)
        self.comfy_instances: Callable[[], List[str]] = lambda: ["main"]
        self.comfy_port: Callable[[str], int] = lambda instance: 8188
        self.comfy_device: Callable[[str], str] = lambda instance: ""
        self.comfy_label: Callable[[str], str] = lambda instance: f"ComfyUI {instance}"
        self.llama_port: Callable[[], int] = lambda: 8080
        # main_server 가 llama.cpp 를 시작할 때 지정한 GPU 목록 (UUID/index/CUDA1 표식 혼용).
        # NVML 실측이 아니라 이 기동 정보로 소속 GPU 를 정한다 (§7 도메인).
        self.llama_devices: Callable[[int], List[str]] = lambda port: []
        self.service_running: Callable[[str], bool] = lambda name: False
        self.llama_restart: Optional[Callable[[], bool]] = None

    # ---------------- 설정 (§23)

    def _migrate_settings(self, merged: Dict[str, Any], stored: Optional[Dict[str, Any]] = None) -> bool:
        """구버전 설정 파일을 현재 스키마로 백필한다. 반환값 = 수정 발생 여부.

        신규 키는 DEFAULT_SETTINGS 에서 이미 채워지므로, 여기서는 (1) GPU 항목별 필드
        타입/범위를 정규화하고 (2) 사라진 키를 정리하고 (3) 스냅샷에 없는 GPU 항목을 채운다.
        enabled 는 사용자 선택이므로 절대 바꾸지 않는다.

        `stored`(디스크에 있던 원본) 도 함께 받는 이유: 현재 스키마에 없는 키는 merged 로
        복사되지 않아 merged 만 검사하면 디스크의 고아 키가 영구히 남는다. 그래서 저장 여부를
        판단할 때 stored 에만 있는 키를 함께 본다.
        """
        changed = False
        if isinstance(stored, dict):
            for orphan in [k for k in stored if k not in merged]:
                print(f"[arbiter] 설정에서 더 이상 쓰지 않는 키를 제거합니다: {orphan}")
                changed = True
        default_margin = int(merged.get("default_safety_margin_mb") or 2048)
        default_threshold = self._clamp_percent(merged.get("default_evict_threshold_percent", 0))
        if merged.get("default_evict_threshold_percent") != default_threshold:
            merged["default_evict_threshold_percent"] = default_threshold
            changed = True
        # 구버전에 있다가 현재 구조에서 사라진 키 (값이 남아도 아무도 읽지 않는다)
        for stale_key in ("llama_router_default_model", "unknown_expected_ratio",
                          "multi_gpu_deficit_mode", "cross_gpu_evict"):
            if stale_key in merged:
                merged.pop(stale_key, None)
                changed = True
        gpus = merged.get("gpus")
        if not isinstance(gpus, dict):
            merged["gpus"] = {}
            gpus = merged["gpus"]
            changed = True
        known = {"enabled", "safety_margin_mb", "evict_threshold_percent", "index", "name"}
        for uuid, entry in list(gpus.items()):
            if not isinstance(entry, dict):
                gpus[uuid] = {"enabled": bool(entry)}
                changed = True
                continue
            for extra in [k for k in entry if k not in known]:
                entry.pop(extra, None)
                changed = True
            if "enabled" not in entry:
                entry["enabled"] = False
                changed = True
            if "safety_margin_mb" not in entry:
                entry["safety_margin_mb"] = default_margin
                changed = True
            if "evict_threshold_percent" not in entry:
                entry["evict_threshold_percent"] = default_threshold
                changed = True
            clamped = self._clamp_percent(entry.get("evict_threshold_percent"))
            if entry.get("evict_threshold_percent") != clamped:
                entry["evict_threshold_percent"] = clamped
                changed = True
            try:
                entry["safety_margin_mb"] = max(0, int(entry.get("safety_margin_mb") or 0))
            except (TypeError, ValueError):
                entry["safety_margin_mb"] = default_margin
                changed = True
        for key in ("paired_handoff_enabled", "unknown_clear_residents"):
            merged[key] = bool(merged.get(key, DEFAULT_SETTINGS[key]))
        try:
            merged["pending_demand_ttl_sec"] = max(30.0, float(merged.get(
                "pending_demand_ttl_sec", DEFAULT_SETTINGS["pending_demand_ttl_sec"])))
        except (TypeError, ValueError):
            merged["pending_demand_ttl_sec"] = DEFAULT_SETTINGS["pending_demand_ttl_sec"]
            changed = True
        return changed

    @staticmethod
    def _clamp_percent(value: Any) -> int:
        try:
            return max(0, min(100, int(value)))
        except (TypeError, ValueError):
            return 0

    def load_settings(self) -> Dict[str, Any]:
        merged = json.loads(json.dumps(DEFAULT_SETTINGS))
        stored: Optional[Dict[str, Any]] = None
        try:
            if os.path.exists(ARBITER_SETTINGS_FILE):
                with open(ARBITER_SETTINGS_FILE, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    stored = loaded
                    for key, value in stored.items():
                        if key in ("gpus", "expected_vram_mb", "learned_peak_mb") and isinstance(value, dict):
                            merged[key].update(value)
                        elif key == "exclude_services" and isinstance(value, list):
                            merged[key] = [str(item) for item in value]
                        elif key in merged:
                            merged[key] = value
        except Exception as error:
            print(f"[arbiter] 설정 로드 실패: {error}")
        migrated = self._migrate_settings(merged, stored)
        with self._settings_lock:
            self.settings = merged
        if migrated:
            # 백필 결과를 디스크에 반영해야 UI 가 첫 로드부터 threshold 를 올바른 기본값으로
            # 보여주고, 이후 부분 수정(PATCH)이 빈 값에서 시작하지 않는다.
            try:
                self.save_settings()
                print("[arbiter] 설정 파일을 현재 스키마로 마이그레이션 했습니다")
            except Exception as error:
                print(f"[arbiter] 마이그레이션 저장 실패 (동작에는 지장 없음): {error}")
        return merged

    def save_settings(self) -> Dict[str, Any]:
        with self._settings_lock:
            payload = json.loads(json.dumps(self.settings))
        try:
            with open(ARBITER_SETTINGS_FILE, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as error:
            raise HTTPException(500, f"Arbiter 설정 저장 실패: {error}") from error
        return payload

    def rebuild_registry(self, detected: Optional[Dict[str, ServiceState]] = None) -> Dict[str, List[str]]:
        """GPU 별 managed service registry 를 다시 만든다 (§7).

        소속 기준은 home_gpu — main_server 가 기동 시점에 안 서비스 정보(type/port/selected GPU)로
        정한 값이고, NVML 실측 점유 카드가 아니다. eviction 관리에서 제외된 멀티 GPU llama 는
        어느 도메인 목록에도 들지 않는다.
        """
        table = self.services if detected is None else detected
        self.gpu_services = {
            uuid: sorted(sid for sid, st in table.items()
                         if st.home_gpu == uuid and st.managed_evict)
            for uuid in self.pools
        }
        return self.gpu_services

    def managed_services(self, gpu_uuid: str) -> List[str]:
        """해당 GPU 도메인의 managed 서비스 id 목록 (§7 독립성)."""
        return list(self.gpu_services.get(gpu_uuid) or [])

    def setting(self, key: str, default: Any = None) -> Any:
        with self._settings_lock:
            value = self.settings.get(key, DEFAULT_SETTINGS.get(key, default))
        return default if value is None else value

    def update_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = payload or {}
        with self._settings_lock:
            for key in ("enabled", "default_safety_margin_mb", "admission_timeout_sec",
                        "unload_timeout_sec", "llama_unload_mode", "active_util_percent",
                        "llama_router_enabled", "llama_router_autoload",
                        "llama_router_use_alias", "llama_inject_missing_model"):
                if key in payload and payload[key] is not None:
                    self.settings[key] = bool(payload[key]) if isinstance(payload[key], bool) else payload[key]
            for key in ("llama_router_preset_path", "llama_router_model_name", "llama_router_aliases",
                        "default_evict_threshold_percent"):
                if key in payload and payload[key] is not None:
                    self.settings[key] = str(payload[key]).strip() if key != "default_evict_threshold_percent" else int(payload[key])
            # GPU 별 설정 부분 수정: {"gpus": {uuid: {enabled, safety_margin_mb, evict_threshold_percent}}}
            gpus_patch = payload.get("gpus")
            if isinstance(gpus_patch, dict):
                table = self.settings.setdefault("gpus", {})
                for uuid, values in gpus_patch.items():
                    if not isinstance(values, dict):
                        continue
                    entry = table.setdefault(str(uuid), {})
                    if "enabled" in values:
                        entry["enabled"] = bool(values["enabled"])
                    if values.get("safety_margin_mb") is not None:
                        entry["safety_margin_mb"] = int(values["safety_margin_mb"])
                    if values.get("evict_threshold_percent") is not None:
                        entry["evict_threshold_percent"] = max(0, min(100, int(values["evict_threshold_percent"])))
            default_evict = payload.get("default_evict_threshold_percent")
            if isinstance(default_evict, int):
                self.settings["default_evict_threshold_percent"] = max(0, min(100, default_evict))
            for key in ("paired_handoff_enabled",):
                if key in payload and payload[key] is not None:
                    self.settings[key] = bool(payload[key])
            if payload.get("pending_demand_ttl_sec") is not None:
                try:
                    self.settings["pending_demand_ttl_sec"] = max(30.0, float(payload["pending_demand_ttl_sec"]))
                except (TypeError, ValueError):
                    pass
            if isinstance(payload.get("exclude_services"), list):
                self.settings["exclude_services"] = [str(x) for x in payload["exclude_services"]]
            if isinstance(payload.get("expected_vram_mb"), dict):
                table = self.settings.setdefault("expected_vram_mb", {})
                for key, value in payload["expected_vram_mb"].items():
                    if value in (None, ""):
                        table.pop(str(key), None)
                    else:
                        try:
                            table[str(key)] = int(float(value))
                        except (TypeError, ValueError):
                            pass
        self.save_settings()
        return {"ok": True}

    def set_gpu_enabled(self, uuid: str, enabled: bool) -> Dict[str, Any]:
        """사용자가 실제로 다루는 유일한 설정: GPU별 ON/OFF (§3)."""
        if not self.pools:
            self.ensure_pools()
        pool = self.pools.get(uuid)
        if pool is None:
            raise HTTPException(404, f"GPU를 찾을 수 없습니다: {uuid}")
        with self._settings_lock:
            entry = self.settings.setdefault("gpus", {}).setdefault(uuid, {})
            entry["enabled"] = bool(enabled)
            entry["index"] = pool.index
            entry["name"] = pool.name
        self.save_settings()
        result = self.reconcile("config")
        self._log_event("enable" if enabled else "disable", uuid=uuid,
                        detail=f"GPU {pool.index} ({pool.name}) Arbiter {'ON' if enabled else 'OFF'}")
        return {"ok": True, "uuid": uuid, "enabled": bool(enabled), "services": result["services"]}

    def gpu_config(self, uuid: str) -> Dict[str, Any]:
        with self._settings_lock:
            return dict((self.settings.get("gpus") or {}).get(uuid) or {})

    # ---------------- 시작 / reconcile (§26)

    def configure(self, **hooks: Callable[..., Any]) -> None:
        for name, hook in hooks.items():
            if hook is not None and hasattr(self, name):
                setattr(self, name, hook)

    def start(self) -> None:
        self.load_settings()
        self._load_events()
        self._started_at = time.time()
        try:
            self.reconcile("startup")
        except Exception as error:
            print(f"[arbiter] 최초 reconcile 실패: {error}")
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-vram-arbiter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(5.0):
            try:
                self.reconcile("poll")
            except Exception as error:
                print(f"[arbiter] reconcile 실패: {error}")

    def ensure_pools(self) -> Dict[str, Dict[str, Any]]:
        """NVML을 다시 읽어 GPU 상태와 ON/OFF를 갱신한다. index는 표시용이고 ON/OFF는 UUID 기준 (§6).

        설정 파일에 아직 없는 GPU 도메인은 여기에서 항목을 만들어 준다 — 재시작 후에도 UI 가
        카드마다 threshold / safety margin 을 바로 편집할 수 있도록 (별도 코드 수정 불필요).
        enabled 는 건드리지 않는다 (기본 OFF = 사용자 선택)."""
        snapshot = read_gpu_snapshot()
        with self._settings_lock:
            configured = dict(self.settings.get("gpus") or {})
            default_margin = int(self.settings.get("default_safety_margin_mb") or 2048)
            master = bool(self.settings.get("enabled"))
        for uuid, info in snapshot.items():
            stored = configured.get(uuid) or {}
            pool = self.pools.get(uuid)
            if pool is None:
                pool = GpuPool(uuid=uuid, index=info["index"], name=info["name"])
                self.pools[uuid] = pool
            pool.apply(info)
            pool.enabled = bool(master and stored.get("enabled", False))
            try:
                pool.safety_margin_mb = int(stored.get("safety_margin_mb") or default_margin)
            except (TypeError, ValueError):
                pool.safety_margin_mb = default_margin
            pool.evict_threshold_percent = self._clamp_percent(stored.get(
                "evict_threshold_percent", self.setting("default_evict_threshold_percent", 0)))
        self._backfill_gpu_settings(snapshot)
        for uuid in list(self.pools):
            if uuid not in snapshot:
                self.pools.pop(uuid, None)
        return snapshot

    def _backfill_gpu_settings(self, snapshot: Dict[str, Any]) -> bool:
        """NVML 스냅샷에만 있고 설정 파일에 없는 GPU 항목을 만든다 (enabled 는 건드리지 않음)."""
        changed = False
        with self._settings_lock:
            table = self.settings.setdefault("gpus", {})
            default_margin = int(self.settings.get("default_safety_margin_mb") or 2048)
            default_threshold = self._clamp_percent(
                self.settings.get("default_evict_threshold_percent", 0))
            for uuid, info in snapshot.items():
                entry = table.get(uuid)
                if not isinstance(entry, dict):
                    entry = table[uuid] = {}
                    changed = True
                if entry.get("index") != info.get("index"):
                    entry["index"] = info.get("index")
                    changed = True
                if entry.get("name") != (info.get("name") or ""):
                    entry["name"] = info.get("name") or ""
                    changed = True
                if "enabled" not in entry:
                    entry["enabled"] = False          # ON/OFF 는 사용자 선택 — 기본 OFF
                    changed = True
                if entry.get("safety_margin_mb") is None:
                    entry["safety_margin_mb"] = default_margin
                    changed = True
                if entry.get("evict_threshold_percent") is None:
                    entry["evict_threshold_percent"] = default_threshold
                    changed = True
        if changed:
            try:
                self.save_settings()
            except Exception:
                pass
        return changed

    def reconcile(self, reason: str = "") -> Dict[str, Any]:
        """NVML → 프로세스 매핑 → Comfy queue → llama capabilities로 상태를 다시 읽는다.

        main_service_new가 재시작돼도 ComfyUI/llama.cpp를 재시작하지 않고 현재 상태만 복제한다 (§26).
        """
        snapshot = self.ensure_pools()
        process_map = read_process_map()
        utilization = {uuid: info.get("util") for uuid, info in snapshot.items()}
        pid_to_gpus: Dict[int, List[str]] = {}
        for uuid, procs in process_map.items():
            for pid in procs:
                pid_to_gpus.setdefault(pid, []).append(uuid)
        self._pid_to_gpus = pid_to_gpus

        excluded = {str(item) for item in (self.setting("exclude_services") or [])}
        detected: Dict[str, ServiceState] = {}

        for instance in self.comfy_instances():
            state = self._detect_comfy(instance, pid_to_gpus, process_map, excluded)
            if state is not None:
                detected[state.id] = state
        for state in self._detect_llama(pid_to_gpus, process_map, excluded):
            detected[state.id] = state

        # 미관리 CUDA 프로세스 사용량: 종료 대상은 아니지만 free 계산에는 포함 (§5)
        # 라우터 부모는 VRAM 을持有하지 않고 모델 자식이持有한다. 그래서 gpu_pids(부모+자식)
        # 전체를 관리 대상으로 봐야, llama 모델 자식이 "외부 CUDA 프로그램"으로 오분류되지 않는다.
        managed_pids = {int(pid) for state in detected.values()
                        for pid in (list(state.gpu_pids) + ([state.pid] if state.pid else []))}
        for uuid, pool in self.pools.items():
            pool.util = utilization.get(uuid)
            unmanaged = 0
            for pid, used in (process_map.get(uuid) or {}).items():
                if pid not in managed_pids:
                    unmanaged += int(used)
            pool.unmanaged_mb = unmanaged

        # §15 자동 학습: 설정 없이도 최근 실측 사용량을 peak 로 기억해 다음 admission 에 쓴다.
        for state in detected.values():
            if state.state in (RESIDENT, ACTIVE) and state.current_vram_mb > 64:
                self._learn(state.id, int(state.current_vram_mb))
        with self._settings_lock:
            self.services = detected
        # GPU 별 managed service registry — main_server 가 기동 시점에 아는 정보를 기준삼아
        # 만든다. 예) gpu0_services=[comfy-8188, llama-8080], gpu1_services=[comfy-8189, llama-8081]
        self.rebuild_registry(detected)
        self._wake_all()
        self._last_reconcile = time.time()
        return {
            "reason": reason,
            "gpus": len(self.pools),
            "services": {key: value.state for key, value in detected.items()},
        }

    def _resolve_uuid(self, token: Any) -> str:
        """main_server 가 기동 시 가진 GPU 표식 → pool UUID.

        설정/기동 정보에는 세 가지 표식이 섞여 들어온다: NVML UUID("GPU-6cd6..."), NVML
        index("0"), CUDA 순번("CUDA1" / "cuda:1"). 라우터 CLI 는 PCI busId 를 쓰기도 한다.
        도메인 판정은 항상 pool UUID 하나로 통일해야 GPU 간 비교가 사라진다.
        """
        text = str(token or "").strip()
        if not text:
            return ""
        if text in self.pools:
            return text
        lowered = text.lower()
        for uuid, pool in self.pools.items():
            if pool.pci_bus_id and lowered == pool.pci_bus_id.lower():
                return uuid
        match = re.search(r"(\d+)\s*$", text)
        if not match or not re.match(r"^(?:\d+|cuda[\s:._-]*\d+|gpu[\s:._-]*\d+)$", lowered):
            return ""
        try:
            index = int(match.group(1))
        except (TypeError, ValueError):
            return ""
        for uuid, pool in self.pools.items():
            if pool.index == index:
                return uuid
        return ""

    def _previous(self, service_id: str) -> Optional[ServiceState]:
        return self.services.get(service_id)

    def _detect_comfy(self, instance: str, pid_to_gpus: Dict[int, List[str]],
                      process_map: Dict[str, Dict[int, int]], excluded: set) -> Optional[ServiceState]:
        service_id = f"comfy-{instance}"
        if service_id in excluded:
            return None
        try:
            port = int(self.comfy_port(instance))
        except Exception:
            return None
        service_name = "comfyui" if instance == "main" else f"comfyui_{instance}"
        pid = self._comfy_pid(instance)
        # 소속 GPU 는 NVML 실측이 아니라 main_server 가 시작 인자로 준 값으로 정한다 (§7).
        # Comfy 는 실질적 multi-GPU 를 고려하지 않으므로, NVML 이 두 카드로 보여도 도메인은
        # 설정된 한 장이다 — 그 경우 gpu_uuids 를 1개로 잘라 다른 GPU victim 을 보는 길을 막는다.
        configured = str(self.comfy_device(instance) or "")
        home = self._resolve_uuid(configured)
        observed = sorted(pid_to_gpus.get(pid, [])) if pid else []
        if not home and len(observed) == 1:
            home = observed[0]          # 설정에 GPU 지정이 없고 실측이 1장일 때만 승인
        uuids = [home] if home else []
        previous = self._previous(service_id)
        state = ServiceState(
            id=service_id, type="comfy", label=self.comfy_label(instance),
            port=port, backend=f"http://127.0.0.1:{port}", pid=pid, gpu_uuids=uuids,
        )
        state.home_gpu = home
        state.multi_gpu = False         # Comfy 는 multi-GPU 시나리오를 관리 대상에서 제외한다
        state.managed_evict = bool(home)
        state.last_used_at = previous.last_used_at if previous else self._started_at or time.time()
        # Comfy 의 동시 작업 수는 queue_running 이 정답이다. on_prompt 훅은 release 를 보내지
        # 않으므로 카운터를 이전값에서 이어받으면 영구 ACTIVE 로 고착된다 → 매 주기 재계산.
        # 단 holds(훅 블로킹 대기) 는 예외로 이어받는다: 이 인스턴스의 /queue 는 훅이 블로킹한
        # 동안 응답할 수 없어 busy=False 로 잘못 읽힌다. 그때 RESIDENT 로 격하되면 다른 서비스의
        # eviction victim 이 될 수 있으므로 홀드 상태만 반드시 계승한다.
        state.active_requests = 0
        state.holds = previous.holds if previous else 0
        state.hold_until = previous.hold_until if previous else 0.0
        state.expected_vram_mb, state.estimate_source = self._expected_for(service_id, "comfy", "", home)
        state.domain_note = ("" if home else "소속 GPU 미확정 — 설정의 GPU 지정을 확인하세요")
        running = bool(pid) and bool(self.service_running(service_name))
        if not running:
            state.state = UNLOADED
            return state
        queue = _http_json("GET", f"{state.backend}/queue", timeout=4.0) or {}
        running = queue.get("queue_running") or []
        held = bool(state.holds) and time.time() < float(state.hold_until or 0)
        busy = bool(running) or held
        state.active_requests = max(len(running), state.holds if held else 0)
        state.current_vram_mb = int(sum(process_map.get(uuid, {}).get(pid, 0) for uuid in uuids)) if pid else 0
        if busy:
            state.state = ACTIVE
            state.last_used_at = time.time()
        elif state.current_vram_mb > 64:
            state.state = RESIDENT          # VRAM에 모델 잔존 = 캐시 (§8)
        else:
            state.state = UNLOADED
        return state

    def _detect_llama(self, pid_to_gpus: Dict[int, List[str]],
                      process_map: Dict[str, Dict[int, int]], excluded: set) -> List[ServiceState]:
        out: List[ServiceState] = []
        # GPU에 올라온 모든 PID를 llama로 오인하면 안 된다. ComfyUI python을 llama로
        # 잡으면 가짜 "llama-8188" 항목이 생기고 eviction 후보로 소모된다.
        # cmdline에 llama-server가 있는 PID만 관리 대상으로 삼고, GPU 매핑은 NVML 결과를 쓴다.
        # 라우터 모드에서는 라우터(parent) 와 모델 자식(child) 이 모두 llama-server 로 보인다.
        # 포트를 리스닝하는 쪽은 parent 이고 VRAM 은 child 가持有하므로, parent 만 인스턴스로
        # 등록하되 GPU/VRAM 은 같은 exe 의 전체 그룹으로 합산한다.
        pids = self._llama_parent_pids(sorted(set(self._find_pids(r"llama-server"))))
        seen_ports: set = set()
        for pid in pids:
            port = self._port_of_pid(pid) or int(self.llama_port())
            if port in seen_ports:
                continue
            seen_ports.add(port)
            service_id = f"llama-{port}"
            if service_id in excluded:
                continue
            previous = self._previous(service_id)
            group = self._llama_group_pids(pid)
            observed = sorted({uuid for one in group for uuid in pid_to_gpus.get(one, [])})
            # §7 도메인: 소속 GPU 는 NVML 이 아니라 main_server 가 이 포트를 시작할 때 지정한
            # GPU 목록으로 정한다. 지정값이 없을 때만 NVML 실측을 참고한다.
            launched = [d for d in (self.llama_devices(port) or []) if str(d).strip()]
            members = list(dict.fromkeys(u for u in (self._resolve_uuid(d) for d in launched) if u))
            if not members:
                members = observed
            multi = len(members) > 1
            home = members[0] if len(members) == 1 else ""
            state = ServiceState(
                id=service_id, type="llama", label=f"llama.cpp :{port}",
                port=port, backend=f"http://127.0.0.1:{port}", pid=pid,
                gpu_uuids=observed,          # 표시/VRAM 합산용 실측 (판정 근거는 home_gpu 다)
            )
            state.home_gpu = home
            state.multi_gpu = multi
            # GPU 에 걸쳐 도는 llama 는 VRAM 공유 요구 시나리오가 아니다 → eviction 관리에서
            # 제외한다. victim 으로도, 요청자로도 Arbiter 를 태우지 않는다 (admit 은 passthrough).
            # GPU0 요청이 멀티 GPU llama 를 내려서 GPU1 의 VRAM 까지 건드리는 것을 이것이 막는다.
            state.managed_evict = bool(home) and not multi
            state.domain_note = (
                "멀티 GPU 인스턴스 — eviction 관리 제외 (victim·requester 모두 우회)" if multi else
                "" if home else "소속 GPU 미확정 — 시작 시 지정한 GPU 를 확인하세요")
            state.gpu_pids = sorted(group)
            state.last_used_at = previous.last_used_at if previous else self._started_at or time.time()
            state.active_requests = previous.active_requests if previous else 0
            state.model = (previous.model if previous else "") or self._llama_model_name(port)
            state.expected_vram_mb, state.estimate_source = self._expected_for(
            service_id, "llama", state.model, state.home_gpu)
            self._probe_llama(state)
            # VRAM 은 라우터 + 모델 자식 전 PID 합산 (모델만 내려가면 자식이 사라져 0 에 가까워진다)
            state.current_vram_mb = int(sum(
                process_map.get(uuid, {}).get(one, 0)
                for uuid in state.gpu_uuids for one in state.gpu_pids)) if pid else 0
            if not pid:
                state.state = UNLOADED
            elif state.active_requests > 0 or state.busy_reported:
                state.state = ACTIVE
                state.last_used_at = time.time()
            elif state.router_capable and not state.model_loaded_reported:
                state.state = UNLOADED
            elif state.current_vram_mb > 64:
                state.state = RESIDENT
            else:
                state.state = UNLOADED
            # 게이트를 거치지 않은 외부 추론도 NVML util 로 보수 보호 (§11).
            # 판정은 소속 GPU 기준이다 — GPU1 이 바쁘다고 GPU0 도메인 서비스가 ACTIVE 가 되면 안 된다.
            if state.state == RESIDENT and state.home_gpu:
                home_pool = self.pools.get(state.home_gpu)
                if (home_pool.util or 0) >= int(self.setting("active_util_percent", 5)):
                    state.state = ACTIVE
            # 라우터 모드에서 모델이 내려가면 GPU 를持有한 자식 프로세스가 없다.
            # 라우터 자체는 살아 있으므로 재시작 대상이 아니며, 상태만 UNLOADED 로 본다.
            if state.router_capable and not state.model_loaded_reported:
                state.current_vram_mb = 0
                state.state = UNLOADED
            out.append(state)
        return out

    def _probe_llama(self, state: ServiceState) -> None:
        """라우터 모드 여부와 모델 상태를 읽는다. GET 한 번으로 모델을 로드시키면 안 된다.

        b10665 의 router proxy_get 은 model 쿼리가 없으면 400 을 반환하고, model 쿼리를
        붙이면 is_autoload() → ensure_model_ready() 로 그 요청만으로 모델이 로드된다
        (server-models.cpp:1900-1911). 따라서 reconcile 루프가 /slots?model=… 나
        /props?model=… 을 호출하는 것은 곧 autoload 를 유발하므로 쓰지 않는다.
        ACTIVE 판정은 자체 게이트 카운터 + NVML util 로만 내린다(§5).

        안전한 GET: /models(status 만 반환), /props(model 파라미터 없음 → router 더미 응답),
                    /health, 단일 모델 서버의 /slots.
        """
        state.busy_reported = False
        state.router_models = []
        payload = _http_json("GET", f"{state.backend}/models", timeout=4.0)
        data = payload.get("data") if isinstance(payload, dict) else None
        data = data if isinstance(data, list) else []
        binary = self._binary_of_pid(state.pid) if state.pid else ""
        features = llama_binary_features(binary) if binary else {"router": False, "default_model": False}
        state.native_default_model = bool(features.get("default_model"))

        known = ("loaded", "unloaded", "loading", "sleeping", "downloading", "downloaded")
        statuses = [_model_status(item) for item in data if isinstance(item, dict)]
        # router 모드 판정: /models 가 status 필드를 함께 준다(단일 모델 서버엔 없다).
        # 기준 바이너리 b10685: status 는 문자열이 아니라 {"value": "loaded", "args": [...]} 객체이고
        # unload 된 모델도 listing 에 남은 채 status 값만 "unloaded" 로 바뀐다
        # (server-models.cpp get_router_models → get_all_meta). 그래서 "목록이 비면 내려간 것" 이
        # 아니라 항목별 status 값으로 판단해야 한다. status 파싱을 빼먹으면 router 판정이 항상
        # False 가 되어 /models/unload 경로가 통째로 죽는다 (handoff 전부 blocked).
        state.router_capable = bool(features.get("router")) and any(s in known for s in statuses)
        state.autoload_enabled = bool(self.setting("llama_router_autoload", True))

        loaded_ids: List[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "")
            status = _model_status(item) or "loaded"
            if not model_id:
                continue
            state.router_models.append({"id": model_id, "status": status})
            if status in ("loaded", "sleeping", "loading"):
                loaded_ids.append(model_id)
        # 단일 모델 서버는 목록이 항상 채워져 있으므로 True 로 본다.
        state.model_loaded_reported = bool(loaded_ids) or not state.router_capable
        if not state.model and loaded_ids:
            state.model = str(loaded_ids[0])

        if state.router_capable:
            # /props 는 model 파라미터 없이도 안전하고, router 면 role=router 를 준다.
            props = _http_json("GET", f"{state.backend}/props", timeout=3.0) or {}
            if str(props.get("role") or "") == "router":
                state.autoload_enabled = bool(props.get("models_autoload", state.autoload_enabled))
            state.default_model = str(self.setting("llama_router_model_name") or "").strip()
        else:
            # 단일 모델 서버에서만 /slots 가 autoload 위험 없이 쓸 수 있다(model 파라미터 불필요).
            slots = _http_json("GET", f"{state.backend}/slots", timeout=4.0)
            state.slots_present = isinstance(slots, list)
            if state.slots_present:
                state.busy_reported = any(bool(item.get("is_processing")) for item in slots
                                          if isinstance(item, dict))
        mode = str(self.setting("llama_unload_mode", "protect"))
        state.unload_supported = bool(state.router_capable) or mode == "restart"

    # ---------------- 게이트 (§10 / §21)

    def _deficit(self, pool: GpuPool, expected_mb: int) -> int:
        """부족량 (MB) — 소속 GPU **한 장**의 수치로만 계산한다 (§7).

            deficit = expected + safety_margin(pool) - free(pool)

        GPU 간 free 합산, 용량 비례 배분, cross-GPU victim 최적화는 쓰지 않는다. GPU0 요청이
        GPU1 의 free 를 참조하는 순간 도메인 독립성이 깨지므로 이 함수는 항상 단일 pool 을 받는다.
        """
        return max(0, int(expected_mb) + int(pool.safety_margin_mb) - int(pool.free_mb))

    @staticmethod
    def _request_percent(pool: GpuPool, expected_mb: int) -> float:
        """이번 요청의 크기가 이 GPU 총 VRAM 의 몇 % 인지 (threshold 비교 단위).

        용량을 읽을 수 없으면 -1.0 을 돌려 "판정 불가" 를 명시한다 (0.0 으로 속여
        threshold 게이트가 닫히는 것을 막는다).
        """
        total = int(pool.total_mb or 0)
        if total <= 0:
            return -1.0
        return int(expected_mb or 0) * 100.0 / total

    def _evict_gate(self, pool: GpuPool, expected_mb: int, *, force_open: bool = False) -> Tuple[bool, str]:
        """GPU 별 evict threshold (%) — **요청 크기 기준** 게이트.

        threshold 는 GPU 사용률이 아니라 “이번 managed request 의 예상 VRAM 이 이 GPU
        총 VRAM 의 몇 % 인지” 와 비교한다. 예) 24GB GPU + threshold 50 → 예상 12GB 이상만
        큰 요청으로 본다.

            free 충분          → 이 게이트 전에 이미 공존 통과 (threshold 무관, UNKNOWN 도 포함)
            free 부족 + 통과  → 같은 home_gpu 의 RESIDENT managed peer eviction 허용
            free 부족 + 차단  → resident 를 건드리지 않는다. 이번 요청은 OOM 날 수 있다 (사용자 선택)
            threshold = 0     → 부족하면 항상 eviction (전통 동작)
            force_open        → 예상치를 알 수 없는 요청(UNKNOWN) 은 크기를 가늠할 수 없으므로
                                “큰 요청” 으로 간주해 게이트를 연다. 단 free 가 충분한 상황에서는
                                이 게이트에 아예 오지 않는다 (§12 에서 이미 통과).
        """
        if force_open:
            return True, (f"예상 VRAM 미학습 (UNKNOWN) → 큰 요청으로 간주, threshold {int(pool.evict_threshold_percent or 0)}% 적용 생략")
        threshold = int(pool.evict_threshold_percent or 0)
        if threshold <= 0:
            return True, ""
        percent = self._request_percent(pool, expected_mb)
        if percent < 0:
            return True, f"GPU{pool.index} 용량 미확인 → threshold 적용 불가, eviction 허용"
        if percent >= threshold:
            return True, (f"요청 {percent:.1f}% >= threshold {threshold}% "
                          f"(expected {int(expected_mb)}MB / total {pool.total_mb}MB)")
        return False, (f"요청 {percent:.1f}% < threshold {threshold}% "
                       f"(expected {int(expected_mb)}MB / total {pool.total_mb}MB) → resident 유지")

    @staticmethod
    def _passthrough_reason(service: ServiceState, pool: Optional[GpuPool]) -> str:
        if service.multi_gpu:
            return "multi-gpu-instance-excluded"
        if pool is None:
            return "no-gpu-domain"
        return "unsupported-service"


    def admit(self, service_id: str, expected_mb: Optional[int] = None,
              timeout_sec: Optional[float] = None, reason: str = "",
              hold: bool = False) -> Dict[str, Any]:
        """요청 허용 판단 (§21).

        hold=True 이면 판정/대기 동안 이 서비스에 홀드를 걸어, 블로킹 대기 중 reconcile 이
        이를 RESIDENT 로 격하하는 것을 막는다 (ComfyUI on_prompt 훅처럼 자기 /queue 를
        응답할 수 없는 상황 보호).
        """
        if not self.pools:
            self.ensure_pools()
        service = self.services.get(service_id) or self._lazy_service(service_id)
        # §7 도메인: 판정·eviction·lock 은 소속 GPU 한 장 안에서만 일어난다.
        # GPU0 요청은 GPU1 의 free / margin / lock / victim 을 단 한 번도 읽지 않는다.
        home = service.home_gpu
        pool = self.pools.get(home) if home else None
        if not service.managed or not service.managed_evict or pool is None:
            # 미지원 서비스 / 소속 GPU 미확정 / 멀티 GPU llama (관리 제외) → 기존 방식 그대로
            return {"allowed": True, "action": "passthrough", "service": service_id,
                    "gpu_uuids": list(service.gpu_uuids), "home_gpu": home, "evicted": [],
                    "reason": self._passthrough_reason(service, pool)}
        if not pool.enabled:
            # OFF GPU → 우회 (§10). 다른 GPU 의 ON/OFF 는 이 판단에 관여하지 않는다.
            return {"allowed": True, "action": "passthrough", "service": service_id,
                    "gpu_uuids": [home], "home_gpu": home, "evicted": [],
                    "reason": f"gpu{pool.index}-arbiter-off"}
        if service.state in (RESIDENT, ACTIVE):
            # 이미 resident → reload 없이 재사용 (§9/§12)
            with self._pool_lock([home]):
                service.active_requests += 1
                service.last_used_at = time.time()
                if service.state == RESIDENT:
                    service.state = ACTIVE
                return {"allowed": True, "action": "reuse", "service": service_id,
                        "gpu_uuids": [home], "home_gpu": home, "evicted": [],
                        "free_mb": {home: pool.free_mb}}
        if expected_mb is None:
            expected_mb, estimate_source = self._expected_for(
                service.id, service.type, service.model, home)
        else:
            estimate_source = "explicit"
        service.expected_vram_mb = int(expected_mb)
        # §15 UNKNOWN 정책 (확정):
        #   free 충분   → UNKNOWN 이어도 eviction 없음 (§12 에서 통과). “추정 불가” 라는 이유만으로
        #                  상주를 미리 내리지 않는다.
        #   free 부족   → 예상치를 알 수 없으므로 큰 요청으로 간주 → threshold 충족 처리 →
        #                  같은 home_gpu 의 RESIDENT managed peer 를 내린다.
        unknown = estimate_source == "unknown" and bool(self.setting("unknown_clear_residents", True))
        expected_mb = int(expected_mb)
        request_percent = self._request_percent(pool, expected_mb)
        base = {"service": service_id, "gpu_uuids": [home], "home_gpu": home,
                "home_index": pool.index, "expected_vram_mb": int(expected_mb),
                "estimate_source": estimate_source,
                "request_percent": round(request_percent, 1) if request_percent >= 0 else None,
                "evict_threshold_percent": int(pool.evict_threshold_percent or 0)}

        deadline = time.monotonic() + float(timeout_sec or self.setting("admission_timeout_sec", 900))
        evicted_total: List[str] = []
        attempt = 0
        if hold:
            self._take_hold(service, max(60.0, float(timeout_sec or self.setting("admission_timeout_sec", 900))))
        try:
            while True:
                attempt += 1
                with self._pool_lock([home]):
                    self._refresh_locked([home])
                    # ① §12 여유 확인 — 소속 GPU 한 장만 본다. 충분하면 threshold 도 UNKNOWN 도
                    #    상관없이 그냥 공존 통과한다 (불필요한 eviction 금지).
                    if self._deficit(pool, expected_mb) <= 0:
                        service.active_requests += 1
                        service.state = ACTIVE
                        service.last_used_at = time.time()
                        return {**base, "allowed": True,
                                "action": "load" if expected_mb else "run",
                                "evicted": evicted_total,
                                "deficit_mb": {home: 0}, "home_deficit_mb": 0,
                                "free_mb": {home: pool.free_mb}, "home_free_mb": pool.free_mb}

                    # ② §13 threshold 게이트 — 이번 요청의 크기가 GPU 총 VRAM 의 몇 % 인가.
                    #    미만이면 resident 를 건드리지 않고 그대로 통과시킨다 (이번 요청은 OOM 가능).
                    gate_open, gate_reason = self._evict_gate(pool, expected_mb, force_open=unknown)
                    if not gate_open:
                        service.active_requests += 1
                        service.state = ACTIVE
                        service.last_used_at = time.time()
                        self._log_event("evict_skipped", uuid=home, service=service_id,
                                        detail=gate_reason)
                        remaining_deficit = self._deficit(pool, expected_mb)
                        return {**base, "allowed": True, "action": "load-no-evict",
                                "evicted": evicted_total,
                                "deficit_mb": {home: remaining_deficit},
                                "home_deficit_mb": remaining_deficit,
                                "free_mb": {home: pool.free_mb}, "home_free_mb": pool.free_mb,
                                "note": gate_reason}

                    # ③ eviction: 같은 home_gpu + RESIDENT + managed_evict + 요청자 제외, LRU 순.
                    #    UNKNOWN 은 요구량을 가늠할 수 없으므로 소속 GPU 의 resident 를 모두 정리한다.
                    evicted_total += self._evict_locked(home, expected_mb, service, clear_all=unknown)
                    self._refresh_locked([home])
                    deficit = self._deficit(pool, expected_mb)
                    if deficit <= 0:
                        service.active_requests += 1
                        service.state = ACTIVE
                        service.last_used_at = time.time()
                        return {**base, "allowed": True, "action": "evicted+load",
                                "evicted": evicted_total,
                                "deficit_mb": {home: 0}, "home_deficit_mb": 0,
                                "free_mb": {home: pool.free_mb}, "home_free_mb": pool.free_mb}

                    blockers = self._active_blockers(home, service)
                    if blockers and attempt <= 2 and time.monotonic() < deadline:
                        # §11 ACTIVE 보호: 대기열(FIFO)에서 상태 변화까지 기다린다
                        pool.waiting_requests += 1
                        wait_for = next((self.services[s] for s in blockers if s in self.services), None)
                        self._log_event("wait", uuid=home, service=service_id,
                                        detail=f"{(wait_for.id if wait_for else 'ACTIVE')} 작업 완료 대기")
                        try:
                            remaining = max(1.0, deadline - time.monotonic())
                            pool.cond.wait(timeout=min(remaining, 20.0))
                        finally:
                            pool.waiting_requests = max(0, pool.waiting_requests - 1)
                        continue

                    deficit = self._deficit(pool, expected_mb)
                    raise VramNotAvailable(
                        f"VRAM 부족: GPU{pool.index} 도메인에서 {service.id}에 약 "
                        f"{int(expected_mb) + pool.safety_margin_mb}MB 필요 — "
                        f"GPU{pool.index} 부족 {deficit}MB(free {pool.free_mb}, "
                        f"margin {pool.safety_margin_mb}, 요청 {request_percent:.1f}% / "
                        f"threshold {pool.evict_threshold_percent}%) "
                        f"(eviction 후보 소진, ACTIVE={', '.join(blockers) or '없음'})",
                        gpu_uuids=[home], free_mb=pool.free_mb,
                        required_mb=int(expected_mb) + pool.safety_margin_mb,
                        blocked_by=blockers, deficits={home: deficit},
                    )
        finally:
            if hold:
                self._release_hold(service)
                # 통과 여부와 무관하게 Comfy 는 이 훅 이후 모델을 올릴 수 있다. queue_running 에
                # 잡히기 전까지의 공백(수 초) 동안 victim 이 되지 않도록 짧게 홀드를 연장한다.
                grace = float(self.setting("post_admit_grace_sec", 30))
                service.hold_until = max(float(service.hold_until or 0.0), time.time() + grace)

    def hold_state(self) -> Dict[str, Dict[str, Any]]:
        """디버깅: 현재 홀드 중인 서비스 (훅 대기/진행 중)."""
        now = time.time()
        return {
            sid: {"holds": int(st.holds), "hold_until": round(float(st.hold_until), 3),
                  "remaining_sec": max(0, round(float(st.hold_until) - now, 1))}
            for sid, st in self.services.items()
            if int(getattr(st, "holds", 0) or 0) > 0
            or float(getattr(st, "hold_until", 0.0) or 0.0) > now
        }

    # ---------------- paired llama handoff (§7 실제 사용 시나리오)
    #
    #   Comfy pending 등록 → JH 노드가 같은 GPU 의 llama 사용 → llama 응답 완료
    #   → 노드 return 직전 handoff → 필요 시 paired llama unload → NVML 반환 확인 → 워크플로우 계속
    #
    # on_prompt 에서 바로 내리지 않는 이유 (b10685 기준): JHLlamaPrompt.create_prompt() 은 채팅
    # 전에 _resolve_model() 로 GET /v1/models 를 친다 (__init__.py:2762 → 2511). unload 된 모델도
    # status.value="unloaded" 로 목록에 남으므로 모델명 해석은 성공하고, 다음 POST
    # /v1/chat/completions 에서 라우터가 autoload 으로 모델을 다시 올린다. 그래서 on_prompt
    # eviction 은 헛수고이고, 그 경로에는 pending 이 없어 handoff 가 no-pending 으로 끝난 뒤에도
    # llama 는 RESIDENT 로 남아 다음 Comfy 모델 로드와 contention 한다.

    def _pending_ttl(self) -> float:
        try:
            return max(30.0, float(self.setting("pending_demand_ttl_sec", 900)))
        except (TypeError, ValueError):
            return 900.0

    def register_pending(self, service_id: str, llama_ports: Optional[List[int]] = None,
                         expected_mb: Optional[int] = None, llama_node_count: int = 1,
                         prompt_id: str = "") -> Dict[str, Any]:
        """on_prompt 게이트: 이 Comfy 인스턴스의 upcoming demand 를 등록만 한다 (unload 없음)."""
        service = self.services.get(service_id)
        home = service.home_gpu if service else ""
        pool = self.pools.get(home) if home else None
        if service is None or pool is None or not pool.enabled or not service.managed_evict:
            return {"registered": False, "service": service_id, "home_gpu": home,
                    "reason": "gpu-arbiter-off" if (pool and not pool.enabled) else "unmanaged"}
        if expected_mb is None:
            expected_mb, estimate_source = self._expected_for(service_id, service.type, "", home)
        else:
            estimate_source = "explicit"
        unknown = estimate_source == "unknown"
        deficit = self._deficit(pool, int(expected_mb))
        gate_open, gate_reason = self._evict_gate(pool, int(expected_mb), force_open=unknown)
        # handoff 는 “같은 도메인의 관리 대상 llama 가 있을 때” 만 의미가 있다.
        peers = self._paired_llamas(home, requester=service)
        now = time.time()
        ttl = self._pending_ttl()
        entry = {
            "registered_at": now,
            "expires_at": now + ttl,
            "home_gpu": home,
            "home_index": pool.index,
            "expected_mb": int(expected_mb),
            "estimate_source": estimate_source,
            "unknown": unknown,
            "deficit_mb_at_register": int(deficit),
            "request_percent": self._request_percent(pool, int(expected_mb)),
            "evict_threshold_percent": int(pool.evict_threshold_percent or 0),
            "planned_evict": bool(peers) and deficit > 0 and gate_open,
            "gate_reason": gate_reason,
            "llama_ports": sorted({int(p) for p in (llama_ports or []) if str(p).strip()}),
            "expected_handoffs": max(1, int(llama_node_count or 1)),
            "handoffs": 0,
            "prompt_id": str(prompt_id or ""),
        }
        with self._pending_lock:
            self.pending[service_id] = entry
        self._log_event("pending", uuid=home, service=service_id,
                        detail=(f"upcoming demand 등록 — expected {int(expected_mb)}MB"
                                f" ({estimate_source}), 부족 {int(deficit)}MB, "
                                f"handoff 대기 llama={[s.id for s in peers]}"))
        return {"registered": True, "service": service_id, "home_gpu": home,
                "home_index": pool.index, "planned_evict": entry["planned_evict"],
                "expected_handoffs": entry["expected_handoffs"],
                "paired_llama": [s.id for s in peers],
                "deficit_mb": {home: int(deficit)}, "home_deficit_mb": int(deficit),
                "free_mb": {home: pool.free_mb}, "home_free_mb": pool.free_mb,
                "estimate_source": estimate_source, "note": gate_reason}

    def _live_pending(self, service_id: str) -> Optional[Dict[str, Any]]:
        with self._pending_lock:
            entry = self.pending.get(service_id)
            if entry is None:
                return None
            if time.time() >= float(entry.get("expires_at") or 0):
                self.pending.pop(service_id, None)
                return None
            return entry

    def clear_pending(self, service_id: str, reason: str = "") -> None:
        with self._pending_lock:
            existed = self.pending.pop(service_id, None) is not None
        if existed:
            self._log_event("pending_cleared", service=service_id, detail=reason or "수요 해소")

    def _paired_llamas(self, home_uuid: str, requester: Optional[ServiceState] = None,
                       ports: Optional[List[int]] = None) -> List[ServiceState]:
        """같은 home_gpu 도메인의 관리 대상 llama. 호출자가 victim id 를 고르지 않는다 (§7)."""
        wanted = {int(p) for p in (ports or []) if str(p).strip()}
        out = [state for state in self.services.values()
               if state.type == "llama" and state.managed_evict and state.home_gpu == home_uuid
               and (requester is None or state is not requester)
               and str(state.id) not in {str(x) for x in (self.setting("exclude_services") or [])}]
        if wanted:
            # JH 노드가 지정한 포트가 이 도메인의 llama 와 다르면 아무것도 내리지 않는다.
            # (예: Comfy :8188(GPU0) 의 노드가 :8081(GPU1) 을 가리키는 경우 — GPU0 요청이
            #  GPU1 llama 를 내리는 순간 도메인 독립성이 깨진다.)
            return [state for state in out if state.port in wanted]
        return out

    def handoff(self, comfy_service_id: str, llama_ports: Optional[List[int]] = None,
                reason: str = "jh-handoff") -> Dict[str, Any]:   # reason="jh-abort" → 수요만 해소
        """JH 노드가 llama 응답을 모두 받고 output 을 반환하기 직전에 부르는 handoff.

        정책:
          pending 없음               → llama 단독 사용. RESIDENT 유지.
          pending + free 충분       → RESIDENT 유지.
          pending + 부족 + threshold 충족 → paired llama unload 후 NVML 반환 확인.
          아직 실행될 JH llama 노드가 남음 → 보류한다 (그 노드의 채팅 요청이 autoload 를 트리거).

        참고로 on_prompt 시점 eviction 이 왜 handoff 로 대체되는지 (b10685): unload 된 모델도
        /v1/models 에 status.value="unloaded" 로 남으므로 JH 의 model="auto" 해석은 성공하고
        POST 에서 라우터가 autoload 로 되살린다. → eviction 헛수고 + pending 없어 no-pending
        → llama RESIDENT 잔존 → Comfy 와 contention 재발.
        """
        service = self.services.get(comfy_service_id)
        home = service.home_gpu if service else ""
        pool = self.pools.get(home) if home else None
        base = {"service": comfy_service_id, "home_gpu": home,
                "home_index": pool.index if pool else None}
        if service is None or pool is None or not pool.enabled:
            return {**base, "action": "passthrough", "reason": "gpu-domain-unavailable"}
        entry = self._live_pending(comfy_service_id)
        if entry is None:
            # pending 없는 llama 호출 = llama 단독 사용 → 상주 유지 (§7 정책)
            return {**base, "action": "resident-kept", "reason": "no-pending-demand"}
        if str(reason or "").startswith("jh-abort"):
            # JH 노드가 llama 응답 없이 실패로 끝났다 → 이 워크플로우의 다음 모델 로드는 없다.
            # 수요만 해소하고 llama 는 내리지 않는다 (상주 유지가 안전).
            self.clear_pending(comfy_service_id, "JH 노드 실패 — 수요만 해소")
            return {**base, "action": "pending-cleared", "reason": "jh-node-failed"}
        with self._pending_lock:
            entry["handoffs"] = int(entry.get("handoffs", 0)) + 1
            handoffs = int(entry["handoffs"])
            expected_handoffs = int(entry.get("expected_handoffs", 1))
        if handoffs < expected_handoffs:
            return {**base, "action": "await-llama-nodes", "handoffs": handoffs,
                    "expected_handoffs": expected_handoffs,
                    "reason": "워크플로우에 실행될 JH llama 노드가 남아 있어 unload 보류"}

        expected_mb = int(entry.get("expected_mb") or 0)
        unknown = bool(entry.get("unknown"))
        with self._pool_lock([home]):
            self._refresh_locked([home])
            deficit = self._deficit(pool, expected_mb)
            if deficit <= 0:
                self.clear_pending(comfy_service_id, f"handoff 시점에 free 충분 ({pool.free_mb}MB)")
                return {**base, "action": "resident-kept", "reason": "free 충분",
                        "deficit_mb": {home: 0}, "home_deficit_mb": 0,
                        "free_mb": {home: pool.free_mb}, "home_free_mb": pool.free_mb}
            gate_open, gate_reason = self._evict_gate(pool, expected_mb, force_open=unknown)
            if not gate_open:
                self.clear_pending(comfy_service_id, gate_reason)
                return {**base, "action": "resident-kept", "reason": gate_reason,
                        "deficit_mb": {home: deficit}, "home_deficit_mb": deficit,
                        "free_mb": {home: pool.free_mb}, "home_free_mb": pool.free_mb}
            peers = self._paired_llamas(home, requester=service, ports=llama_ports)
            if not peers:
                self.clear_pending(comfy_service_id, "같은 도메인에 관리 대상 llama 없음")
                return {**base, "action": "no-paired-llama", "deficit_mb": {home: deficit},
                        "home_deficit_mb": deficit, "free_mb": {home: pool.free_mb},
                        "home_free_mb": pool.free_mb}
            # 상태를 최신으로 다시 읽는다 — JH 노드가 응답을 받은 직후라는 사실만으로 캐시된
            # ACTIVE 를 믿으면 handoff 가 영구히 보류된다 (reconcile 주기는 5 초, 노드 재 실행은
            # 수 ms). 반대로 진행 중 요청(active_requests/hold) 은 절대 내리지 않는다.
            for peer in peers:
                try:
                    self._probe_llama(peer)
                except Exception:
                    pass
            victims = [state for state in peers if self._handoff_victim_ok(state)]
            if not victims:
                blockers = ", ".join(f"{s.id}({s.state}, req={s.active_requests}, "
                                     f"unload_supported={s.unload_supported})" for s in peers)
                self._log_event("handoff_blocked", uuid=home, service=comfy_service_id,
                                detail=f"paired llama 를 내릴 수 없음 — {blockers}")
                return {**base, "action": "blocked", "reason": f"paired llama unload 불가: {blockers}",
                        "deficit_mb": {home: deficit}, "home_deficit_mb": deficit,
                        "free_mb": {home: pool.free_mb}, "home_free_mb": pool.free_mb}
            victims.sort(key=lambda item: item.last_used_at)          # LRU
            freed_total = 0
            unloaded: List[str] = []
            for victim in victims:
                freed = self._unload(victim, home)
                unloaded.append(victim.id)
                freed_total += int(freed or 0)
                self._refresh_locked([home])
                if self._deficit(pool, expected_mb) <= 0:
                    break
            remaining = self._deficit(pool, expected_mb)
        self.clear_pending(comfy_service_id,
                           f"handoff 완료 — {', '.join(unloaded)} 반환 {freed_total}MB")
        return {**base, "action": "unloaded" if freed_total > 0 else "unload-failed",
                "unloaded": unloaded, "freed_mb": freed_total,
                "deficit_mb": {home: remaining}, "home_deficit_mb": remaining,
                "free_mb": {home: pool.free_mb}, "home_free_mb": pool.free_mb,
                "expected_vram_mb": expected_mb, "handoffs": handoffs}

    def _handoff_victim_ok(self, peer: ServiceState) -> bool:
        """handoff 로 내려도 되는 paired llama 인지 (재 probe 직후 기준으로만 판단)."""
        if int(getattr(peer, "active_requests", 0) or 0) > 0:
            return False                                  # 우리 게이트가 세는 진행 중 요청
        if int(getattr(peer, "holds", 0) or 0) > 0 or \
                float(getattr(peer, "hold_until", 0.0) or 0.0) > time.time():
            return False                                  # 다른 훅이 잡아둔 상태
        if not bool(getattr(peer, "unload_supported", True)):
            return False                                  # 단일 모델 + protect → 내릴 수단 없음
        if bool(getattr(peer, "router_capable", False)) and not bool(peer.model_loaded_reported):
            return False                                  # 이미 내려간 상태
        return True

    def pending_state(self) -> Dict[str, Any]:
        now = time.time()
        with self._pending_lock:
            live = {sid: dict(entry) for sid, entry in self.pending.items()
                    if time.time() < float(entry.get("expires_at") or 0)}
        return {sid: {"home_index": entry.get("home_index"),
                      "expected_mb": entry.get("expected_mb"),
                      "estimate_source": entry.get("estimate_source"),
                      "planned_evict": entry.get("planned_evict"),
                      "handoffs": entry.get("handoffs"),
                      "expected_handoffs": entry.get("expected_handoffs"),
                      "remaining_sec": max(0, round(float(entry.get("expires_at") or 0) - now, 1))}
                for sid, entry in live.items()}

    def release(self, service_id: str, observed_used_delta_mb: Optional[int] = None) -> None:
        """작업 종료. unload 하지 않고 ACTIVE→RESIDENT, last_used_at 갱신 (§22)."""
        service = self.services.get(service_id)
        if not service:
            return
        service.active_requests = max(0, service.active_requests - 1)
        service.last_used_at = time.time()
        if service.active_requests == 0 and service.state in (ACTIVE, LOADING):
            service.state = RESIDENT if service.current_vram_mb > 64 else UNLOADED
        if observed_used_delta_mb and 0 < int(observed_used_delta_mb) <= 8 * 1024 * 1024:
            self._learn(service_id, int(observed_used_delta_mb))
        self._wake_all()

    def _evict_locked(self, home_uuid: str, expected_mb: int, requester: ServiceState,
                      clear_all: bool = False) -> List[str]:
        """LRU 순으로 RESIDENT 를 내려 소속 GPU 의 부족량을 채운다 (§13).

        후보는 **home_gpu 가 요청자와 같은 서비스만**이다. 예전에 쓰던
        `any(u in enabled_uuids for u in state.gpu_uuids)` 는 멀티 GPU 인스턴스를 양쪽 도메인
        모두의 victim 으로 만들 뿐 아니라, NVML 이 잠시 두 카드로 보여준 서비스를 교차로 소모하게
        한다 (§7). GPU0 요청이 GPU1 victim 을 고를 수 있는 경로를 원천적으로 차단한다.
        clear_all=True(UNKNOWN 예상치) 면 부족량과 무관하게 소속 GPU 의 resident 를 모두 정리한다.
        """
        evicted: List[str] = []
        pool = self.pools.get(home_uuid)
        if pool is None:
            return evicted
        remaining = self._deficit(pool, expected_mb)
        now = time.time()
        excluded = {str(x) for x in (self.setting("exclude_services") or [])}
        candidates = [
            state for state in self.services.values()
            if state is not requester and state.managed and state.managed_evict
            and state.home_gpu == home_uuid                       # ← 도메인 소속만 (교차 GPU 불가)
            and state.state == RESIDENT
            and str(state.id) not in excluded
            # in-process 게이트에서 블로킹 대기/진행 중(hold) 인 서비스는 내려선 안 된다.
            # ComfyUI 는 훅이 대기하는 동안 자기 /queue 를 응답할 수 없어 ACTIVE 로 안 잡힌다.
            # holds>0 : 지금 다른 요청이 판정/대기 중. hold_until>now : 통과 후 공백 보호 구간.
            and not (int(getattr(state, "holds", 0) or 0) > 0
                     or float(getattr(state, "hold_until", 0.0) or 0.0) > now)
        ]
        candidates.sort(key=lambda item: item.last_used_at)          # LRU (§12/§13)
        for candidate in candidates:
            if not clear_all and remaining <= 0:
                break
            freed_mb = self._unload(candidate, home_uuid)
            evicted.append(candidate.id)
            remaining = max(0, remaining - freed_mb)
        return evicted

    def _unload(self, service: ServiceState, home_uuid: str = "") -> int:
        """RESIDENT 모델을 내린다. 서버 프로세스 자체는 종료하지 않는다 (§15/§16/§29-15).

        반환: 해당 도메인 GPU 에서 반환된 MB (int). 실패해도 같은 타입을 돌려준다 — 호출부가
        값에 연산만 하고 .items() 하지 않는다 (한때 Dict 을 반환하다 실패 경로만 0 을 돌려
        AttributeError 가 났던 자리).
        """
        home = home_uuid or service.home_gpu or (service.gpu_uuids[0] if service.gpu_uuids else "")
        pool = self.pools.get(home)
        if pool is None:
            service.state, service.last_error = ERROR, "소속 GPU 를 할당할 수 없어 unload 할 수 없습니다"
            return 0
        before = pool.free_mb
        service.state = UNLOADING
        service.last_error = ""
        ok = False
        try:
            if service.type == "comfy":
                response = _http("POST", f"{service.backend}/free",
                                 json={"unload_models": True, "free_memory": True}, timeout=30.0)
                ok = response is not None and response.status_code < 300
            else:
                if service.router_capable:
                    # 라우터: 라우터 프로세스는 유지하고 모델 인스턴스만 내린다.
                    # b10665 에서 unload 은 GPU 를持有한 자식 llama-server 를 graceful stop
                    # (timeout 후 force-kill) 하고 VRAM 을 반환한다. autoload 가 켜져 있으면
                    # 다음 LLM 요청에서 라우터가 같은 모델을 자동으로 다시 올린다 (§3).
                    names = self._loaded_model_names(service.backend)
                    if not names:
                        service.state = UNLOADED
                        service.last_error = ""
                        return 0   # 이미 내려간 상태
                    for name in names:
                        response = _http("POST", f"{service.backend}/models/unload",
                                         json={"model": name}, timeout=60.0)
                        ok = response is not None and response.status_code < 300
                        if not ok:
                            break
                elif str(self.setting("llama_unload_mode", "protect")) == "restart" and self.llama_restart:
                    print(f"[arbiter] {service.id}: router 미지원 → restart 모드로 llama-server 재시작")
                    ok = bool(self.llama_restart())
                else:
                    service.state = ERROR
                    service.last_error = ("llama.cpp가 단일 모델 모드라 unload 엔드포인트(/models/unload)가 "
                                          "없습니다. 라우터 모드(--models-preset) 실행 또는 'restart' 모드 필요")
                    self._log_event("unload_unsupported", uuid=home, service=service.id,
                                    detail=service.last_error)
                    return 0
        except Exception as error:
            service.state, service.last_error = ERROR, f"unload 요청 실패: {error}"
            return 0

        # unload 후 실제 NVML free 증가 확인 (§16/§27) — 소속 GPU 한 장만 관찰한다.
        timeout = float(self.setting("unload_timeout_sec", 60))
        deadline = time.monotonic() + timeout
        snapshot: Dict[str, Any] = {}
        freed = 0
        while True:
            snapshot = read_gpu_snapshot()
            now_free = int((snapshot.get(home) or {}).get("free_mb") or 0)
            freed = max(0, now_free - int(before))
            if freed > 64:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        if freed <= 64:
            service.state = ERROR
            service.last_error = f"unload 요청 후 {int(timeout)}초 동안 VRAM 반환이 관찰되지 않았습니다"
            self._log_event("unload_failed", uuid=home, service=service.id, detail=service.last_error)
            return 0
        service.state = UNLOADED
        service.current_vram_mb = 0
        service.last_used_at = time.time()
        if home in snapshot:
            pool.apply(snapshot[home])
        self._log_event("evict", uuid=home, service=service.id,
                        detail=f"GPU{pool.index} {freed} MB 반환 (도메인 {home_uuid or service.home_gpu})")
        return freed

    def _take_hold(self, service: ServiceState, seconds: float) -> None:
        service.holds = int(service.holds) + 1
        service.hold_until = max(float(service.hold_until or 0.0), time.time() + float(seconds))

    def _release_hold(self, service: ServiceState) -> None:
        service.holds = max(0, int(service.holds) - 1)
        if service.holds == 0:
            service.hold_until = 0.0

    def _active_blockers(self, home_uuid: str, requester: ServiceState) -> List[str]:
        """같은 도메인의 ACTIVE 만 대기 대상으로 본다 (다른 GPU 의 ACTIVE 는 무관)."""
        return [state.id for state in self.services.values()
                if state is not requester and state.state == ACTIVE
                and state.home_gpu == home_uuid]

    # ---------------- 예상 VRAM (§15)

    def _expected_for(self, service_id: str, type_name: str, model: str = "",
                      home_uuid: str = "") -> Tuple[int, str]:
        """예상 VRAM 자동 산정 (§15).

        사용자가 인스턴스별로 값을 넣어야만 동작하는 구조를 제거했다.
          1) config 오버라이드 — 디버깅/특수 환경용 선택 항목
          2) learned — reconcile 이 매 주기 기록하는 실측 peak
          3) llama 는 GGUF(+mmproj) 파일크기 × 1.15 (KV/버퍼 여유)
          4) UNKNOWN — 비율 추정 대신 "resident 를 먼저 비우는" 보수 경로 (§15).
             추정 불가 요청을 0MB 로 봐 eviction 이 영원히 발생하지 않던 결함을 대체하고,
             대형 워크플로우에서 과소추정될 위험도 없앤다. source == "unknown" 으로 노출된다.
        """
        overrides = self.setting("expected_vram_mb") or {}
        for key in (service_id, f"{service_id}:{model}" if model else "", type_name):
            if key and key in overrides:
                try:
                    return int(overrides[key]), "config"
                except (TypeError, ValueError):
                    pass
        learned = self.setting("learned_peak_mb") or {}
        if service_id in learned:
            try:
                learned_mb = int(learned[service_id])
                if learned_mb > 0:
                    return learned_mb, "learned"
            except (TypeError, ValueError):
                pass
        if type_name == "llama":
            file_mb = self._llama_file_size_mb(home_uuid)
            if file_mb > 0:
                return file_mb, "file"
        # UNKNOWN: 비율 추정치는 쓰지 않는다 (대형 Comfy 워크플로우에서 과소추정 위험).
        # 여기 값은 " victim 을 모두 내린 뒤 남는 요구량" 용도로만 쓴다 — admit 쪽이
        # unknown_clear_residents 로 resident 를 먼저 정리하고, 그 다음 이 하한으로
        # 부족량을 계산한다. 그래서 일부러 크게 잡지 않는다 (§15).
        return int(self.setting("unknown_expected_min_mb", 4096)), "unknown"

    def _llama_file_size_mb(self, home_uuid: str = "") -> int:
        """last_run.json 의 모델(+mmproj) 파일크기를 VRAM 추정치로 쓴다 (×1.15).

        GGUF 는 멀티샤드(-00001-of-00003.gguf) 가 일반적이고 첫 샤드가 수 MB 짜리인
        경우가 많다(이 머신의 Qwen3.8 IQ4_XS: 0.01 + 46.4 + 40.8 GB). 경로 한 개만
        세면 추정이 11MB 로 나와 admission 이 항상 통과해버리므로 같은 샤드를 모두 더한다.
        """
        try:
            with open(os.path.join(BASE_DIR, "last_run.json"), "r", encoding="utf-8") as handle:
                launch = json.load(handle)
            total = 0
            for key in ("model", "mmproj"):
                path = str(launch.get(key) or "")
                if not path or not os.path.isfile(path):
                    continue
                match = re.search(r"-(\d+)-of-(\d+)\.gguf$", path, flags=re.I)
                if match:
                    width = len(match.group(1))
                    count = int(match.group(2))
                    base = path[:match.start()]
                    directory = os.path.dirname(path)
                    for index in range(1, count + 1):
                        for candidate in (f"-{index:0{width}d}-of-{count:0{width}d}.gguf",
                                          f"-{index}-of-{count}.gguf"):
                            shard = base + candidate
                            if os.path.isfile(shard):
                                total += os.path.getsize(shard)
                                break
                else:
                    total += os.path.getsize(path)
            if not total:
                return 0
            cache_key = f"{launch.get('model') or ''}|{home_uuid or 'any'}"
            cached = self._file_size_cache.get(cache_key)
            if cached is not None:
                return cached
            # 단위 주의: total 은 bytes, capacity 는 MB 이다. 반드시 bytes → MB 로 바꾼 뒤
            # 상한을 적용해야 한다 (혼합하면 추정치가 수 KB 로 쪼그라든다).
            value = int(total / 1048576 * 1.15)
            # 파일크기는 VRAM 요구량의 **상한**으로만 쓴다(일부는 CPU/RAM 오프로드, KV 는 별도).
            # 용량 상한은 **소속 GPU 한 장**으로만 재단한다 — GPU0 + GPU1 합산(144GB) 을 기준으로
            # 삼으면 60GB 짜리 요청이 “두 카드 합치면 들어가” 가 되어 도메인 독립성이 깨진다 (§7).
            if home_uuid in self.pools:
                capacity = int(self.pools[home_uuid].total_mb or 0)
            else:
                capacity = 0
            if capacity:
                value = min(value, int(capacity * 0.95))
            self._file_size_cache[cache_key] = value
            return value
        except Exception:
            return 0

    _file_size_cache: Dict[str, int] = {}

    def _learn(self, service_id: str, used_mb: int) -> None:
        """실측 peak 학습 (§15).

        단봉(spiky) 값이 상한으로 굳으면 이후 공존이 영구적으로 막힌다. 그래서 최근 값을
        max 가 아니라 완만하게 반영한다 (70% 유지 + 30% 반영) — 감소에도 대칭이라
        워크플로우가 가벼워지면 다시 공존을 시도할 수 있다.
        """
        used_mb = int(used_mb)
        updated = False
        with self._settings_lock:
            table = self.settings.setdefault("learned_peak_mb", {})
            current = int(table.get(service_id, 0) or 0)
            if current <= 0:
                candidate = used_mb
            else:
                candidate = max(used_mb, int(current * 0.7 + used_mb * 0.3))
            if abs(candidate - current) >= 256:
                table[service_id] = candidate
                updated = True
        if updated:
            try:
                self.save_settings()
            except Exception:
                pass

    # ---------------- 락 유틸 (§19)

    class _PoolLock:
        """UUID 정렬 순서로 GPU lock을 획득해 교차 데드락을 막는다. inference 중에는 잡지 않는다."""

        def __init__(self, arbiter: "VramArbiter", uuids: List[str]):
            self.arbiter = arbiter
            self.uuids = sorted(set(uuids))
            self.acquired: List[GpuPool] = []

        def __enter__(self):
            for uuid in self.uuids:
                pool = self.arbiter.pools.get(uuid)
                if pool is None:
                    continue
                if not pool.lock.acquire(timeout=180):
                    raise RuntimeError(f"GPU lock 획득 실패: {uuid}")
                self.acquired.append(pool)
            return self

        def __exit__(self, *_exc):
            for pool in reversed(self.acquired):
                try:
                    with pool.cond:
                        pool.cond.notify_all()
                except RuntimeError:
                    pass
                pool.lock.release()
            self.acquired = []
            return False

    def _pool_lock(self, uuids: List[str]) -> "VramArbiter._PoolLock":
        return self._PoolLock(self, uuids)

    def _refresh_locked(self, uuids: List[str]) -> None:
        snapshot = read_gpu_snapshot()
        for uuid in uuids:
            pool = self.pools.get(uuid)
            if pool and uuid in snapshot:
                pool.apply(snapshot[uuid])

    def _wake_all(self) -> None:
        for pool in list(self.pools.values()):
            try:
                with pool.cond:
                    pool.cond.notify_all()
            except RuntimeError:  # lock를 이미 가진 스레드에서 호출된 경우
                continue

    # ---------------- 헬퍼

    def _lazy_service(self, service_id: str) -> ServiceState:
        type_name = "comfy" if service_id.startswith("comfy") else "llama"
        state = ServiceState(id=service_id, type=type_name, label=service_id)
        state.expected_vram_mb, state.estimate_source = self._expected_for(service_id, type_name)
        self.services[service_id] = state
        return state

    def _pid_for(self, service_name: str) -> Optional[int]:
        """서비스의 실제 PID. pidfile 값은 재활용된 PID일 수 있어 cmdline으로 검증한다."""
        pattern = r"llama-server" if service_name == "llama" else r"main\.py"
        pidfile = os.path.join(BASE_DIR, "logs", f"{service_name}.pid")
        try:
            with open(pidfile) as handle:
                pid = int(handle.read().strip())
            if self._pid_matches(pid, pattern):
                return pid
        except Exception:
            pass
        pids = self._find_pids(pattern)
        return pids[0] if pids else None

    def _comfy_pid(self, instance: str) -> Optional[int]:
        """어떤 PID가 어느 ComfyUI 인스턴스의 것인지 가린다.

        메인 인스턴스는 --user-directory 표식이 없고 보조는 user-<key>를 쓰므로,
        패턴 하나로 잡으면 두 인스턴스가 서로 뒤섞입니다. 포트도 함께 봅니다.
        """
        try:
            import psutil
        except Exception:
            return self._pid_for("comfyui")
        try:
            port = int(self.comfy_port(instance))
        except Exception:
            port = 0
        want_marker = None if instance == "main" else f"user-{instance}"
        fallback: Optional[int] = None
        for pid in self._find_pids(r"main\.py"):
            try:
                cmdline = " ".join(psutil.Process(pid).cmdline() or [])
            except Exception:
                continue
            if want_marker is not None:
                owner = want_marker in cmdline
            else:
                # 메인: 다른 인스턴스 표식이 없고, 명시적 --port가 이 인스턴스 포트거나 없음
                owner = not re.search(r"--user-directory[= ]+\S*user-[a-z0-9_]+", cmdline)
                port_match = re.search(r"--port[= ]+(\d+)", cmdline)
                if port_match and port and int(port_match.group(1)) != port:
                    owner = False
            if port and re.search(rf"--port[= ]+{port}\b", cmdline):
                return pid
            if owner and fallback is None:
                fallback = pid
        return fallback

    @staticmethod
    def _pid_matches(pid: int, pattern: str) -> bool:
        try:
            import psutil
            if not psutil.pid_exists(pid):
                return False
            cmdline = " ".join(psutil.Process(pid).cmdline() or [])
            return bool(re.search(pattern, cmdline))
        except Exception:
            return False

    @staticmethod
    def _find_pids(pattern: str) -> List[int]:
        try:
            from process_mgr import find_process
            return find_process(pattern)
        except Exception:
            return []

    @staticmethod
    def _port_of_pid(pid: int) -> Optional[int]:
        try:
            import psutil
            cmdline = " ".join(psutil.Process(pid).cmdline() or [])
            match = re.search(r"--port[= ]+(\d+)", cmdline)
            return int(match.group(1)) if match else None
        except Exception:
            return None

    @staticmethod
    def _binary_of_pid(pid: int) -> str:
        try:
            import psutil
            return str(psutil.Process(int(pid)).exe() or "")
        except Exception:
            return ""

    @staticmethod
    def _llama_processes() -> Dict[int, Any]:
        try:
            import psutil
        except Exception:
            return {}
        found: Dict[int, Any] = {}
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = str(proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            except Exception:
                continue
            if "llama-server" in name or "llama-server" in cmdline or "llama serve" in cmdline:
                found[int(proc.info["pid"])] = proc
        return found

    @classmethod
    def _llama_parent_pids(cls, pids: List[int]) -> List[int]:
        """llama-server 목록에서 라우터 부모만 남긴다.

        라우터는 자식(모델 인스턴스)을 띄우므로, 둘 다 관리 대상으로 삼으면 같은 서버를
        두 번 추적하고 VRAM 을 가진 쪽은 자식이라 GPU 매핑이 어긋난다.
        같은 llama-server 집합 안의 자식인 PID 를 제외하는 방식으로 가린다.
        """
        procs = cls._llama_processes()
        if not procs:
            return pids
        child_pids: set = set()
        for proc in procs.values():
            try:
                for child in proc.children(recursive=False):
                    if child.pid in procs:
                        child_pids.add(int(child.pid))
            except Exception:
                continue
        parents = [pid for pid in pids if int(pid) not in child_pids]
        return sorted(parents) or sorted(pids)

    @classmethod
    def _llama_group_pids(cls, parent_pid: int) -> set:
        """라우터 부모와 그 아래 모델 자식 PID 집합 (VRAM 합산용)."""
        group = {int(parent_pid)}
        procs = cls._llama_processes()
        proc = procs.get(int(parent_pid))
        if proc is None:
            return group
        try:
            for child in proc.children(recursive=True):
                if child.pid in procs:
                    group.add(int(child.pid))
        except Exception:
            pass
        return group

    @staticmethod
    def _llama_model_name(port: int) -> str:
        payload = _http_json("GET", f"http://127.0.0.1:{port}/props", timeout=3.0)
        if isinstance(payload, dict):
            path = str(payload.get("model_path") or payload.get("model_alias") or "")
            if path:
                return os.path.basename(path.rstrip("/"))
        return ""

    @staticmethod
    def _loaded_model_names(backend: str) -> List[str]:
        payload = _http_json("GET", f"{backend}/models", timeout=5.0)
        if not isinstance(payload, dict):
            return []
        names = []
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            status = _model_status(item) or "loaded"
            # b10685 의 POST /models/unload 는 "model is not running" 에서 400 을 낸다
            # (server-models.cpp post_router_models_unload). 이미 내려간 이름을 넘기지 않는다.
            if status in ("loaded", "sleeping", "loading") and item.get("id"):
                names.append(str(item["id"]))
        return names

    # ---------------- 이벤트 로그 (§27 디버깅)

    def _load_events(self) -> None:
        try:
            if os.path.exists(ARBITER_EVENTS_FILE):
                with open(ARBITER_EVENTS_FILE, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, list):
                    self.events = data[-200:]
        except Exception:
            self.events = []

    def _log_event(self, kind: str, *, uuid: str = "", service: str = "", detail: str = "") -> None:
        event = {"id": f"{int(time.time() * 1000)}-{kind}", "at": time.time(), "kind": kind,
                 "gpu_uuid": uuid, "service": service, "detail": detail}
        with self._events_lock:
            self.events.append(event)
            self.events = self.events[-200:]
            snapshot = list(self.events)
        try:
            with open(ARBITER_EVENTS_FILE, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False, indent=1)
        except OSError:
            pass
        print(f"[arbiter] {kind}: {detail} ({service or uuid})")

    # ---------------- 상태 API (§25)

    def summary(self) -> Dict[str, Any]:
        """NVML을 다시 읽지 않는 캐시 스냅샷. /api/gpus 응답에 얹어 카드 갱신과 맞춘다."""
        with self._events_lock:
            events = list(self.events)[-15:]
        gpus: Dict[str, Any] = {}
        for uuid, pool in self.pools.items():
            entry: Dict[str, Any] = {
                "index": pool.index, "name": pool.name, "arbiter_enabled": pool.enabled,
                "free_vram_mb": pool.free_mb, "used_vram_mb": pool.used_mb,
                "total_vram_mb": pool.total_mb, "safety_margin_mb": pool.safety_margin_mb,
                # 이번 요청의 예상 VRAM 이 이 GPU 총 용량의 몇 % 이상일 때만 resident eviction.
                "evict_threshold_percent": int(pool.evict_threshold_percent or 0),
                "unmanaged_vram_mb": pool.unmanaged_mb, "waiting_requests": pool.waiting_requests,
            }
            # 도메인 registry — 이 카드 소속 서비스만 담는다 (GPU 의 실측 점유 카드가 아니다).
            entry["managed_services"] = self.managed_services(uuid)
            if pool.enabled:
                entry["services"] = [self._service_payload(state) for state in self.services.values()
                                     if state.home_gpu == uuid]
            gpus[uuid] = entry
        return {
            "enabled": bool(self.setting("enabled")),
            "llama_unload_mode": self.setting("llama_unload_mode"),
            "admission_timeout_sec": self.setting("admission_timeout_sec"),
            "router_enabled": bool(self.setting("llama_router_enabled", True)),
            "pending": self.pending_state(),
            "reconciled_at": round(self._last_reconcile, 3),
            "uptime_sec": round(time.time() - self._started_at) if self._started_at else 0,
            "gpus": gpus,
            "events": events,
        }

    def status(self) -> Dict[str, Any]:
        self.ensure_pools()
        with self._events_lock:
            events = list(self.events)[-50:]
        gpu_list = []
        for pool in sorted(self.pools.values(), key=lambda item: (item.index is None, item.index)):
            entry = {
                "index": pool.index, "name": pool.name, "uuid": pool.uuid,
                "pci_bus_id": pool.pci_bus_id, "arbiter_enabled": pool.enabled,
                "total_vram_mb": pool.total_mb, "used_vram_mb": pool.used_mb,
                "free_vram_mb": pool.free_mb, "safety_margin_mb": pool.safety_margin_mb,
                "unmanaged_vram_mb": pool.unmanaged_mb, "util": pool.util,
                "waiting_requests": pool.waiting_requests,
                "evict_threshold_percent": int(pool.evict_threshold_percent or 0),
                "observed_at": round(pool.observed_at, 3),
            }
            entry["managed_services"] = self.managed_services(pool.uuid)
            if pool.enabled:
                entry["services"] = [self._service_payload(state) for state in self.services.values()
                                     if state.home_gpu == pool.uuid]
            gpu_list.append(entry)
        return {
            "enabled": bool(self.setting("enabled")),
            "llama_unload_mode": self.setting("llama_unload_mode"),
            "admission_timeout_sec": self.setting("admission_timeout_sec"),
            "unload_timeout_sec": self.setting("unload_timeout_sec"),
            "active_util_percent": self.setting("active_util_percent"),
            "exclude_services": self.setting("exclude_services"),
            "expected_vram_mb": self.setting("expected_vram_mb"),
            "learned_peak_mb": self.setting("learned_peak_mb"),
            "unknown_clear_residents": bool(self.setting("unknown_clear_residents", True)),
            "paired_handoff_enabled": bool(self.setting("paired_handoff_enabled", True)),
            "pending_demand_ttl_sec": self._pending_ttl(),
            "pending": self.pending_state(),
            "reconciled_at": round(self._last_reconcile, 3),
            "uptime_sec": round(time.time() - self._started_at) if self._started_at else 0,
            "holds": self.hold_state(),
            "gpus": gpu_list,
            "services": [self._service_payload(state, with_gpu=True) for state in
                         sorted(self.services.values(), key=lambda item: item.id)],
            "events": events,
        }

    def _service_payload(self, state: ServiceState, with_gpu: bool = False) -> Dict[str, Any]:
        payload = {
            "id": state.id, "type": state.type, "label": state.label or state.id,
            "state": state.state, "state_label": STATE_LABELS.get(state.state, state.state),
            "model": state.model, "port": state.port, "pid": state.pid,
            "active_requests": state.active_requests,
            "expected_vram_mb": state.expected_vram_mb,
            "estimate_source": state.estimate_source,
            "current_vram_mb": state.current_vram_mb,
            "last_used_at": round(state.last_used_at, 3),
            "idle_seconds": int(time.time() - state.last_used_at) if state.last_used_at else None,
            "router_capable": state.router_capable,
            "unload_supported": state.unload_supported,
            "last_error": state.last_error,
            "router_models": list(state.router_models),
            "router_default_model": state.default_model,
            "native_default_model": state.native_default_model,
            "autoload_enabled": state.autoload_enabled,
            "gpu_pids": list(state.gpu_pids),
        }
        # §7 도메인 정보 — UI 가 “이 서비스의 소속 카드” 와 “실측 점유 카드” 를 구별할 수 있게.
        home_pool = self.pools.get(state.home_gpu) if state.home_gpu else None
        payload["home_gpu"] = state.home_gpu
        payload["home_index"] = home_pool.index if home_pool is not None else None
        payload["multi_gpu"] = bool(state.multi_gpu)
        payload["managed_evict"] = bool(state.managed_evict)
        payload["domain_note"] = state.domain_note
        if with_gpu:
            payload["gpu_uuids"] = list(state.gpu_uuids)
        return payload


# --------------------------------------------------------------------- HTTP 헬퍼

def _model_status(item: Dict[str, Any]) -> str:
    """llama.cpp /models 항목의 status 를 문자열로 정규화한다.

    단일 모델 서버는 status 필드가 없고, router 는 버전마다 형태가 다르다:
      * 구버전 listing: "status": "loaded"
      * b10677/b10685 : "status": {"value": "loaded", "args": [...], ...}
    문자열로만 읽으면 b10685 에서 라우터 판정과 unload 대상 선별이 조용히 죽는다.
    """
    status = item.get("status")
    if isinstance(status, dict):
        return str(status.get("value") or status.get("state") or "").strip().lower()
    return str(status or "").strip().lower()


def _http(method: str, url: str, **kwargs):
    timeout = float(kwargs.pop("timeout", 10.0))
    try:
        with httpx.Client(timeout=timeout) as client:
            return client.request(method, url, **kwargs)
    except Exception as error:
        print(f"[arbiter] {method} {url} 실패: {error}")
        return None


def _http_json(method: str, url: str, **kwargs):
    response = _http(method, url, **kwargs)
    if response is None or response.status_code >= 400:
        return None
    try:
        return response.json()
    except Exception:
        return None


arbiter = VramArbiter()


# --------------------------------------------------------------------- REST API (§25)

@router.get("/status")
def get_status():
    return arbiter.status()


@router.post("/reconcile")
def post_reconcile():
    return {"ok": True, **arbiter.reconcile("manual")}


@router.post("/gpus/{uuid}/enable")
def set_gpu_enabled(uuid: str, payload: dict):
    return arbiter.set_gpu_enabled(uuid, bool((payload or {}).get("enabled")))


@router.post("/settings")
def set_settings(payload: dict):
    result = arbiter.update_settings(payload)
    arbiter.reconcile("config")
    return result


@router.post("/admit")
async def admit(payload: dict):
    """인프로세스 handshake용 게이트. 통과한 뒤 실제 추론을 시작하면 된다 (§18)."""
    payload = payload or {}
    service_id = str(payload.get("service") or "")
    if not service_id:
        raise HTTPException(400, "service 값이 필요합니다")
    expected = payload.get("expected_vram_mb")
    try:
        result = await asyncio.to_thread(
            arbiter.admit, service_id,
            int(expected) if expected not in (None, "") else None,
            float(payload["timeout_sec"]) if payload.get("timeout_sec") else None,
            str(payload.get("reason") or "api"),
        )
    except VramNotAvailable as error:
        return JSONResponse(
            {"allowed": False, "error": "VRAM_NOT_AVAILABLE", "detail": str(error),
             "free_vram_mb": error.free_mb, "required_vram_mb": error.required_mb,
             "blocked_by": error.blocked_by}, status_code=429)
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return result


@router.post("/release")
def release(payload: dict):
    service_id = str((payload or {}).get("service") or "")
    if not service_id:
        raise HTTPException(400, "service 값이 필요합니다")
    arbiter.release(service_id, (payload or {}).get("observed_used_delta_mb"))
    return {"ok": True}


@router.get("/router/status")
def router_status():
    """라우터 실행 준비 상태 (읽기 전용 — 어떤 GPU/서버에도 요청하지 않는다).

    바이너리 지원 matrix는 --help 검사 결과(캐시)이고, INI 는 현재 디스크에 있는 내용이다.
    """
    path = router_preset_path()
    content = ""
    exists = os.path.exists(path)
    try:
        if exists:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
    except OSError:
        pass
    binaries: Dict[str, Any] = {}
    for state in arbiter.services.values():
        if state.type != "llama":
            continue
        binary = arbiter._binary_of_pid(state.pid) if state.pid else ""
        features = llama_binary_features(binary) if binary else {"router": False, "default_model": False}
        binaries[binary or "unknown"] = {
            **features,
            "router_running": bool(state.router_capable),
            "autoload_enabled": bool(state.autoload_enabled),
            "loaded_models": [item.get("id") for item in state.router_models
                              if str(item.get("status")) in ("loaded", "sleeping")],
        }
    return {
        "enabled": bool(arbiter.setting("llama_router_enabled", True)),
        "autoload": bool(arbiter.setting("llama_router_autoload", True)),
        "model_name": arbiter.setting("llama_router_model_name"),
        "use_alias": bool(arbiter.setting("llama_router_use_alias", False)),
        "inject_missing_model": bool(arbiter.setting("llama_inject_missing_model", False)),
        "preset_path": path,
        "preset_exists": exists,
        "preset_ini": content,
        "native_default_model_available": any(
            bool(item.get("default_model")) for item in binaries.values()),
        "binaries": binaries,
    }


@router.post("/router/prepare")
def router_prepare(payload: dict = None):
    """last_run.json 의 모델을 라우터 preset INI 에 등록만 한다 (서버 재시작 없음).

    A안: 첫 버전은 모델 1개. section name 은 기존 클라이언트가 보내는 model 값에 맞춘다
    (미지정 시 GGUF 파일명에서 유도 — JH 노드처럼 data[0].id 를 읽는 클라이언트는 무관).
    INI 작성만 수행하고 llama-server 에는 아무 요청도 보내지 않는다.
    """
    payload = payload or {}
    model = str(payload.get("model") or "").strip()
    mmproj = str(payload.get("mmproj") or "").strip()
    if not model:
        last_run_path = os.path.join(BASE_DIR, "last_run.json")
        try:
            with open(last_run_path, "r", encoding="utf-8") as handle:
                model = str(json.load(handle).get("model") or "").strip()
        except Exception:
            model = ""
    if not model or not os.path.isfile(model):
        raise HTTPException(400, f"모델 파일이 없습니다: {model or '(last_run.json 에 모델 없음)'}")
    binary = str(payload.get("binary") or "")
    native = bool(llama_binary_features(binary).get("default_model")) if binary else False
    name = sanitize_section_name(payload.get("name") or arbiter.setting("llama_router_model_name")
                                 or router_section_name(model))
    entries = [{"name": name, "model": model,
                "mmproj": mmproj or str(payload.get("mmproj") or ""),
                "alias": str(arbiter.setting("llama_router_aliases") or ""),
                "default": True}]
    info = write_router_preset(entries, native_default_model=native)
    with arbiter._settings_lock:
        arbiter.settings["llama_router_model_name"] = name
    arbiter.save_settings()
    return {"ok": True, "section": name, **info}


def _comfy_service_id(port: int) -> str:
    def find_id():
        return next((sid for sid, st in arbiter.services.items()
                     if st.type == "comfy" and st.port == port), None)
    service_id = find_id()
    if service_id is None:
        arbiter.reconcile("comfy-hook")
        service_id = find_id()
    return service_id or ""


def _scan_jh_llama_nodes(json_data: Any) -> Tuple[int, List[int]]:
    """queue 에 들어갈 prompt 그래프에서 JH llama 노드 수/포트를 읽는다 (정적 스캔).

    handoff 가 “아직 실행될 JH llama 노드가 남았는지” 를 판단하는 근거다. 마지막 JH 노드의
    return 에서만 unload 해야 앞선 노드의 채팅 요청이 라우터 autoload 로 모델을 되살리지 않는다.
    실패해도 (워크플로우 형식이 달라도) handoff 는 home_gpu 도메인 전부로 안전 폴백한다.
    """
    if not isinstance(json_data, dict):
        return 0, []
    count, ports = 0, []
    for node in json_data.values():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type") or "") != "JHLlamaPrompt":
            continue
        count += 1
        raw = (node.get("inputs") or {}).get("server_url")
        match = re.search(r"://[^/:]+(?::(\d+))?", str(raw or ""))
        if match and match.group(1):
            try:
                ports.append(int(match.group(1)))
            except (TypeError, ValueError):
                pass
    return count, ports


@router.post("/comfy/admit")
async def comfy_admit(payload: dict):
    """ComfyUI on_prompt 훅 게이트 — **수요 등록** 지점 (§7 handoff).

    큐잉 직전에 불려 이 GPU 의 upcoming demand 를 등록한다. 같은 도메인의 상주 llama 를
    **지금 내리지 않는다** — workflow 안의 JH llama 노드가 곧 같은 llama 를 호출한다. b10685
    라우터는 unload 된 모델도 /v1/models 에 status.value="unloaded" 로 남기 때문에 JH 의
    model="auto" 해석은 성공하고 곧 이은 POST 에서 라우터가 autoload 으로 모델을 다시 올린다.
    따라서 on_prompt eviction 은 헛수고일 뿐 아니라 그 경로엔 pending 이 없어 handoff 가
    no-pending 으로 끝나고, llama 는 RESIDENT 로 돌아가 다음 Comfy 모델 로드와 contention 한다.
    실제 unload 은 JH 노드의 handoff 가 llama 응답 완료 직후에 수행한다.

    같은 GPU 에 llama 가 없는 워크플로우(또는 handoff 미탑재 환경)에서는 예전처럼 on_prompt 단계에서
    필요한 resident peer 를 바로 내린다 (fallback 워커).
    항상 200 으로 답하고, 판단 실패로 Comfy 워크플로우를 막지 않는다 (fail-open).
    """
    payload = payload or {}
    try:
        port = int(payload.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    service_id = _comfy_service_id(port)
    if not service_id:
        return {"allowed": True, "action": "unknown-service", "port": port}
    # 훅은 그래프 원문 대신 스캔 결과를 보내므로 jh_llama_nodes / llama_ports 를 1 차로 쓴다.
    # 이 값들을 무시하면 JH 워크플로우가 전부 즉시-eviction 경로로 빠져, eviction 은 autoload 로
    # 헛수고가 되고 handoff 는 no-pending 으로 끝나 contention 이 그대로 남는다.
    try:
        node_count = int(payload.get("jh_llama_nodes") or 0)
    except (TypeError, ValueError):
        node_count = 0
    llama_ports: List[int] = []
    raw_ports = payload.get("llama_ports")
    if isinstance(raw_ports, (list, tuple)):
        for one in raw_ports:
            try:
                llama_ports.append(int(one))
            except (TypeError, ValueError):
                pass
    if not node_count:
        # 그래프 원문을 함께 보내는 구버전 훅과의 호환 경로
        node_count, scan_ports = _scan_jh_llama_nodes(payload.get("prompt"))
        llama_ports = llama_ports or scan_ports
    if not bool(arbiter.setting("paired_handoff_enabled", True)) or node_count == 0:
        # llama 를 호출하지 않는 워크플로우 → 즉시 eviction 이 정답이다.
        timeout = payload.get("timeout_sec")
        try:
            result = await asyncio.to_thread(
                arbiter.admit, service_id, None,
                float(timeout) if timeout else None, "comfy-ui", True)
            return {**result, "allowed": True}
        except VramNotAvailable as error:
            return {"allowed": False, "action": "vram_not_available", "detail": str(error),
                    "free_vram_mb": error.free_mb, "required_vram_mb": error.required_mb,
                    "blocked_by": error.blocked_by, "deficit_mb": error.deficits}
        except Exception as error:
            return {"allowed": True, "action": "error", "detail": str(error)}
    try:
        result = await asyncio.to_thread(
            arbiter.register_pending, service_id, llama_ports, None, node_count,
            str(payload.get("prompt_id") or ""))
        return {**result, "allowed": True, "handoff_required": True, "jh_llama_nodes": node_count}
    except Exception as error:
        return {"allowed": True, "action": "error", "detail": str(error)}


@router.post("/comfy/handoff")
def comfy_handoff(payload: dict):
    """JH llama 노드가 llama 응답을 모두 받고 output 을 반환하기 직전에 부르는 handoff (§7).

    호출자는 Comfy 인스턴스 포트(= 자기 도메인) 만 알려준다. victim llama id 는 고르지 않는다
    — main_server 가 comfy :8188 → GPU0, llama :8080 → GPU0 를 이미 알고 있으므로 같은
    home_gpu 도메인의 paired resident 를 Arbiter 가 결정한다.
    """
    payload = payload or {}
    try:
        port = int(payload.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    service_id = _comfy_service_id(port)
    if not service_id:
        return {"action": "unknown-service", "port": port}
    ports = payload.get("llama_ports") or []
    if not isinstance(ports, (list, tuple)):
        ports = []
    try:
        return arbiter.handoff(service_id, [int(p) for p in ports if str(p).strip()],
                               str(payload.get("reason") or "jh-handoff"))
    except Exception as error:
        # handoff 실패로 Comfy 워크플로우를 죽이지 않는다.
        return {"action": "error", "detail": str(error)}


@router.post("/evict/{service_id}")
def evict(service_id: str):
    """디버깅용 강제 unload. ACTIVE는 거부한다 (§11)."""
    state = arbiter.services.get(service_id)
    if state is None:
        raise HTTPException(404, f"알 수 없는 서비스: {service_id}")
    if state.state == ACTIVE or state.active_requests > 0:
        raise HTTPException(409, "ACTIVE 작업은 강제 unload 할 수 없습니다")
    home = state.home_gpu
    pool = arbiter.pools.get(home) if home else None
    if pool is None:
        raise HTTPException(409, f"{service_id}는 소속 GPU 가 확정되지 않아 unload 할 수 없습니다")
    if not pool.enabled:
        raise HTTPException(409, f"GPU {pool.index}는 Arbiter OFF 상태입니다")
    with arbiter._pool_lock([home]):
        freed_mb = arbiter._unload(state, home)
    return {"ok": freed_mb > 0, "freed_mb": freed_mb, "home_gpu": home,
            "home_index": pool.index, "state": state.state, "error": state.last_error}


# --------------------------------------------------------------------- Request Gate 프록시 (§18)

def _clean_headers(headers: Dict[str, str]) -> Dict[str, str]:
    # x-arbiter-* 는 게이트가 추가하는 내부 헤더로 백엔드에 전달하지 않는다.
    drop = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
    return {key: value for key, value in headers.items()
            if key.lower() not in drop and not key.lower().startswith("x-arbiter")}


LLAMA_INFERENCE_RE = re.compile(
    r"^(v1/)?(chat/completions|completions|completion|embeddings?|tokenize|detokenize|rerank|messages)$")


def _inject_default_model(service_id: str, body: bytes) -> bytes:
    """A안 기본값은 OFF. 필요 시에만 게이트가 body 의 model 을 대신 채운다.

    기본 동작은 section name 을 클라이언트 값에 맞추는 것이므로 주입이 불필요하다.
    다만 바이너리가 default-model 미지원(PR #19855 미머지)인 상태에서
    model 필드를 아예 보내지 않는 클라이언트를 지원해야 할 때만 ON 으로 쓴다.
    """
    if not str(service_id).startswith("llama-"):
        return body
    if not bool(arbiter.setting("llama_inject_missing_model", False)):
        return body
    service = arbiter.services.get(service_id)
    if not service or not service.router_capable or service.native_default_model:
        return body
    default_model = str(service.default_model or "").strip()
    if not default_model or not body:
        return body
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return body                      # JSON 이 아니면 그대로 전달
    if not isinstance(parsed, dict) or parsed.get("model"):
        return body
    parsed["model"] = default_model
    print(f"[arbiter] {service_id}: model 필드 없음 → '{default_model}' 주입 (inject_missing_model=ON)")
    return json.dumps(parsed, ensure_ascii=False).encode("utf-8")


async def _gated_forward(service_id: str, url: str, headers: Dict[str, str], body: bytes,
                         *, allow_retry: bool) -> Any:
    """gate → forward → (OOM 시 reconcile + 1회 retry) (§21/§27). 무한 retry 하지 않는다."""
    body = _inject_default_model(service_id, body)
    try:
        gate = await asyncio.to_thread(arbiter.admit, service_id, None, None, "proxy")
    except VramNotAvailable as error:
        return JSONResponse(
            {"error": {"type": "vram_not_available", "message": str(error),
                       "free_vram_mb": error.free_mb, "required_vram_mb": error.required_mb,
                       "blocked_by": error.blocked_by}}, status_code=429)
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error

    # §7 도메인: 사용량 관측도 소속 GPU 한 장만 한다. GPU0 서비스 작업 중에 GPU1 이 다른
    # 워크로드로 비어가는 것이 delta 로 잡혀 학습 peak 가 오염되고 release 가 엉뚱한 값을 배운다.
    gate_home = str((gate or {}).get("home_gpu") or "")
    if not gate_home:
        gate_service = arbiter.services.get(service_id)
        gate_home = gate_service.home_gpu if gate_service else ""
    gate_pool = arbiter.pools.get(gate_home)
    baseline = int(gate_pool.free_mb) if gate_pool is not None else 0
    clean = _clean_headers(headers)
    attempts = 0
    while True:
        attempts += 1
        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
        try:
            upstream = await client.send(client.build_request("POST", url, headers=clean, content=body),
                                         stream=True)
        except Exception as error:
            await client.aclose()
            await asyncio.to_thread(arbiter.release, service_id)
            return JSONResponse({"error": {"type": "upstream_unreachable", "message": str(error)}},
                                status_code=502)

        if upstream.status_code >= 500 and allow_retry and attempts == 1:
            payload = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            text = payload.decode("utf-8", errors="replace")
            if "out of memory" in text.lower() or "cuda oom" in text.lower():
                # §27: reconcile + 추가 eviction 후 딱 1회만 retry. 무한 retry 금지.
                # NVML/백엔드 probe가 event loop를 막지 않도록 스레드로 넘긴다.
                await asyncio.to_thread(arbiter.reconcile, "oom")
                await asyncio.to_thread(arbiter.release, service_id)
                continue
            await asyncio.to_thread(arbiter.release, service_id)
            return JSONResponse({"error": {"type": "upstream_error", "message": text[:2000]}},
                                status_code=upstream.status_code)

        async def stream(upstream=upstream, client=client):
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
                home_pool = arbiter.pools.get(gate_home)
                delta = max(0, baseline - int(home_pool.free_mb)) if home_pool is not None else 0
                await asyncio.to_thread(arbiter.release, service_id, delta)

        response_headers = {key: value for key, value in upstream.headers.items()
                            if key.lower() not in ("content-length", "transfer-encoding", "connection")}
        # 게이트 결과(action/evicted)는 헤더를 변형하지 않고 로그로만 남긴다.
        # 클라이언트가 받는 응답 형상은 백엔드와 완전히 동일해야 기존 UI가 깨지지 않는다.
        if gate.get("evicted"):
            print(f"[arbiter] {service_id} 실행 전 eviction: {', '.join(gate['evicted'])}")
        return StreamingResponse(stream(), status_code=upstream.status_code, headers=response_headers)


@proxy_router.post("/comfy/{instance}/prompt")
async def proxy_comfy_prompt(instance: str, request: Request):
    body = await request.body()
    try:
        port = int(arbiter.comfy_port(instance))
    except Exception as error:
        raise HTTPException(404, f"알 수 없는 ComfyUI 인스턴스: {instance}") from error
    # ComfyUI /prompt는 즉시 JSON을 반환하므로 OOM retry 가능
    return await _gated_forward(f"comfy-{instance}", f"http://127.0.0.1:{port}/prompt",
                                dict(request.headers), body, allow_retry=True)


@proxy_router.api_route("/llama/{path:path}", methods=["POST"])
async def proxy_llama(path: str, request: Request):
    if not LLAMA_INFERENCE_RE.match(path):
        raise HTTPException(404, "inference 요청만 프록시합니다")
    body = await request.body()
    streaming = b'"stream"' in body and b"true" in body.lower()
    return await _gated_forward(f"llama-{int(arbiter.llama_port())}",
                                f"http://127.0.0.1:{arbiter.llama_port()}/{path}",
                                dict(request.headers), body, allow_retry=not streaming)
