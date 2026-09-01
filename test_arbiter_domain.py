"""Arbiter GPU 도메인 독립성 — 정적 검증 하네스.

NVML / 네트워크 / 실제 프로세스에 전혀 접근하지 않는다. read_gpu_snapshot,
read_process_map, _unload 을 스텁으로 대체하고, GPU lock 에 감시(spy) 를 물려
"다른 GPU 의 lock 이 실제로 잡히지 않는가" 까지 확인한다.

실행:  .venv/bin/python test_arbiter_domain.py
"""

import ast
import json
import os
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import vram_arbiter as va

U0 = "GPU-00000000-0000-0000-0000-000000000000"   # GPU0 · CMP 170HX 64GB
U1 = "GPU-11111111-1111-1111-1111-111111111111"   # GPU1 · RTX 3090  24GB

SNAP = {
    U0: dict(index=0, name="CMP 170HX", pci_bus_id="0000:01:00.0",
             total_mb=65536, used_mb=5536, free_mb=60000, util=0),
    U1: dict(index=1, name="RTX 3090", pci_bus_id="0000:02:00.0",
             total_mb=24576, used_mb=21576, free_mb=3000, util=0),
}
va.read_gpu_snapshot = lambda: {k: dict(v) for k, v in SNAP.items()}
va.read_process_map = lambda: {}

# 하네스가 실제 설정/이벤트 파일을 건드리지 않도록 전부 임시 경로로 돌린다.
import tempfile
SANDBOX = tempfile.mkdtemp(prefix="arbiter_domain_test_")
va.ARBITER_SETTINGS_FILE = os.path.join(SANDBOX, "settings.json")
va.ARBITER_EVENTS_FILE = os.path.join(SANDBOX, "events.json")

FAILURES = []
CHECKS = 0


def check(name, condition, detail=""):
    global CHECKS
    CHECKS += 1
    ok = bool(condition)
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


# ------------------------------------------------------------------ arbiter 조립
A = va.VramArbiter()
A.settings = json.loads(json.dumps(va.DEFAULT_SETTINGS))
A.settings.update(enabled=True, unknown_clear_residents=True, active_util_percent=5)
A.settings["gpus"] = {
    U0: {"enabled": True, "evict_threshold_percent": 0},    # GPU0 : 부족하면 항상 eviction
    U1: {"enabled": True, "evict_threshold_percent": 50},   # GPU1 : 예상 12288MB 이상만 큰 요청
}
A.settings["expected_vram_mb"] = {}
A.settings["learned_peak_mb"] = {}
A._started_at = time.time()
A._log_event = lambda *a, **k: None          # gpu_arbiter_events.json 부작용 차단
A.ensure_pools()

# GPU lock 스파이 — 어느 카드의 lock 이 실제로 잡혔는지 기록한다.
# _PoolLock 은 notify 를 위해 cond(같은 lock) 를 한 번 더 잡으므로 집합으로 비교한다.
ACQUIRED = set()


_POOL_ENTER = va.VramArbiter._PoolLock.__enter__


def _spy_pool_enter(self):
    """_PoolLock 이 실제로 어떤 GPU lock 을 획득했는지 기록한다.

    GpuPool.lock 을 래퍼로 감싸면 Condition 의 내부 프로토콜(_is_owned 등) 과 부딪혀
    테스트 하네스가 제품 코드를 왜곡한다. 그래서 락은 원래 RLock 그대로 두고,
    획득 결과(self.acquired) 만 훑어 어떤 카드를 잡았는지 확인한다.
    """
    result = _POOL_ENTER(self)
    for pool in self.acquired:
        ACQUIRED.add(pool.uuid)
    return result


va.VramArbiter._PoolLock.__enter__ = _spy_pool_enter

UNLOADED_CALLS = []


def fake_unload(service, home_uuid=""):
    """_unload 스텁 — 도메인 GPU 한 장에만 반환량을 반영한다."""
    home = home_uuid or service.home_gpu
    UNLOADED_CALLS.append((service.id, home))
    freed = int(service.current_vram_mb or 0)
    if freed:
        SNAP[home]["free_mb"] += freed
        SNAP[home]["used_mb"] = max(0, SNAP[home]["total_mb"] - SNAP[home]["free_mb"])
        service.current_vram_mb = 0
    service.state = va.UNLOADED
    A.pools[home].apply(SNAP[home])
    return freed


A._unload = fake_unload

FREE0, FREE1 = 60000, 3000


def make_service(sid, typ, home, state, vram=0, multi=False, managed=True, port=0):
    st = va.ServiceState(id=sid, type=typ, label=sid, port=port,
                         backend=f"http://127.0.0.1:{port}",
                         gpu_uuids=([home] if home else []) + ([U0, U1] if multi else []))
    st.home_gpu, st.state, st.current_vram_mb = home, state, vram
    st.multi_gpu, st.managed_evict = multi, managed
    st.expected_vram_mb = 1000
    return st


def world(comfy0=va.UNLOADED, llama0=va.UNLOADED, comfy1=va.UNLOADED, llama1=va.UNLOADED,
          free0=FREE0, free1=FREE1):
    """매 케이스 시작 시 세계를 초기화한다. 요청자는 항상 UNLOADED 여야 admit 이 전체 경로를 탄다."""
    SNAP[U0].update(free_mb=free0, used_mb=SNAP[U0]["total_mb"] - free0, util=0)
    SNAP[U1].update(free_mb=free1, used_mb=SNAP[U1]["total_mb"] - free1, util=0)
    A.pools[U0].apply(SNAP[U0])
    A.pools[U1].apply(SNAP[U1])
    A.services = {
        "comfy-main":  make_service("comfy-main",  "comfy", U0, comfy0,
                                    3000 if comfy0 == va.RESIDENT else 0, port=8188),
        "llama-8080":  make_service("llama-8080",  "llama", U0, llama0,
                                    40000 if llama0 == va.RESIDENT else 0, port=8080),
        "comfy-gpu1":  make_service("comfy-gpu1",  "comfy", U1, comfy1,
                                    2000 if comfy1 == va.RESIDENT else 0, port=8189),
        "llama-8081":  make_service("llama-8081",  "llama", U1, llama1,
                                    18000 if llama1 == va.RESIDENT else 0, port=8081),
        # GPU0+GPU1 에 걸쳐 도는 llama — 기동 정보로 확정된 멀티 GPU 인스턴스.
        "llama-cross": make_service("llama-cross", "llama", "", va.RESIDENT, 30000,
                                    multi=True, managed=False, port=8090),
    }
    A.rebuild_registry()
    ACQUIRED.clear()
    del UNLOADED_CALLS[:]


def admit_safe(sid, expected, reason):
    try:
        return A.admit(sid, expected, 2.0, reason)
    except va.VramNotAvailable as error:
        return {"action": "VRAM_NOT_AVAILABLE", "evicted": [], "detail": str(error)[:200]}


print("=== 0) GPU 별 managed service registry (home_gpu 기준) ===")
world(comfy0=va.UNLOADED, llama0=va.RESIDENT, comfy1=va.UNLOADED, llama1=va.RESIDENT)
reg0, reg1 = A.managed_services(U0), A.managed_services(U1)
check("gpu0_services = [comfy-main, llama-8080]", reg0 == ["comfy-main", "llama-8080"], f"{reg0}")
check("gpu1_services = [comfy-gpu1, llama-8081]", reg1 == ["comfy-gpu1", "llama-8081"], f"{reg1}")
check("멀티 GPU llama 는 양쪽 registry 모두 부재",
      "llama-cross" not in reg0 and "llama-cross" not in reg1, f"gpu0={reg0} gpu1={reg1}")

print("\n=== 1) GPU0 request -> GPU1 victim 0건 ===")
# GPU1 은 free 가 크게 부족하고 resident 가 상주해 있다. GPU0 입장에서 GPU1 을 건드릴
# 유인이 얼마든지 있어도, 구현은 GPU1 을 전혀 만지지 않아야 한다.
world(comfy0=va.UNLOADED, llama0=va.RESIDENT, comfy1=va.RESIDENT, llama1=va.RESIDENT,
      free0=1000, free1=3000)
r = admit_safe("comfy-main", 4000, "t1")
check("GPU0 req → GPU1 서비스 unload 0건",
      all(h != U1 for _, h in UNLOADED_CALLS), f"unloaded={UNLOADED_CALLS} action={r.get('action')}")
check("GPU0 req → evicted 는 GPU0 소속만",
      all(sid in ("comfy-main", "llama-8080") for sid in r.get("evicted", [])),
      f"evicted={r.get('evicted')}")
check("GPU0 req → GPU1 lock 미획득 (도메인 lock 독립)",
      ACQUIRED == {U0}, f"acquired={sorted(ACQUIRED)}")

print("\n=== 2) GPU1 request -> GPU0 victim 0건 ===")
world(comfy0=va.RESIDENT, llama0=va.RESIDENT, comfy1=va.UNLOADED, llama1=va.RESIDENT,
      free0=1000, free1=3000)
r = admit_safe("comfy-gpu1", 13000, "t2")   # 13000/24576 = 52.9% >= 50 → eviction 허용
check("GPU1 req → GPU0 서비스 unload 0건",
      all(h != U0 for _, h in UNLOADED_CALLS), f"unloaded={UNLOADED_CALLS} action={r.get('action')}")
check("GPU1 req → 같은 도메인 peer(llama-8081) 만 unload",
      UNLOADED_CALLS == [("llama-8081", U1)], f"unloaded={UNLOADED_CALLS}")
check("GPU1 req → GPU0 lock 미획득", ACQUIRED == {U1}, f"acquired={sorted(ACQUIRED)}")

print("\n=== 3) 멀티 GPU llama → victim · requester 양쪽 관리 제외 ===")
world(comfy0=va.UNLOADED, llama0=va.RESIDENT, comfy1=va.UNLOADED, llama1=va.RESIDENT,
      free0=1000, free1=3000)
r = admit_safe("llama-cross", 30000, "t3")
check("requester: passthrough", r.get("action") == "passthrough",
      f"action={r.get('action')} reason={r.get('reason')}")
check("requester: eviction 0건", r.get("evicted") == [] and UNLOADED_CALLS == [],
      f"evicted={r.get('evicted')} unloaded={UNLOADED_CALLS}")
check("requester: GPU0/GPU1 lock 모두 미획득", ACQUIRED == set(), f"acquired={sorted(ACQUIRED)}")

# victim 측 — 양쪽 도메인이 모두 부족에 빠져도 멀티 GPU llama 는 후보가 아니다.
world(comfy0=va.UNLOADED, llama0=va.RESIDENT, comfy1=va.RESIDENT, llama1=va.RESIDENT,
      free0=1000, free1=3000)
admit_safe("comfy-main", 60000, "t3b")
check("victim(GPU0 큰 요청): llama-cross 제외 + GPU1 도메인도 미침",
      "llama-cross" not in [i for i, _ in UNLOADED_CALLS]
      and all(h != U1 for _, h in UNLOADED_CALLS), f"unloaded={UNLOADED_CALLS}")
world(comfy0=va.RESIDENT, llama0=va.RESIDENT, comfy1=va.UNLOADED, llama1=va.RESIDENT,
      free0=1000, free1=3000)
admit_safe("comfy-gpu1", 13000, "t3c")
check("victim(GPU1 큰 요청): llama-cross 제외 + GPU0 도메인도 미침",
      "llama-cross" not in [i for i, _ in UNLOADED_CALLS]
      and all(h != U0 for _, h in UNLOADED_CALLS), f"unloaded={UNLOADED_CALLS}")

print("\n=== 4) threshold = 요청 크기 % (GPU1 total 24576, threshold 50 → 12288MB) ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free0=60000, free1=3000)
r = admit_safe("comfy-gpu1", 8000, "t4a")    # 32.6% < 50 → resident 유지
check("threshold 미만 → resident 유지 (eviction 0)",
      r.get("action") == "load-no-evict" and UNLOADED_CALLS == []
      and A.services["llama-8081"].state == va.RESIDENT,
      f"action={r.get('action')} pct={r.get('request_percent')} unloaded={UNLOADED_CALLS}")
check("threshold 미만 → 부족량이 응답에 노출 (OOM 가능성 전달)",
      int(r.get("home_deficit_mb", 0)) > 0,
      f"home_deficit_mb={r.get('home_deficit_mb')} note={r.get('note')}")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free0=60000, free1=3000)
r = admit_safe("comfy-gpu1", 13000, "t4b")   # 52.9% >= 50 → eviction
check("threshold 이상 → resident eviction",
      r.get("action") == "evicted+load" and UNLOADED_CALLS == [("llama-8081", U1)],
      f"action={r.get('action')} pct={r.get('request_percent')} unloaded={UNLOADED_CALLS}")
world(comfy0=va.UNLOADED, llama0=va.RESIDENT, free0=1000, free1=60000)
r = admit_safe("comfy-main", 12000, "t4c")   # GPU0 threshold=0 → 부족 시 항상
check("threshold=0 → 부족 시 항상 eviction",
      r.get("action") == "evicted+load" and UNLOADED_CALLS == [("llama-8080", U0)],
      f"action={r.get('action')} unloaded={UNLOADED_CALLS}")

print("\n=== 5) free 충분 → threshold 이상이어도 불필요한 eviction 없음 ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free0=60000, free1=20000)
r = admit_safe("comfy-gpu1", 13000, "t5")    # 52.9% >= 50 이지만 free 20GB 로 충분
check("free 충분 + threshold 이상 → eviction 0",
      r.get("action") in ("load", "reuse") and UNLOADED_CALLS == []
      and A.services["llama-8081"].state == va.RESIDENT,
      f"action={r.get('action')} pct={r.get('request_percent')} unloaded={UNLOADED_CALLS}")

print("\n=== 6) UNKNOWN 정책 (확정안) — expected 생략 호출로 source=unknown 재현 ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free0=60000, free1=20000)
r = A.admit("comfy-gpu1", None, 2.0, "t6a")
check("UNKNOWN 감지 (estimate_source=unknown)", r.get("estimate_source") == "unknown",
      f"source={r.get('estimate_source')} expected={r.get('expected_vram_mb')}")
check("UNKNOWN + free 충분 → 미리 내리지 않는다 (eviction 0)",
      r.get("action") in ("load", "run", "reuse") and UNLOADED_CALLS == []
      and A.services["llama-8081"].state == va.RESIDENT,
      f"action={r.get('action')} unloaded={UNLOADED_CALLS}")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free0=60000, free1=3000)
r = A.admit("comfy-gpu1", None, 2.0, "t6b")
pct = r.get("request_percent")
check("UNKNOWN + free 부족 → threshold(50%) 미만 크기여도 큰 요청으로 간주해 eviction",
      r.get("action") == "evicted+load" and UNLOADED_CALLS == [("llama-8081", U1)]
      and (pct is not None and pct < 50),
      f"action={r.get('action')} pct={pct} unloaded={UNLOADED_CALLS}")

print("\n=== 7) API 응답 형상 호환 (free_mb / deficit_mb 는 dict 유지) ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free0=60000, free1=3000)
r = admit_safe("comfy-gpu1", 13000, "t7")
check("free_mb 는 dict", isinstance(r.get("free_mb"), dict), f"{type(r.get('free_mb')).__name__}")
check("deficit_mb 는 dict", isinstance(r.get("deficit_mb"), dict), f"{type(r.get('deficit_mb')).__name__}")
check("home_free_mb / home_deficit_mb scalar 병존",
      isinstance(r.get("home_free_mb"), int) and isinstance(r.get("home_deficit_mb"), int),
      f"home_free_mb={r.get('home_free_mb')} home_deficit_mb={r.get('home_deficit_mb')}")

print("\n=== 8) _resolve_uuid (기동 정보 표식 통일) ===")
check("UUID 그대로", A._resolve_uuid(U0) == U0)
check("index '1'", A._resolve_uuid("1") == U1)
check("CUDA1 → GPU1", A._resolve_uuid("CUDA1") == U1)
check("cuda:0 → GPU0", A._resolve_uuid("cuda:0") == U0)
check("PCI busId", A._resolve_uuid("0000:01:00.0") == U0)
check("미지 토큰 → ''", A._resolve_uuid("CUDA9") == "" and A._resolve_uuid("") == "")

print("\n=== 9) C: app.py _arbiter_llama_devices (NVML 아닌 기동 정보 1차) ===")
app_src = open(os.path.join(BASE_DIR, "app.py"), encoding="utf-8").read()
tree = ast.parse(app_src)
func = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_arbiter_llama_devices":
        func = node
        break
check("_arbiter_llama_devices 가 app.py 에 존재", func is not None)
if func:
    last_run = json.load(open(os.path.join(BASE_DIR, "last_run.json"), encoding="utf-8"))
    declared = [str(x).strip() for x in (last_run.get("gpuDevices") or [])]

    class FakeSvc:
        device = ["GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"]   # 메모리 값은 일부러 다른 값

    ns = {
        "json": json, "open": open,
        "LAST_RUN_FILE": os.path.join(BASE_DIR, "last_run.json"),
        "services": {"llama": FakeSvc()},
        "_llama_port_value": lambda p: 8080,
        "load_llama_settings": lambda: {},
        "LLAMA_PORT": 8080,
    }
    exec(compile(ast.Module(body=[func], type_ignores=[]), "app.py", "exec"), ns)
    hook = ns["_arbiter_llama_devices"]
    got = hook(8080)
    check("last_run.json gpuDevices 를 1차로 반환", got == declared, f"got={got} declared={declared}")
    check("gpuDevices 2개 → 멀티 GPU 판정 가능", len(got) == 2, f"{got}")
    check("포트 불일치 → [] (남의 기동정보 참조 금지)", hook(9999) == [], f"{hook(9999)}")
    check("1차 우선: 메모리 device 가 아니라 디스크 gpuDevices", got != FakeSvc.device,
          f"disk={got} mem={FakeSvc.device}")
    # last_run.json 의 UUID 는 이 머신의 실물 UUID 라 A(가짜 풀) 로는 해석되지 않는다.
    # 실제 토큰이 그대로 pool UUID 로 확정되는지 확인하려면 같은 UUID 를 가진 풀이 필요하다.
    real = va.VramArbiter()
    real.pools = {uuid: va.GpuPool(uuid=uuid, index=idx, name=f"sim{idx}", total_mb=1000)
                  for idx, uuid in enumerate(declared)}
    resolved = [real._resolve_uuid(one) for one in got]
    check("훅 결과 → pool UUID 확정 (실제 last_run UUID 기준, 왜곡 없음)",
          resolved == declared, f"resolved={resolved}")

    # 훅 → 도메인 확정 시뮬레이션: 소속이 2장이면 multi_gpu → 관리 제외
    check("훅 결과 2장 → multi_gpu → eviction 관리 제외 (victim·requester 모두)",
          len([u for u in resolved if u]) == 2, f"members={resolved}")

print("\n=== 10) ACTIVE 대기도 도메인 한정 ===")
world(comfy0=va.UNLOADED, llama0=va.ACTIVE, comfy1=va.UNLOADED, llama1=va.ACTIVE)
check("_active_blockers(GPU0) → GPU0 ACTIVE 만",
      A._active_blockers(U0, A.services["comfy-main"]) == ["llama-8080"],
      f"{A._active_blockers(U0, A.services['comfy-main'])}")
check("_active_blockers(GPU1) → GPU1 ACTIVE 만",
      A._active_blockers(U1, A.services["comfy-gpu1"]) == ["llama-8081"],
      f"{A._active_blockers(U1, A.services['comfy-gpu1'])}")

print("\n" + "=" * 64)
print(f"총 {CHECKS}건 중 실패 {len(FAILURES)}건")
for name in FAILURES:
    print(f"  FAIL: {name}")
print("=" * 64)
sys.exit(1 if FAILURES else 0)
