"""pending / handoff / abort / 다중 JH 노드 + b10685 Router 파싱 — 정적 검증 하네스.

NVML / 네트워크 / 실제 프로세스에 전혀 접근하지 않는다. read_gpu_snapshot,
read_process_map, _unload, _probe_llama, _http_json 을 스텁으로 대체한다.

실행:  .venv/bin/python test_arbiter_handoff.py
"""

import json
import os
import sys
import tempfile
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
SANDBOX = tempfile.mkdtemp(prefix="arbiter_handoff_test_")
va.ARBITER_SETTINGS_FILE = os.path.join(SANDBOX, "handoff_settings.json")
va.ARBITER_EVENTS_FILE = os.path.join(SANDBOX, "handoff_events.json")

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
A.settings.update(enabled=True, unknown_clear_residents=True, active_util_percent=5,
                  paired_handoff_enabled=True, pending_demand_ttl_sec=900.0)
A.settings["gpus"] = {
    U0: {"enabled": True, "evict_threshold_percent": 0},    # GPU0 : 부족하면 항상 eviction
    U1: {"enabled": True, "evict_threshold_percent": 50},   # GPU1 : 예상 12288MB 이상만 큰 요청
}
A.settings["expected_vram_mb"] = {}
A.settings["learned_peak_mb"] = {}
A._started_at = time.time()
A._log_event = lambda *a, **k: None          # gpu_arbiter_events.json 부작용 차단
A.ensure_pools()

ACQUIRED = set()
_POOL_ENTER = va.VramArbiter._PoolLock.__enter__


def _spy_pool_enter(self):
    result = _POOL_ENTER(self)
    for pool in self.acquired:
        ACQUIRED.add(pool.uuid)
    return result


va.VramArbiter._PoolLock.__enter__ = _spy_pool_enter

UNLOADED_CALLS = []


def fake_unload(service, home_uuid=""):
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
_PROBE_ORIGINAL = va.VramArbiter._probe_llama
A._probe_llama = lambda state: None          # H1~H11: handoff 직전 재probe 를 no-op 으로


def make_service(sid, typ, home, state, vram=0, multi=False, managed=True, port=0,
                 unload_supported=True, router=True):
    st = va.ServiceState(id=sid, type=typ, label=sid, port=port,
                         backend=f"http://127.0.0.1:{port}",
                         gpu_uuids=([home] if home else []) + ([U0, U1] if multi else []))
    st.home_gpu, st.state, st.current_vram_mb = home, state, vram
    st.multi_gpu, st.managed_evict = multi, managed
    st.unload_supported, st.router_capable = unload_supported, router
    st.expected_vram_mb = 1000
    return st


def world(comfy0=va.UNLOADED, llama0=va.RESIDENT, comfy1=va.UNLOADED, llama1=va.RESIDENT,
          free0=60000, free1=3000, unload_ok0=True, unload_ok1=True):
    SNAP[U0].update(free_mb=free0, used_mb=SNAP[U0]["total_mb"] - free0, util=0)
    SNAP[U1].update(free_mb=free1, used_mb=SNAP[U1]["total_mb"] - free1, util=0)
    A.pools[U0].apply(SNAP[U0])
    A.pools[U1].apply(SNAP[U1])
    A.services = {
        "comfy-main":  make_service("comfy-main",  "comfy", U0, comfy0, 0, port=8188),
        "llama-8080":  make_service("llama-8080",  "llama", U0, llama0, 40000,
                                    port=8080, unload_supported=unload_ok0),
        "comfy-gpu1":  make_service("comfy-gpu1",  "comfy", U1, comfy1, 0, port=8189),
        "llama-8081":  make_service("llama-8081",  "llama", U1, llama1, 18000,
                                    port=8081, unload_supported=unload_ok1),
        "llama-cross": make_service("llama-cross", "llama", "", va.RESIDENT, 30000,
                                    multi=True, managed=False, port=8090),
    }
    A.rebuild_registry()
    with A._pending_lock:
        A.pending.clear()
    ACQUIRED.clear()
    del UNLOADED_CALLS[:]


def call_handoff(sid, ports=None, reason="jh-handoff"):
    return A.handoff(sid, ports, reason)


print("=== H1) pending 없음 + handoff → llama 단독 사용, RESIDENT 유지 ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000)
r = call_handoff("comfy-gpu1")
check("H1 resident-kept / no-pending-demand",
      r.get("action") == "resident-kept" and r.get("reason") == "no-pending-demand"
      and UNLOADED_CALLS == [] and A.services["llama-8081"].state == va.RESIDENT,
      f"action={r.get('action')} reason={r.get('reason')} unloaded={UNLOADED_CALLS}")

print("\n=== H2) pending 있음 + free 충분 → RESIDENT 유지 ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=20000)
reg = A.register_pending("comfy-gpu1", [8081], 13000, 1, "p-h2")
check("H2 register_pending 은 unload 하지 않는다",
      reg.get("registered") and UNLOADED_CALLS == []
      and A.services["llama-8081"].state == va.RESIDENT,
      f"registered={reg.get('registered')} unloaded={UNLOADED_CALLS}")
check("H2 planned_evict=False (free 충분)", reg.get("planned_evict") is False,
      f"planned_evict={reg.get('planned_evict')} deficit={reg.get('home_deficit_mb')}")
r = call_handoff("comfy-gpu1", [8081])
check("H2 handoff → resident-kept (free 충분)",
      r.get("action") == "resident-kept" and UNLOADED_CALLS == []
      and A.services["llama-8081"].state == va.RESIDENT,
      f"action={r.get('action')} reason={r.get('reason')} unloaded={UNLOADED_CALLS}")

print("\n=== H3) pending + 부족 + threshold 충족 → paired llama unload + 반환 확인 ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000)
reg = A.register_pending("comfy-gpu1", [8081], 13000, 1, "p-h3")
check("H3 planned_evict=True (52.9% >= 50%)", reg.get("planned_evict") is True,
      f"planned_evict={reg.get('planned_evict')} pct={reg.get('request_percent')}")
check("H3 pending 등록 상태 노출", "comfy-gpu1" in A.pending_state(), f"{A.pending_state()}")
r = call_handoff("comfy-gpu1", [8081])
check("H3 handoff → unloaded + 같은 도메인 llama 만",
      r.get("action") == "unloaded" and UNLOADED_CALLS == [("llama-8081", U1)],
      f"action={r.get('action')} unloaded={UNLOADED_CALLS}")
check("H3 NVML 반환 확인 (freed_mb > 0, free 갱신)",
      int(r.get("freed_mb") or 0) == 18000 and int(r.get("home_free_mb") or 0) == 21000
      and r.get("home_deficit_mb") == 0 and r.get("deficit_mb") == {U1: 0},
      f"freed={r.get('freed_mb')} free={r.get('home_free_mb')} deficit={r.get('home_deficit_mb')}")
check("H3 handoff 후 pending 해소", A.pending_state() == {}, f"{A.pending_state()}")

print("\n=== H4) pending + 부족 + threshold 미만 (KNOWN) → resident 유지 ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000)
reg = A.register_pending("comfy-gpu1", [8081], 8000, 1, "p-h4")
check("H4 planned_evict=False (32.6% < 50%)", reg.get("planned_evict") is False,
      f"pct={reg.get('request_percent')} planned={reg.get('planned_evict')}")
r = call_handoff("comfy-gpu1", [8081])
check("H4 handoff → resident-kept + 부족량 노출 (OOM 가능성)",
      r.get("action") == "resident-kept" and UNLOADED_CALLS == []
      and int(r.get("home_deficit_mb") or 0) > 0
      and A.services["llama-8081"].state == va.RESIDENT,
      f"action={r.get('action')} deficit={r.get('home_deficit_mb')} unloaded={UNLOADED_CALLS}")

print("\n=== H5) 다중 JH 노드 — 마지막 노드 전에는 unload 보류 ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000)
reg = A.register_pending("comfy-gpu1", [8081], 13000, 2, "p-h5")
check("H5 expected_handoffs=2", int(reg.get("expected_handoffs") or 0) == 2,
      f"expected_handoffs={reg.get('expected_handoffs')}")
r1 = call_handoff("comfy-gpu1", [8081])
check("H5 1번째 handoff → await-llama-nodes, unload 0",
      r1.get("action") == "await-llama-nodes" and UNLOADED_CALLS == []
      and A.services["llama-8081"].state == va.RESIDENT,
      f"action={r1.get('action')} {r1.get('handoffs')}/{r1.get('expected_handoffs')}")
r2 = call_handoff("comfy-gpu1", [8081])
check("H5 마지막 handoff → unloaded",
      r2.get("action") == "unloaded" and UNLOADED_CALLS == [("llama-8081", U1)],
      f"action={r2.get('action')} unloaded={UNLOADED_CALLS}")

print("\n=== H6) abort (JH 노드 실패) → 수요만 해소, llama 유지 ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000)
A.register_pending("comfy-gpu1", [8081], 13000, 1, "p-h6")
r = call_handoff("comfy-gpu1", [8081], reason="jh-abort:timeout")
check("H6 pending-cleared + unload 0건",
      r.get("action") == "pending-cleared" and UNLOADED_CALLS == []
      and A.services["llama-8081"].state == va.RESIDENT and A.pending_state() == {},
      f"action={r.get('action')} unloaded={UNLOADED_CALLS} pending={A.pending_state()}")

print("\n=== H7) GPU0 도메인 handoff → GPU1 victim/lock 미접촉 ===")
world(comfy0=va.UNLOADED, llama0=va.RESIDENT, comfy1=va.UNLOADED, llama1=va.RESIDENT,
      free0=1000, free1=3000)
A.register_pending("comfy-main", [8080], 12000, 1, "p-h7")
r = call_handoff("comfy-main", [8080])
check("H7 GPU0 handoff → GPU1 llama 미unload",
      all(h != U1 for _, h in UNLOADED_CALLS) and "llama-8081" not in [i for i, _ in UNLOADED_CALLS],
      f"unloaded={UNLOADED_CALLS}")
check("H7 GPU1 lock 미획득", ACQUIRED == {U0}, f"acquired={sorted(ACQUIRED)}")
check("H7 GPU1 상태 그대로 RESIDENT", A.services["llama-8081"].state == va.RESIDENT,
      f"llama-8081={A.services['llama-8081'].state}")

print("\n=== H8) multi-GPU llama → paired 후보에서 제외 ===")
world(comfy0=va.UNLOADED, llama0=va.UNLOADED, comfy1=va.UNLOADED, llama1=va.UNLOADED,
      free0=1000, free1=1000)
peers0 = [s.id for s in A._paired_llamas(U0, requester=A.services["comfy-main"])]
peers1 = [s.id for s in A._paired_llamas(U1, requester=A.services["comfy-gpu1"])]
check("H8 GPU0/GPU1 paired 후보 모두에 llama-cross 부재",
      "llama-cross" not in peers0 and "llama-cross" not in peers1, f"peers0={peers0} peers1={peers1}")
A.services["llama-cross"].state = va.RESIDENT
A.services["llama-cross"].current_vram_mb = 30000
A.register_pending("comfy-main", [], 60000, 1, "p-h8")
call_handoff("comfy-main", [])
check("H8 handoff 가 llama-cross 를 내리지 않는다",
      "llama-cross" not in [i for i, _ in UNLOADED_CALLS], f"unloaded={UNLOADED_CALLS}")

print("\n=== H9) JH 가 다른 GPU 의 llama 를 가리키면 아무것도 내리지 않는다 ===")
world(comfy0=va.UNLOADED, llama0=va.RESIDENT, comfy1=va.UNLOADED, llama1=va.RESIDENT,
      free0=1000, free1=3000)
A.register_pending("comfy-main", [8081], 12000, 1, "p-h9")
r = call_handoff("comfy-main", [8081])
check("H9 no-paired-llama (도메인 밖 포트) + unload 0건",
      r.get("action") == "no-paired-llama" and UNLOADED_CALLS == [],
      f"action={r.get('action')} unloaded={UNLOADED_CALLS}")
check("H9 GPU0/GPU1 llama 모두 유지",
      A.services["llama-8080"].state == va.RESIDENT and A.services["llama-8081"].state == va.RESIDENT,
      f"g0={A.services['llama-8080'].state} g1={A.services['llama-8081'].state}")

print("\n=== H10) pending TTL 만료 → resident-kept ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000)
A.register_pending("comfy-gpu1", [8081], 13000, 1, "p-h10")
with A._pending_lock:
    A.pending["comfy-gpu1"]["expires_at"] = time.time() - 1.0
r = call_handoff("comfy-gpu1", [8081])
check("H10 TTL 만료 후 → resident-kept / no-pending-demand",
      r.get("action") == "resident-kept" and r.get("reason") == "no-pending-demand"
      and UNLOADED_CALLS == [], f"action={r.get('action')} reason={r.get('reason')}")
check("H10 만료 항목 자동 제거", A.pending_state() == {}, f"{A.pending_state()}")

print("\n=== H11) paired llama 가 unload 불가 (단일 모델 + protect) → blocked ===")
world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000, unload_ok1=False)
A.register_pending("comfy-gpu1", [8081], 13000, 1, "p-h11")
r = call_handoff("comfy-gpu1", [8081])
check("H11 blocked + unload 0건 + pending 유지",
      r.get("action") == "blocked" and UNLOADED_CALLS == []
      and A.services["llama-8081"].state == va.RESIDENT
      and "comfy-gpu1" in A.pending_state(),
      f"action={r.get('action')} reason={str(r.get('reason'))[:90]}")

print("\n=== H12) b10685 Router 응답 파싱 (status = object, unload 모델도 목록에 남음) ===")
b10685 = {
    "data": [
        {"id": "qwen38", "status": {"value": "loaded", "args": ["-m", "x.gguf"]}},
        {"id": "draft",  "status": {"value": "unloaded"}},
    ],
    "object": "list",
}
legacy = {"data": [{"id": "qwen38", "status": "loaded"}, {"id": "draft", "status": "unloaded"}]}
single = {"data": [{"id": "local-model"}]}

check("H12-1 _model_status: b10685 object 형태",
      va._model_status(b10685["data"][0]) == "loaded"
      and va._model_status(b10685["data"][1]) == "unloaded",
      f"{[va._model_status(i) for i in b10685['data']]}")
check("H12-2 _model_status: 구버전 문자열 형태",
      va._model_status(legacy["data"][0]) == "loaded"
      and va._model_status(legacy["data"][1]) == "unloaded",
      f"{[va._model_status(i) for i in legacy['data']]}")
check("H12-3 _model_status: 단일 모델 서버 (status 없음)", va._model_status(single["data"][0]) == "")

_ORIG_HTTP = va._http_json
va._http_json = lambda method, url, **kw: b10685 if "/models" in url else {}
names_b = va.VramArbiter._loaded_model_names("http://x")
check("H12-4 _loaded_model_names(b10685) = loaded 만 (unloaded 제외 → 400 방지)",
      names_b == ["qwen38"], f"{names_b}")

world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000)
target = A.services["llama-8081"]
target.pid = 4242
va.llama_binary_features = lambda b: {"router": True, "default_model": False}
A._binary_of_pid = staticmethod(lambda pid: "/opt/llama/llama-b10685/bin/llama-server")
_PROBE_ORIGINAL(A, target)
check("H12-5 _probe_llama(b10685) → router_capable=True",
      bool(target.router_capable) is True,
      f"router_capable={target.router_capable} models={target.router_models}")
check("H12-6 _probe_llama(b10685) → unload_supported=True",
      bool(target.unload_supported) is True, f"unload_supported={target.unload_supported}")
check("H12-7 model_loaded_reported=True (loaded 모델 존재)",
      bool(target.model_loaded_reported) is True, f"{target.model_loaded_reported}")

va._http_json = lambda method, url, **kw: {
    "data": [{"id": "qwen38", "status": {"value": "unloaded"}}]}
A._probe_llama = lambda state: None      # restore no-op
_PROBE_ORIGINAL(A, target)
check("H12-8 전부 unloaded → model_loaded_reported=False (UNLOADED 판정 가능)",
      bool(target.model_loaded_reported) is False and bool(target.router_capable) is True,
      f"loaded={target.model_loaded_reported} router={target.router_capable}")
va._http_json = _ORIG_HTTP

print("\n=== H14) 설정 migration (구버전 파일 → 현재 스키마) ===")
stale = {
    "enabled": True, "gpus": {}, "default_safety_margin_mb": 2048,
    "unknown_expected_ratio": 0.35, "llama_router_default_model": "x",   # 사라진 키
    "llama_inject_missing_model": False,
}
stale_path = os.path.join(SANDBOX, "stale.json")
with open(stale_path, "w", encoding="utf-8") as fh:
    json.dump(stale, fh)
M = va.VramArbiter()
M.settings = json.loads(json.dumps(va.DEFAULT_SETTINGS))
M._log_event = lambda *a, **k: None
saved_path = va.ARBITER_SETTINGS_FILE
va.ARBITER_SETTINGS_FILE = stale_path
loaded = M.load_settings()
check("H14-1 사라진 키 제거 (unknown_expected_ratio / llama_router_default_model)",
      "unknown_expected_ratio" not in loaded and "llama_router_default_model" not in loaded,
      f"남은 관련 키={[k for k in loaded if 'ratio' in k or 'default_model' in k]}")
check("H14-2 신규 키 백필 (paired_handoff_enabled / pending_demand_ttl_sec / threshold 기본)",
      loaded.get("paired_handoff_enabled") is True
      and float(loaded.get("pending_demand_ttl_sec")) == 900.0
      and int(loaded.get("default_evict_threshold_percent")) == 0,
      f"handoff={loaded.get('paired_handoff_enabled')} ttl={loaded.get('pending_demand_ttl_sec')}")
check("H14-3 마이그레이션 결과가 디스크에 반영됐다",
      json.load(open(stale_path)).get("paired_handoff_enabled") is True,
      f"{json.load(open(stale_path)).get('paired_handoff_enabled')}")
check("H14-4 사용자 선택 유지 (enabled / llama_inject_missing_model)",
      loaded.get("enabled") is True and loaded.get("llama_inject_missing_model") is False,
      f"enabled={loaded.get('enabled')} inject={loaded.get('llama_inject_missing_model')}")

partial = {"enabled": True,
           "gpus": {U0: {"enabled": True},                        # 필드 대부분 없음
                    U1: {"enabled": True, "safety_margin_mb": 4096,
                         "evict_threshold_percent": 140, "junk": 1}}}   # 범위초과 + 잡키
partial_path = os.path.join(SANDBOX, "partial.json")
with open(partial_path, "w", encoding="utf-8") as fh:
    json.dump(partial, fh)
va.ARBITER_SETTINGS_FILE = partial_path
P = va.VramArbiter()
P._log_event = lambda *a, **k: None
loaded_p = P.load_settings()
g0 = loaded_p["gpus"][U0]
g1 = loaded_p["gpus"][U1]
check("H14-5 GPU 항목 필드 백필 (safety_margin / evict_threshold 기본값)",
      int(g0.get("safety_margin_mb") or 0) == 2048 and g0.get("evict_threshold_percent") == 0,
      f"g0={g0}")
check("H14-6 threshold 범위 clamp (140 → 100) + 잡키 제거",
      g1.get("evict_threshold_percent") == 100 and "junk" not in g1, f"g1={g1}")
check("H14-7 사용자 enabled 값 보존", g0.get("enabled") is True and g1.get("enabled") is True,
      f"g0.enabled={g0.get('enabled')} g1.enabled={g1.get('enabled')}")
P.settings["gpus"][U1]["evict_threshold_percent"] = 40
P.settings["gpus"][U0]["evict_threshold_percent"] = 0
P.save_settings()
P2 = va.VramArbiter()
P2._log_event = lambda *a, **k: None
re_loaded = P2.load_settings()
check("H14-8 부분 수정(PATCH) 후 재로드 유지",
      re_loaded["gpus"][U1]["evict_threshold_percent"] == 40, f"{re_loaded['gpus'][U1]}")

# 스냅샷에만 있는 GPU → ensure_pools 가 항목을 만들어 준다 (UI 가 바로 편집 가능하도록)
before_keys = set(P2.settings.get("gpus") or {})
snapshot_only = {"GPU-new-uuid": dict(index=2, name="SIM", pci_bus_id="", total_mb=8192,
                                      used_mb=0, free_mb=8192, util=0)}
P2.pools["GPU-new-uuid"] = va.GpuPool(uuid="GPU-new-uuid", index=2, name="SIM")
P2._backfill_gpu_settings(snapshot_only)
new_entry = (P2.settings.get("gpus") or {}).get("GPU-new-uuid") or {}
check("H14-9 NVML 스냅샷의 신규 GPU 항목 자동 생성 (enabled 는 건드리지 않음)",
      new_entry.get("index") == 2 and new_entry.get("enabled") is False
      and int(new_entry.get("safety_margin_mb") or 0) == 2048
      and new_entry.get("evict_threshold_percent") == 0,
      f"before={sorted(before_keys)} new={new_entry}")
va.ARBITER_SETTINGS_FILE = saved_path

print("\n=== H13) 엔드포인트 배선: /comfy/admit 이 훅의 jh_llama_nodes 를 사용하는지 ===")
# 라우터 함수들은 모듈 전역 `arbiter` 를 쓰므로 하네스 인스턴스를 바인딩한다.
import asyncio
va.arbiter = A
world(comfy0=va.UNLOADED, llama0=va.RESIDENT, comfy1=va.UNLOADED, llama1=va.RESIDENT,
      free0=1000, free1=3000)
payload_hook = {"port": 8188, "jh_llama_nodes": 1, "llama_ports": [8080], "timeout_sec": 2}
result = asyncio.run(va.comfy_admit(payload_hook))
check("H13-1 훅 스캔 결과 → 수요 등록 경로 (즉시 eviction 아님)",
      result.get("registered") is True and result.get("handoff_required") is True
      and int(result.get("jh_llama_nodes") or 0) == 1,
      f"action={result.get('action')} registered={result.get('registered')} nodes={result.get('jh_llama_nodes')}")
check("H13-2 등록 경로에서는 on_prompt 시점 unload 발생하지 않음",
      UNLOADED_CALLS == [] and A.services["llama-8080"].state == va.RESIDENT,
      f"unloaded={UNLOADED_CALLS} llama={A.services['llama-8080'].state}")

world(comfy0=va.UNLOADED, llama0=va.RESIDENT, free0=1000)
result = asyncio.run(va.comfy_admit({"port": 8188, "jh_llama_nodes": 0, "llama_ports": []}))
check("H13-3 JH 노드 없음 → 즉시 eviction 경로",
      result.get("action") == "evicted+load" and UNLOADED_CALLS == [("llama-8080", U0)],
      f"action={result.get('action')} unloaded={UNLOADED_CALLS}")

world(comfy0=va.UNLOADED, llama0=va.RESIDENT, free0=1000)
legacy_graph = {"1": {"class_type": "JHLlamaPrompt", "inputs": {"server_url": "http://127.0.0.1:8080"}}}
result = asyncio.run(va.comfy_admit({"port": 8188, "prompt": legacy_graph}))
check("H13-4 구버전 훅(그래프 원문) 폴백도 수요 등록",
      result.get("registered") is True and UNLOADED_CALLS == [],
      f"action={result.get('action')} registered={result.get('registered')}")

world(comfy1=va.UNLOADED, llama1=va.RESIDENT, free1=3000)
A.register_pending("comfy-gpu1", [8081], 13000, 1, "p-h13")
hres = va.comfy_handoff({"port": 8189, "llama_ports": [8081]})
check("H13-5 /comfy/handoff 엔드포인트가 paired llama 를 내린다",
      hres.get("action") == "unloaded" and UNLOADED_CALLS == [("llama-8081", U1)],
      f"action={hres.get('action')} unloaded={UNLOADED_CALLS}")

print("\n" + "=" * 64)
print(f"총 {CHECKS}건 중 실패 {len(FAILURES)}건")
for name in FAILURES:
    print(f"  FAIL: {name}")
print("=" * 64)
sys.exit(1 if FAILURES else 0)
