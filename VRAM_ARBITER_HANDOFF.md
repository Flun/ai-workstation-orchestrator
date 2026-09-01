# GPU VRAM Arbiter — GPU 도메인 독립 + Comfy↔llama handoff

**작업 디렉터리**: `/home/flux/main_server_new`
**작성일**: 2026-08-30 (마지막 검증 06:14 KST)
**기준 바이너리**: llama.cpp **b10685** (`/opt/llama/llama-b10685/bin/llama-server`)

## 상태 요약

```
구현 완료:   YES          py_compile:  PASS
domain:      40/40 PASS   handoff/migration: 47/47 PASS
EXIT:        0            TODO/미완성 배선: 0
남은 단계:    실제 런타임 테스트만  (재시작·load/unload·실제 eviction — 지시대로 미수행)
```

> **⚠ 라이브 미반영** — 이 문서는 **디스크** 상태입니다. 실행 중인 main_server / ComfyUI 는
> 편집 **이전** 코드를 읽고 있습니다. 반영하려면 main_server 재시작 + ComfyUI 인스턴스
> 재시작이 필요하며, 수행하지 않았습니다. (llama.cpp 는 재시작 불필요)

---

## 1. 확정 구조

```
GPU0 domain                          GPU1 domain
  ComfyUI :8188  <->  llama.cpp :8080    ComfyUI :8189  <->  llama.cpp :8081
```

| 원칙 | 내용 |
|---|---|
| 소속 GPU의 source of truth | NVML 추정이 아니라 **main_server가 서비스를 시작할 때 이미 아는 service type / port / selected GPU** |
| NVML의 역할 | 실제 VRAM `free`/`used` 측정 및 unload 후 **반환 확인** 용도 (소속 판정 아님) |
| 독립성 | 두 Arbiter가 모두 ON이어도 상태·free·lock·victim 을 **전혀 참조하지 않음** |
| 관리 대상 | main_server가 실행한 ComfyUI / llama.cpp 만. 기타 CUDA 프로세스는 free 에만 반영하고 제어하지 않음 |
| multi-GPU llama | `multi_gpu=True`, `managed_evict=False` → **victim·requester 양쪽 모두 Arbiter 관리 제외** |
| Comfy multi-GPU | 고려하지 않는다 (설정된 1장을 도메인으로 강제) |
| threshold | GPU 별 `evict_threshold_percent` = **이번 요청 예상 VRAM 이 이 GPU 총 VRAM 의 몇 %** (GPU 사용률이 아님) |
| on_prompt | 즉시 eviction 이 아니라 **pending Comfy demand 등록** 지점 |
| handoff | JH llama 호출 완료 후 **Comfy node return 직전** |
| unload 조건 | pending demand 있음 + 실제 VRAM 확보 필요 + threshold 정책 만족 → **같은 GPU llama 만** |
| 그 외 | llama 단독 사용·contention 없음 → **RESIDENT 유지** |

---

## 2. 왜 handoff 가 필요한가 (핵심 판단 근거)

### 2-1. JHLlamaPrompt 실제 경로 (정적 추적 결과)

llama HTTP 를 호출하는 JH 클래스는 **`JHLlamaPrompt` 하나뿐**입니다
(`ComfyUI_JH_Nodes/__init__.py`, 684/697 행의 다른 `urlopen` 은 Civitai 이미지 API).

```
class JHLlamaPrompt            :2438      FUNCTION = "create_prompt"  :2485
  INPUT server_url             :2449      default "http://127.0.0.1:8080" (localhost 전용 :2744)

def _request_json              :2499      urllib.request.urlopen — 동기 블로킹
def _resolve_model             :2511      GET {server_url}/v1/models
                                         → data 비면 RuntimeError("llama.cpp returned no loaded models") :2517

def create_prompt              :2741
  ├─ model = self._resolve_model(server_url, model, timeout)   :2762   ← ①
  ├─ for sheet_mode in sheet_modes:                            :2779   ← Combined 면 2 회
  │    └─ for attempt in range(2):                             :2796   ← truncation 재시도 2 회
  │         response = self._request_json(
  │             f"{server_url}/v1/chat/completions", ...)      :2809   ← ② 마지막 llama 응답
  ├─ 텍스트 후처리 / 캐시 쓰기 / ui payload 구성 (I/O 없음)        :2832-2853
  └─ return {"ui": …, "result": (prompts,)}                    :2854   ← ③ output 반환
```

**handoff 지점 = ② 와 ③ 사이** (original `create_prompt()` 가 돌아온 직후). 여기는 llama 응답을
모두 받았으면서 Comfy 가 output 을 받기 직전이라 요청한 조건을 정확히 만족합니다.

### 2-2. on_prompt 즉시 eviction 이 실패하는 이유 (b10685 기준)

`_resolve_model()`(①) 이 채팅(②) 보다 먼저 `GET /v1/models` 를 칩니다. b10685 라우터는
**unload 된 모델도 `status.value="unloaded"` 로 목록에 남기 때문에** 모델명 해석은 성공하고,
곧 이은 `POST /v1/chat/completions` 에서 라우터가 **autoload 으로 같은 모델을 다시 올립니다**
(`server-models.cpp` `proxy_post` → `ensure_model_ready`).

```
JH node 존재 → node_count=0 으로 오인 → on_prompt 즉시 eviction
→ JH 가 llama autoload → pending 이 없으므로 handoff 는 no-pending
→ llama RESIDENT 유지 → 이후 Comfy 와 다시 contention
```

즉 on_prompt eviction 은 **헛수고**이고 contention 이 그대로 남습니다. 그래서 on_prompt 는
**수요 등록** 만 하고, 실제 unload 은 마지막 JH 노드의 handoff 가 수행합니다.
(JH 노드가 없는 워크플로우는 on_prompt 즉시 eviction 이 정답이므로 그 경로는 유지합니다.)

### 2-3. b10685 에서 실제로 발견된 blocker

`get_router_models()` (server-models.cpp:1963-2032) 는 `status` 를 문자열이 아니라
`{"value": "loaded", "args": [...]}` **object** 로 내고, unload 된 모델도 목록에 남습니다.
기존 코드는 `str(item["status"])` 만 봐서 **`router_capable` 이 영구 False** →
`unload_supported=False` → 라우터 `/models/unload` 경로 전체 사멸 → handoff 전부 `blocked`.

→ `_model_status()` 신설 (dict → `status.value`, 구버전 문자열·단일모델 양립) 를
`_probe_llama()` 2곳과 `_loaded_model_names()` 에 적용. 후자에는 b10685
`post_router_models_unload` 의 `"model is not running"` 400 방지(이미 내려간 이름 미전달) 포함.

`--help` 실측: `--models-dir` O / `router server` O / `default-model` **X** →
`llama_binary_features()` 정상, `llama_inject_missing_model=False` 유지 정확.

---

## 3. 구현 내용

### 3-1. 도메인 결정

| 서비스 | 소속 GPU (`home_gpu`) | 규칙 |
|---|---|---|
| Comfy | `comfyui_settings.json > gpu_device` (UUID) **단일 강제** | NVML 이 두 카드로 보여도 도메인은 설정 1장. `gpu_uuids=[home]` 로 잘라 교차 GPU 접근 원천 차단 |
| llama | `last_run.json > gpuDevices` (**1차**, 디스크라 재시작 후에도 유효) → `services["llama"].device` (2차) → `[]` 면 NVML 폴백 | 소속이 **2장 이상 → `multi_gpu=True`, `managed_evict=False`** (victim·requester 모두 passthrough) |

`_resolve_uuid()` 가 기동 정보의 세 표식(NVML UUID / index `"1"` / `CUDA1`·`cuda:0` / PCI busId)을
pool UUID 하나로 통일합니다.

`reconcile()` → `rebuild_registry()` 로 GPU 별 managed registry 구성:
`gpu_services[GPU0] = [comfy-main, llama-8080]`, `gpu_services[GPU1] = [comfy-gpu1, llama-8081]`
(`home_gpu` 기준, `managed_evict=False` 는 어느 도메인에도 미포함).

### 3-2. 판정 순서 (`admit`)

```
① _deficit(pool, expected) <= 0            → action="load"   (threshold·UNKNOWN 무관, eviction 0)
② _evict_gate(pool, expected, force_open=unknown)
     예상/total*100 >= threshold            → eviction 진행
     미만                                  → action="load-no-evict" + evict_skipped 이벤트
                                             (이번 요청은 OOM 가능 — 사용자 선택)
     UNKNOWN + free 부족                    → 큰 요청으로 간주 (force_open), threshold 적용 생략
③ _evict_locked(home, expected, requester, clear_all=unknown)
     victim = state.home_gpu == home AND managed_evict AND RESIDENT AND hold 없음, LRU
```

UNKNOWN 정책 확정: **free 충분하면 미리 내리지 않는다.** 부족할 때만 큰 요청 취급.

### 3-3. 제거된 교차 GPU 로직

| 제거 대상 | 대체 |
|---|---|
| GPU 간 free 합산 baseline (`_gated_forward`) | `gate_home` 단일 카드 |
| `required = expected + sum(margins)`, `free = sum(free)` | 단일 pool 수치 |
| `sum(gains.values())` 부분성공 판정 | victim 은 소속 GPU 1장 |
| 전역 `max/sum` 용량 상한 (`_llama_file_size_mb`) | 소속 GPU 95% |
| victim 조건 `any(u in enabled_uuids for u in state.gpu_uuids)` | `state.home_gpu == home_uuid AND managed_evict` |
| `_deficit_per_gpu(uuids) -> Dict` | `_deficit(pool, expected) -> int` |

### 3-4. 덤으로 잡은 실동작 크래시 2건

- `_unload()` 실패 시 `return 0` 인데 호출부가 `.items()` → eviction 실패 순간 `AttributeError`
- `_log_event("evict", ..., gpu=...)` — 시그니처에 없는 kwarg 라 **성공 경로 자체가 TypeError**

### 3-5. pending / handoff

```python
register_pending(comfy_id, llama_ports, expected_mb, llama_node_count, prompt_id)
  → pending[comfy_id] = {registered_at, expires_at, home_gpu, expected_mb, estimate_source,
                         deficit_mb_at_register, request_percent, evict_threshold_percent,
                         planned_evict, gate_reason, llama_ports, expected_handoffs, handoffs}
  TTL = pending_demand_ttl_sec (900s) — handoff 없이 끝나도 상주 잠금 영구화 방지
```

```python
handoff(comfy_id, llama_ports, reason)
  pending 없음                        → resident-kept / no-pending-demand   (llama 단독 사용)
  reason startswith "jh-abort"        → pending-cleared (JH 실패 → 수요만 해소, llama 유지)
  handoffs < expected_handoffs        → await-llama-nodes  (아직 실행될 JH 노드)
  free 충분                            → resident-kept
  threshold 미만 (KNOWN)               → resident-kept (OOM 감수)
  통과                                 → paired llama unload → NVML 반환 확인 → unloaded
```

- victim 은 **Arbiter 가 결정** — 호출자는 Comfy 포트만 보냅니다.
- unload 직전 `_probe_llama()` 로 재확인 후 `_handoff_victim_ok()`(진행 중 요청/hold/
  `unload_supported=False`/이미 unload 된 라우터 거부). 재probe 없으면 reconcile 5 초 주기 때문에
  handoff 가 영구 보류됩니다.
- JH 노드 `server_url` 포트가 이 도메인 llama 와 다르면 `[]` → GPU0 요청이 GPU1 llama 를
  내리는 경로 차단.

### 3-6. API

| endpoint | 용도 |
|---|---|
| `POST /api/gpu-arbiter/comfy/admit` | on_prompt 게이트. 훅이 보낸 `jh_llama_nodes`/`llama_ports` 를 1차로 사용. JH 노드 0 이면 즉시 eviction. 항상 200·fail-open |
| `POST /api/gpu-arbiter/comfy/handoff` | JH 노드 return 직전. `{port, llama_ports, reason}` |
| `GET /api/gpu-arbiter/status` | `pending`, `paired_handoff_enabled`, GPU 별 `evict_threshold_percent`·`managed_services`, 서비스 별 `home_gpu/home_index/multi_gpu/managed_evict/domain_note` |
| `POST /api/gpu-arbiter/settings` | `gpus.{uuid}.evict_threshold_percent` 부분 수정, `paired_handoff_enabled`, `pending_demand_ttl_sec` |
| `POST /api/gpu-arbiter/gpus/{uuid}/enable` | GPU 도메인 ON/OFF |

### 3-7. Comfy 사이드

- `main_server_vram_arbiter/__init__.py` — `on_prompt` 에서 그래프를 스캔해 `jh_llama_nodes`
  (class_type 카운트) / `llama_ports`(서버_url 포트) 를 매니저로 전송 → **수요 등록**.
  `jh_handoff.installed()` 확인 후 필요 시 `install()` 재시도 (로드 순서·Manager 재활성 대비).
  예외는 전부 fail-open, **받은 prompt JSON 을 그대로 반환**.
- `jh_handoff.py` — `nodes.NODE_CLASS_MAPPINGS["JHLlamaPrompt"].create_prompt` 를 멱등 래핑
  (JH 소스 무수정). 성공 시 ②→③ 사이 handoff, 예외 시 `jh-abort` 통보 후 **원본 예외 재던지기**
  (Comfy 실패 동작 불변). `HANDOFF_WAIT_SEC=120` (`MAIN_SERVER_ARBITER_HANDOFF_WAIT`).

### 3-8. 설정 migration

`load_settings()` → `_migrate_settings(merged, stored)`:
- 디스크에만 있는 고아 키 제거 (`unknown_expected_ratio`, `llama_router_default_model` 등) —
  `merged` 만 검사하면 영구 잔존하므로 `stored` 와 대조
- 신규 키 백필 (`paired_handoff_enabled`, `pending_demand_ttl_sec`, `default_evict_threshold_percent`)
- GPU 항목 필드 정규화 (`safety_margin_mb`, `evict_threshold_percent` clamp 0-100, 잡키 제거)
- `enabled` 는 사용자 선택이므로 절대 변경 안 함 (기본 OFF)
- `_backfill_gpu_settings()`: NVML 스냅샷에만 있는 GPU 항목 자동 생성 → 재시작 후 UI 에서
  카드마다 threshold 를 바로 편집 가능
- 변경 시 디스크 저장 (UI 가 첫 로드부터 올바른 기본값 표시)

---

## 4. 변경 파일

| 파일 | 줄 | 내용 |
|---|---|---|
| `main_server_new/vram_arbiter.py` | 2636 | 도메인 단일화, threshold(요청 크기 %), UNKNOWN 확정 정책, `_model_status()` b10685, pending/handoff/abort, `/comfy/admit`의 `jh_llama_nodes` 수신, `rebuild_registry()`, 설정 migration |
| `main_server_new/app.py` | 3504 | `_arbiter_llama_devices(port)` 신설 + `configure(llama_devices=…)` |
| `main_server_new/index.html` | — | GPU 별 `Evict __ %` 입력, `arbiterThresholdHelp()` (판단 4단계 + OOM 경고 + “GPU 사용률 아님”), 멀티 GPU `제외` 배지 |
| `main_server_new/test_arbiter_domain.py` | 332 | 도메인 독립성 40건 (신규) |
| `main_server_new/test_arbiter_handoff.py` | 436 | pending/handoff/abort/b10685/migration 47건 (신규) |
| `ComfyUI/custom_nodes/main_server_vram_arbiter/__init__.py` | 164 | on_prompt 수요등록 + 훅 스캔 전송 + 설치 재시도 |
| `ComfyUI/custom_nodes/main_server_vram_arbiter/jh_handoff.py` | 158 | `create_prompt` 래퍼 (신규) |

주요 설정 키 (`vram_arbiter.py DEFAULT_SETTINGS`):

```
enabled=True  gpus={}  default_evict_threshold_percent=0  default_safety_margin_mb=2048
admission_timeout_sec=900  unload_timeout_sec=60  llama_unload_mode="protect"
active_util_percent=5  unknown_clear_residents=True  post_admit_grace_sec=30
unknown_expected_min_mb=4096  paired_handoff_enabled=True  pending_demand_ttl_sec=900
llama_router_enabled=True  llama_inject_missing_model=False
```

---

## 5. 검증 결과 (최종 실행)

```
--- py_compile (main_server_new) ---
py_compile: PASS      # vram_arbiter.py app.py test_arbiter_domain.py test_arbiter_handoff.py
--- py_compile (Comfy hook) ---
hook py_compile: PASS # __init__.py jh_handoff.py
--- domain ---
domain EXIT=0  PASS=40  FAIL=0
--- handoff ---
handoff EXIT=0  PASS=47  FAIL=0
OVERALL_EXIT=0
```

실행 방법:

```bash
cd /home/flux/main_server_new
.venv/bin/python -m py_compile vram_arbiter.py app.py test_arbiter_domain.py test_arbiter_handoff.py
.venv/bin/python test_arbiter_domain.py
.venv/bin/python test_arbiter_handoff.py
```

두 하네스 모두 NVML / 네트워크 / 실제 프로세스 / 실제 설정·이벤트 파일에 **접근하지 않습니다**
(`read_gpu_snapshot`·`read_process_map`·`_unload`·`_probe_llama`·`_http*` 스텁,
설정/이벤트 경로는 tempfile sandbox 로 우회). GPU lock 은 `_PoolLock.__enter__` 를 관측해
“다른 카드의 lock 이 실제로 잡히지 않는가” 를 확인합니다.

### domain 40건

| 섹션 | 확인 |
|---|---|
| 0 | registry = `gpu0[comfy-main, llama-8080]` / `gpu1[comfy-gpu1, llama-8081]`, 멀티 GPU llama 양쪽 부재 |
| 1 | GPU0 req → GPU1 unload 0건, evicted 는 GPU0 소속만, `acquired=={GPU0}` |
| 2 | GPU1 req → GPU0 unload 0건, peer(`llama-8081`) 만 unload, `acquired=={GPU1}` |
| 3 | 멀티 GPU llama — requester `passthrough`/`multi-gpu-instance-excluded`, lock 0, victim 양 도메인 제외 |
| 4 | threshold = 요청 크기 % — 32.6%<50 → `load-no-evict`+`home_deficit_mb=7048` / 52.9%≥50 → `evicted+load` / threshold=0 → 항상 |
| 5 | free 충분 + threshold 이상 → eviction 0 (`action=load`, peer RESIDENT 유지) |
| 6 | UNKNOWN — free 충분 → 미리 내리지 않음 / 부족 + 16.7%<50 → 큰 요청 간주 eviction |
| 7 | `free_mb`/`deficit_mb` dict 유지 + `home_free_mb`/`home_deficit_mb` scalar 병존 |
| 8 | `_resolve_uuid` — UUID / `"1"` / `CUDA1` / `cuda:0` / PCI busId, 미지 토큰 `''` |
| 9 | C 훅 `_arbiter_llama_devices` 를 **app.py 에서 AST 로 추출해 실실행** — `last_run.json` gpuDevices 1차, 포트 불일치 `[]`, 메모리 `device` 보다 우선 |
| 10 | `_active_blockers` — 각 도메인 ACTIVE 만 |

### handoff 47건

| # | 확인 |
|---|---|
| H1 | pending 없음 → `resident-kept`/`no-pending-demand`, unload 0 |
| H2 | pending + free 충분 → `planned_evict=False`, 등록 시 unload 0, handoff → `resident-kept` |
| H3 | pending + 부족 + 52.9%≥50 → `unloaded`, `freed=18000 free=21000 deficit=0`, pending 해소 |
| H4 | threshold 미만 (32.6%) → `resident-kept` + 부족량 노출 |
| H5 | 다중 JH — `await-llama-nodes 1/2` 후 마지막에만 unload |
| H6 | `jh-abort` → `pending-cleared`, unload 0, pending 소멸 |
| H7 | GPU0 handoff → GPU1 미unload, `acquired=={U0}`, GPU1 RESIDENT 유지 |
| H8 | 멀티 GPU llama — `_paired_llamas` 양 도메인 후보 제외, handoff 도 미unload |
| H9 | JH 가 다른 GPU 포트를 가리킴 → `no-paired-llama`, unload 0, 양쪽 RESIDENT |
| H10 | TTL 만료 → `resident-kept`/`no-pending-demand` + 자동 제거 |
| H11 | `unload_supported=False` → `blocked`, unload 0, pending 유지 |
| H12 | b10685 `status.value` object / 구버전 문자열 / 단일모델, `_loaded_model_names` unloaded 제외, `router_capable=True`, `unload_supported=True`, 전부 unloaded → `model_loaded_reported=False` |
| H14 | migration — 고아 키 제거+디스크 반영, 신규 키 백필, 사용자 선택 유지, 필드 백필, clamp 140→100, PATCH 재로드 유지, 신규 GPU 항목 자동 생성 |
| H13 | `/comfy/admit` 이 훅의 `jh_llama_nodes` 로 수요등록(`registered=True`), 노드 0 → 즉시 eviction, 구버전 그래프 폴백, `/comfy/handoff` unload |

---

## 6. 남은 단계 — 실제 런타임 테스트만

### 런타임 전 필수 확인

현재 `vram_arbiter_settings.json` 의 `gpus` 가 `{}` → **두 GPU Arbiter OFF** 상태입니다
(기존과 동일, 회귀 아님). ON 토글이 절차 1 단계에 포함되어 있습니다.

이 머신의 실제 구성:

```
GPU0 CMP 170HX 64GB  GPU-6cd658de-2bfc-dbd7-78d3-beec843855d9
GPU1 RTX 3090 24GB   GPU-309d93a2-b61f-60c2-d5d3-9a79f26e7468
llama-server :8080  --device CUDA1,CUDA0 --split-mode layer --tensor-split 7,1
```

`:8080` 은 2장 걸침이므로 정책상 **관리 제외**입니다. GPU0 도메인에서 llama 관리를 실측하려면
GPU0 단독 실행 구성이 필요합니다 (현재 서비스는 건드리지 않았습니다).

### 절차

```bash
cd /home/flux/main_server_new
cp gpu_arbiter_events.json gpu_arbiter_events.before.json
```

1. **main_server 재시작** → 로그: `[arbiter] 설정 파일을 현재 스키마로 마이그레이션 했습니다`,
   reconcile 로그. (매니저 프로세스만 재시작 — Comfy/llama 는 `start_new_session` 로 분리돼 살아남음)
2. **Comfy `:8188` / `:8189` 각 인스턴스만 재시작** → 로그:
   - `[vram-arbiter] JH llama handoff 설치 완료 (llama 응답 완료 → 노드 return 직전, 대기 120초)`
   - `[vram-arbiter] on_prompt 수요등록 훅 준비 완료`
3. **llama.cpp `:8080` 은 재시작하지 않음** (RESIDENT 유지 확인용)
4. **Arbiter ON + GPU 별 threshold**

```bash
for u in GPU-6cd658de-2bfc-dbd7-78d3-beec843855d9 GPU-309d93a2-b61f-60c2-d5d3-9a79f26e7468; do
  curl -s -X POST localhost:8999/api/gpu-arbiter/gpus/$u/enable \
    -H 'Content-Type: application/json' -d '{"enabled":true}'; done

curl -s -X POST localhost:8999/api/gpu-arbiter/settings -H 'Content-Type: application/json' \
 -d '{"gpus":{"GPU-6cd658de-2bfc-dbd7-78d3-beec843855d9":{"evict_threshold_percent":0},
              "GPU-309d93a2-b61f-60c2-d5d3-9a79f26e7468":{"evict_threshold_percent":40}}}'
```

5. **상태 스냅샷**

```bash
curl -s localhost:8999/api/gpu-arbiter/status | .venv/bin/python -c "
import json,sys; d=json.load(sys.stdin)
print('handoff:',d['paired_handoff_enabled'],'pending:',d['pending'])
for g in d['gpus']:
  print(f\"GPU{g['index']} ON={g['arbiter_enabled']} free={g['free_vram_mb']} thr={g['evict_threshold_percent']}% reg={g['managed_services']}\")
for s in d['services']:
  print(f\"  {s['id']:12s} home=GPU{str(s['home_index']):4s} multi={s['multi_gpu']} managed={s['managed_evict']} router={s['router_capable']} unload_ok={s['unload_supported']} {s['state']} {s.get('domain_note','')}\")"
```

### 합격 기준

| # | 시나리오 | 기대 |
|---|---|---|
| 0 | status | `managed_services` GPU0 `[comfy-main, llama-8080]` / GPU1 `[comfy-gpu1, llama-8081]`, 멀티 GPU llama `multi_gpu=True·managed_evict=False`, **`router_capable=True`** |
| 1 | llama 단독 사용 | 응답 후에도 `RESIDENT`, 이벤트 없음 |
| 2 | JH 없는 이미지 워크플로우 | on_prompt 즉시 eviction, `evict` `GPU1 N MB 반환` |
| 3 | **JH llama + 큰 모델** | `pending` → `evict`, 워크플로우 **성공**, 로그 `[vram-arbiter] handoff — llama-8081 unload 완료 (+NMB 반환 확인)` |
| 4 | Combined(다중 JH) | 첫 handoff debug `await-llama-nodes`, 마지막 노드에서만 unload |
| 5 | GPU 독립 | GPU0 워크플로우 중 GPU1 llama `RESIDENT` 유지, `evict` 이벤트에 GPU1 미포함 |

### 추적 / 롤백

```bash
tail -f gpu_arbiter_events.json   # kind: pending / evict / evict_skipped / handoff_blocked / wait / pending_cleared

# handoff 만 끄기 (on_prompt 즉시 eviction 으로 회귀, 코드 유지)
curl -s -X POST localhost:8999/api/gpu-arbiter/settings -H 'Content-Type: application/json' \
  -d '{"paired_handoff_enabled":false}'
# Comfy 훅 전체 끄기 (삭제 하지 않고 이름 변경)
mv /home/flux/ComfyUI/custom_nodes/main_server_vram_arbiter{,.disabled}
```

`handoff_blocked` 가 나면 `curl :8080/models` 의 `status.value` 와 `unload_supported` 를 먼저
대조하십시오 (b10685 경로).

---

## 7. 환경 변수

| 변수 | 기본 | 위치 |
|---|---|---|
| `MAIN_SERVER_ARBITER_URL` | `http://127.0.0.1:8999` | Comfy 훅 |
| `MAIN_SERVER_ARBITER_WAIT` | `900` (초) | on_prompt 수요등록 대기 (fallback 즉시 eviction 시) |
| `MAIN_SERVER_ARBITER_HANDOFF_WAIT` | `120` (초) | JH 노드 return 직전 handoff 대기 |

---

## 7-B. ⚠ 라이브 매니저가 설정 파일을 구스키마로 덮어씁니다 (실측)

작업 중 `vram_arbiter_settings.json` 이 **17:33** 에 다시 쓰였습니다. 이 문서 작성 중 재확인한
상태:

```
실행 중: PID 1609  2026-08-29 03:10:39 시작  app.py        ← 구코드 (도메인화/handoff 이전)
소스  : app.py 08-29 22:18 / vram_arbiter.py 08-30 05:23   ← 디스크는 신코드
설정  : 08-30 17:33 (라이브 프로세스가 저장)
```

라이브는 **구코드 인메모리 설정** 을 그대로 디스크에 쓰므로:

| 항목 | 현재 디스크 | 원인 |
|---|---|---|
| `paired_handoff_enabled`, `pending_demand_ttl_sec`, `default_evict_threshold_percent`, `post_admit_grace_sec`, `unknown_clear_residents`, `unknown_expected_min_mb` | **없음** (6 개 키 소실) | 라이브 구코드 `DEFAULT_SETTINGS` 에 없는 키 → 저장 시 유실 |
| `llama_inject_missing_model` | `true` (이전 `false`) | 라이브 인메모리 값 |
| `gpus` | `{GPU1: {enabled:false, index:1, name:...}}` | 라이브가 GPU1 을 OFF 로 저장 |

**중요**: 이건 하네스 소행이 아닙니다. 두 하네스는 설정/이벤트 경로를 tempfile sandbox 로
우회하며, 16:59 기준 mtime 을 재확인했습니다. 원인은 **재시작되지 않은 라이브 프로세스** 입니다.

### 대응

1. **라이브를 건드리지 마십시오.** 지금 설정 파일을 손으로 고쳐도 라이브가 다음 저장 때 다시
   덮어씁니다 (경쟁 불가).
2. `llama_inject_missing_model` 은 A안 기준 **`false`** 입니다 (게이트가 `model` 필드를 대신
   채우지 않음). 재시작 **후** 아래로 맞춰야 유지됩니다:
   ```bash
   curl -s -X POST localhost:8999/api/gpu-arbiter/settings -H 'Content-Type: application/json' \
     -d '{"llama_inject_missing_model":false}'
   ```
3. 재시작하면 신코드의 `_migrate_settings()` 가 위 6 개 키를 자동 백필하고 고아 키를 정리합니다
   (H14-1〜H14-9 에서 검증). 즉 **재시작 + 마이그레이션** 이 정답이고, 수동 복구는 필요 없습니다.
4. 재시작 직후 확인:
   ```bash
   .venv/bin/python -c "import json;d=json.load(open('vram_arbiter_settings.json'))
   print('주요 키:', {k:d.get(k) for k in ('paired_handoff_enabled','pending_demand_ttl_sec','default_evict_threshold_percent','llama_inject_missing_model')})
   print('gpus:', d.get('gpus'))"
   ```
   `paired_handoff_enabled=true`, `pending_demand_ttl_sec=900`, `llama_inject_missing_model=false`,
   GPU0/GPU1 항목에 `safety_margin_mb`·`evict_threshold_percent` 가 채워져 있어야 합니다.

---

## 8. 이 문서가 대체하는 것

- `ARBITER_GPU_DOMAIN.md` (2026-08-30 00:25) — 도메인 독립화 1차, handoff 이전
- `ARBITER_DOMAIN_PROGRESS.md` (2026-08-30 04:48) — 도메인 독립화 진행, handoff 이전

두 문서는 위 기능의 부분집계이고 handoff·b10685·migration 을 담지 않으므로, 이 문서가
최종 기준입니다. (원본 삭제는 하지 않았습니다 — 불필요하면 삭제하셔도 됩니다.)
