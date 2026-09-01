# GPU VRAM Arbiter — GPU 도메인 독립화 작업 요약

**작업 디렉터리**: `/home/flux/main_server_new`
**작성일**: 2026-08-29 (최종 확인 2026-08-30 00:21 KST)
**상태**: 코드 적용 완료 · `py_compile` 통과 · 정적 검증 **40건 / 실패 0건**
**런타임 테스트(서비스 재시작·모델 load/unload·실제 eviction): 지시에 따라 미수행**

> **⚠ 라이브 적용 여부** — 이 문서가 적힌 시점의 디스크 상태는 반영됐지만, **메인 매니저
> 프로세스는 코드를 다시 읽지 않았습니다.** 실행 중인 매니저는 `app.py` 편집 **이전**
> (2026-08-29 03:10:39 시작, PID 1609) 에 뜬 프로세스입니다. 즉 도메인화·threshold·
> `llama_devices` 훅은 **디스크에만 있고 라이브에는 아직 없습니다.** 반영하려면 매니저
> 재시작이 필요하며, 이건 지시대로 수행하지 않았습니다.

---

## 1. 목표

기존 Arbiter는 NVML 실측 기반으로 GPU 간 free VRAM 을 합산·비례배분하는 멀티 GPU
스케줄링을 썼습니다. 이를 **GPU 별 완전 독립 도메인**으로 단순화합니다.

```
GPU0 domain                    GPU1 domain
  comfy :8188  <-> llama :8080    comfy :8189  <-> llama :8081
```

- GPU0 에서 발생한 managed request 는 **GPU0 소속 managed 서비스만** 판단/eviction 대상.
  GPU1 의 상태·free·lock·victim 을 **전혀 참조하지 않는다.**
- 소속 GPU 판단의 1차 소스는 NVML 추정이 아니라 **main_server 가 서비스를 시작할 때 이미
  아는 type / port / selected GPU 정보**. NVML 은 실제 free/used 와 검증용으로만 사용.
- GPU 별 사용자 설정 **evict threshold(%)** 신설.
- request/victim 은 main_server 가 관리하는 ComfyUI · llama.cpp 로 한정.
  기타 CUDA 프로세스는 NVML free 에만 반영하고 제어하지 않는다.
- multi-GPU llama 인스턴스는 eviction 관리에서 **제외** (victim·requester 모두).

---

## 2. 정책 확정

### 2-1. evict_threshold_percent = **요청 크기 %** (GPU 사용률이 아님)

> 24GB GPU + threshold 50 → 예상 VRAM **12GB 이상**인 요청만 "큰 요청"으로 본다.

판단 순서:

| # | 조건 | 동작 |
|---|---|---|
| ① | `free >= expected + safety_margin` | 그대로 공존, **eviction 없음** (threshold·UNKNOWN 무관) |
| ② | free 부족 + `expected / total * 100 >= threshold` | 같은 `home_gpu` 의 RESIDENT managed peer eviction 허용 |
| ③ | free 부족 + `expected / total * 100 < threshold` | **resident eviction 하지 않음** → 이번 요청은 OOM 가능 (사용자 선택) |
| ④ | `threshold = 0` | 부족하면 항상 eviction (전통 동작) |

### 2-2. UNKNOWN (예상 VRAM 미학습) 정책

| 조건 | 동작 |
|---|---|
| free 충분 | **UNKNOWN 이어도 eviction 없음** — "추정 불가"라는 이유만으로 상주를 미리 내리지 않는다 |
| free 부족 + UNKNOWN | **큰 요청으로 간주 → threshold 충족 취급** → 같은 `home_gpu` 의 RESIDENT eviction |

### 2-3. multi-GPU llama

기동 정보의 소속 GPU 가 2장 이상 → `multi_gpu=True`, `managed_evict=False`.
**victim 후보에서도, requester 게이트에서도 제외** (`admit()` 은 `passthrough`,
`reason=multi-gpu-instance-excluded`). 어느 도메인 registry 에도 들지 않는다.

---

## 3. 변경 파일

| 파일 | 줄 수 | 변경 |
|---|---|---|
| `vram_arbiter.py` | 2144 | 도메인 단일화 + threshold 게이트 + 크래시 수정 (미추적 신규 파일) |
| `app.py` | 3504 | `llama_devices` 훅 배선 |
| `index.html` | — | GPU 별 Evict threshold 입력 + 툴팁 + 제외 배지 |
| `test_arbiter_domain.py` | 326 | **신규** 정적 검증 하네스 |
| Comfy 훅 `custom_nodes/main_server_vram_arbiter/__init__.py` | — | **변경 없음** (자기 포트만 POST → 서버 쪽 도메인화로 자동 독립) |

### 3-1. `vram_arbiter.py` 핵심 위치

| 라인 | 심볼 | 내용 |
|---|---|---|
| 525 | `rebuild_registry()` | GPU 별 managed service registry (기준 `home_gpu`) |
| 540 | `managed_services(uuid)` | 도메인 소속 서비스 id 목록 |
| 730 | `_resolve_uuid(token)` | UUID / index / `CUDA1` / PCI busId → pool UUID 통일 |
| 761 | `_detect_comfy()` | 소속 = `comfy_device`(설정) **단일 강제**, `multi_gpu=False` |
| 819 | `_detect_llama()` | 소속 = `llama_devices(port)` 기동 정보, 2장↑ → 관리 제외 |
| 961 | `_deficit(pool, expected)` | 부족량 (MB) — **항상 단일 pool** |
| 972 | `_request_percent(pool, expected)` | 요청 크기 %. 용량 미확인 시 `-1.0` (게이트 오판 방지) |
| 983 | `_evict_gate(pool, expected, *, force_open=False)` | threshold 게이트. `force_open` = UNKNOWN |
| 1013 | `_passthrough_reason()` | `multi-gpu-instance-excluded` / `no-gpu-domain` / `unsupported-service` |
| 1021 | `admit()` | ①여유 → ②threshold → ③eviction. lock/판정 전부 `[home]` |
| 1188 | `_evict_locked(home_uuid, ...)` | victim 후보 `state.home_gpu == home_uuid and managed_evict` |
| 1226 | `_unload(service, home_uuid) -> int` | 반환형 int 통일, 소속 GPU 1장만 관찰 |
| 1315 | `_active_blockers(home_uuid, ...)` | 같은 도메인 ACTIVE 만 대기 대상 |
| 1688/1719 | `summary()` / `status()` | 필터 `state.home_gpu == uuid`, `evict_threshold_percent`·`managed_services` 노출 |
| 1759 | `_service_payload()` | 인스턴스 메서드 전환 + 도메인 필드 |
| 2053 | `_gated_forward()` | baseline 을 소속 GPU 1장으로 (전 GPU 합산 삭제) |

### 3-2. 제거된 로직

| 제거 대상 | 이유 |
|---|---|
| `_deficit_per_gpu(uuids) -> Dict` | 도메인 1장이므로 `_deficit(pool) -> int` 로 대체 (잔재 0건 확인) |
| GPU 간 free 합산 baseline (`sum(pool.free_mb for pool in arbiter.pools.values())`) | 도메인 독립 위반 + 학습 peak 오염 |
| `required = expected + sum(margins)` / `free = sum(free)` | 교차 합산 판정 잔재 |
| `sum(gains.values())` 부분성공 판정 | victim 은 소속 GPU 1장 |
| 전역 `max/sum` 용량 상한 (`_llama_file_size_mb`) | GPU0+GPU1 합산 144GB 기준 → 소속 1장으로 |
| victim 조건 `any(u in enabled_uuids for u in state.gpu_uuids)` | 멀티 GPU 인스턴스가 **양쪽 도메인 모두**의 victim 이 됨 |
| GPU 용량 비례 expected 분배 | 이전 세션에서 이미 제거 (도메인 모델에서 불필요) |

### 3-3. 덤으로 수정한 실동작 결함

1. `_unload()` 실패 시 `return 0` 인데 호출부가 `.items()` → **eviction 실패 순간 AttributeError**
2. `_log_event("evict", ..., gpu=...)` — 시그니처에 없는 kwarg → **성공 경로 자체가 TypeError**
3. `settings["gpus"]` 의 `evict_threshold_percent` 가 파싱·저장만 되고 **어디서도 참조되지 않음**
4. `ServiceState.home_gpu / multi_gpu / managed_evict` 가 **선언만 되고 대입되지 않음**
   → `gpu_services` registry 가 매번 빈 dict

### 3-4. API 응답 형상

기존 호환 **유지** + scalar 추가:

```python
"free_mb":        {home_uuid: int},   # dict 유지
"deficit_mb":     {home_uuid: int},   # dict 유지
"home_free_mb":   int,                # 신규 scalar
"home_deficit_mb": int,               # 신규 scalar
"home_gpu":       uuid, "home_index": int, "request_percent": float,
"evict_threshold_percent": int, "note": str   # load-no-evict 사유
```

> `index.html` 과 Comfy 훅이 `free_mb`/`deficit_mb`/`freed_per_gpu` 를 **읽지 않음**을
> grep 으로 확인 → 형상 변경으로 인한 소비자 파손 없음.

### 3-5. `app.py` — 소속 GPU 1차 소스 (3354행)

```python
def _arbiter_llama_devices(port):
    # 1) last_run.json > gpuDevices  ← UUID 로 기록, 디스크라 main_server 재시작 후에도 유효
    # 2) services["llama"].device    ← 이번 프로세스에서 방금 start() 한 경우
    # 3) []                          ← 없으면 arbiter 가 NVML 로 폴백 (검증용)
```

포트가 일치하는 기동정보만 반환하므로 **남의 포트 기동정보를 참조하지 않는다**.

### 3-6. `index.html`

- 183–188행: GPU 별 `Evict __ %` 입력 → `setArbiterThreshold()` (1446행)
- 1399행 `arbiterThresholdHelp()`: "GPU 사용률이 아니라 이번 요청 예상 VRAM 이 이 카드 총
  VRAM 의 몇 % 인지 기준", 판단 4단계, 그리고 명확한 경고:

  > ※ 3 번에서 요청은 상주 모델을 지켜준 대가로 VRAM 부족(OOM) 이 날 수 있습니다. 의도된 동작입니다.

- 193행: `multi_gpu || managed_evict === false` → `제외` 배지
- estimate 소스 라벨 정정: `unknown` → `첫 실행 · free 충분하면 eviction 없음`

---

## 4. 검증 결과

### 4-1. 컴파일

```
.venv/bin/python -m py_compile vram_arbiter.py app.py   →  OK (A·B·C·D 반영 후 실행)
```

### 4-2. 정적 도메인 독립성 테스트

```bash
cd /home/flux/main_server_new
.venv/bin/python test_arbiter_domain.py       # 총 40건 중 실패 0건 (EXIT=0)
```

하네스는 NVML / 네트워크 / 실제 프로세스에 **전혀 접근하지 않습니다.**
`read_gpu_snapshot`, `read_process_map`, `_unload`, `_log_event` 를 스텁/no-op 으로
대체 → 이벤트 파일 기록도, Comfy `/free`·llama `/models/unload` POST 도 발생하지 않습니다.
`_PoolLock.__enter__` 를 관측해 **어떤 GPU lock 이 실제로 잡혔는지** 까지 확인합니다.

| 요청하신 검증 항목 | 결과 | 근거 출력 |
|---|---|---|
| **GPU0 request → GPU1 victim 0건** | ✅ | `unloaded=[('llama-8080', GPU0)]`, `acquired=[GPU0]` (GPU1 lock 미획득) |
| **GPU1 request → GPU0 victim 0건** | ✅ | `unloaded=[('llama-8081', GPU1)]`, `acquired=[GPU1]` |
| **multi-GPU llama → 양쪽 victim/requester 제외** | ✅ 5건 | requester `reason=multi-gpu-instance-excluded`, `acquired=[]` / victim 측 양 도메인 후보 제외 |
| **threshold → request 크기 % 기준** | ✅ 4건 | `pct=32.6%` → `load-no-evict` + `home_deficit_mb=7048` / `pct=52.9%` → `evicted+load` / `threshold=0` → 항상 eviction |
| **free 충분 → threshold 이상이어도 불필요한 eviction 없음** | ✅ | `pct=52.9%` 인데 free 20GB → `action=load`, `unloaded=[]`, peer `RESIDENT` 유지 |

추가 검증 항목:

- **registry**: `gpu0=[comfy-main, llama-8080]`, `gpu1=[comfy-gpu1, llama-8081]`, `llama-cross` 양쪽 부재
- **UNKNOWN 확정안**: free 충분 → `action=load` + `unloaded=[]` (미리 내리지 않음) /
  free 부족 + `pct=16.7% < 50%` → `evicted+load` (큰 요청 간주)
- **응답 호환**: `free_mb`·`deficit_mb` = `dict`, `home_free_mb=21000`, `home_deficit_mb=0`
- **`_resolve_uuid`**: `UUID` / `"1"` / `CUDA1` / `cuda:0` / PCI busId 해석, `CUDA9`·`""` → `""`
- **C 훅**: `app.py` 에서 AST 로 함수 소스를 추출해 **실제 코드 그대로** 실행 →
  `last_run.json` 실제 UUID 2개 반환, 포트 불일치 → `[]`, 디스크 값이 메모리 `device` 보다 우선
- **`_active_blockers`**: GPU0 은 GPU0 ACTIVE 만, GPU1 은 GPU1 ACTIVE 만

테스트 세계: GPU0 = CMP 170HX 65536MB / GPU1 = RTX 3090 24576MB, threshold GPU0=0 · GPU1=50.

### 4-3. 최종 재검증 (문서 작성 직전 재실행)

```
wc -l vram_arbiter.py app.py test_arbiter_domain.py   →  2144 / 3504 / 326   (문서 기재와 일치)
.venv/bin/python -m py_compile vram_arbiter.py app.py test_arbiter_domain.py  →  OK
.venv/bin/python test_arbiter_domain.py              →  EXIT=0, 총 40건 중 실패 0건
§3-1 의 심볼 라인 번호 17건                              →  grep 재대조 결과 전부 일치
§3-6 의 index.html 심볼 (183/187/193/1399/1446)          →  일치
```

---

## 5. 현재 머신 상태 (테스트로 확인된 사실)

```
GPU0  GPU-6cd658de-2bfc-dbd7-78d3-beec843855d9  CMP 170HX        65536 MiB
GPU1  GPU-309d93a2-b61f-60c2-d5d3-9a79f26e7468  GeForce RTX 3090 24576 MiB

comfyui_settings.json   main → GPU0(UUID)      gpu1 → GPU1(UUID)
last_run.json           port 8080, gpuDevices = [GPU0, GPU1]      ← 2장 걸침
                        (mtime 08-29 03:58:40 — 아래 실행 중인 인스턴스보다 오래됨)
실행 중 llama-server    build-qsa/bin/llama-server --device CUDA1,CUDA0
                        --split-mode layer --tensor-split 6,1 --port 8080
```

> `--tensor-split` 은 관측 시점에 따라 바뀔 수 있습니다 (08-29 확인값 `7,1` → 08-30 00:21 확인값 `6,1`).
> 도메인 판정에 영향을 주는 것은 **`--device` 에 걸린 GPU 장 수**뿐이고, 그 값은 어느 쪽이든 2장입니다.

→ **현재 떠 있는 `llama :8080` 은 신규 정책에서 `multi_gpu=True` 로 자동 관리 제외**됩니다.
   GPU0 도메인에서 관리받게 하려면 GPU0 단독으로 재실행해야 합니다 (재시작하지 않았습니다).

### 5-1. 최종 확인 시점의 실제 프로세스 (2026-08-30 00:21 KST 재측정)

| 대상 | 상태 |
|---|---|
| 메인 매니저 (`app.py`) | **실행 중** — PID 1609, 08-29 03:10:39 시작. `--reload` 아님 → **코드 편집 반영 안 됨** (22:18 편집분 미반영) |
| `llama-server :8080` | **실행 중** — 08-30 **00:12:06** (재)시작, `build-qsa` 빌드, GPU0+GPU1 걸침. 문서 최초 작성(00:03) 이후 **한 번 더 재시작**된 상태 |
| ComfyUI (main / gpu1) | **실행 중 아님** — 두 인스턴스 모두 떠 있지 않음 |

> **`llama_devices` 훅과 스테일 데이터 주의** — 훅은 `last_run.json`(03:58:40) 을 1차로 읽습니다.
> 지금처럼 API 가 아니라 **CLI 로 직접** llama 을 띄우면 `last_run.json` 이 갱신되지 않아
> 디스크 값과 실제 실행 인자가 어긋날 수 있습니다. 이번에는 둘 다 "2장 걸침" 이라 결론이
> 우연히 일치했지만, **CLI 로 GPU 1장만 지정해 띄우면 훅은 낡은 2장 값을 돌려
> 부당하게 관리 제외(`multi_gpu=True`) 될 수 있습니다.** 이 경우 `last_run.json` 을
> 실제 실행과 맞추거나, `/api/llama/start` 로 시작해 기록을 갱신하세요.

> 매니저가 `--reload` 없이 떠 있으므로 **`app.py` 를 다시 읽으려면 수동 재시작이 필요**합니다.
> ComfyUI 가 지금 꺼져 있다는 뜻은, 훅이 등록될 프로세스 자체가 없다는 점이기도 합니다
> (Comfy 훅 자체는 이번 작업에서 변경하지 않았고, 켜지면 기존 방식으로 등록됩니다).

---

## 6. 남은 작업 / 주의사항

### 반드시 확인

1. **`vram_arbiter_settings.json` 의 `gpus` 가 `{}` 입니다.**
   `evict_threshold_percent` 키가 없으면 **기본 0 = "부족 시 항상 eviction"** 이라 기존
   동작과 동일합니다. GPU1 에 40 을 적용하려면:
   ```bash
   # UI: GPU 카드 > Evict 입력 (권장)
   # 또는
   curl -s -X POST http://127.0.0.1:8999/api/gpu-arbiter/settings \
     -H 'Content-Type: application/json' \
     -d '{"gpus":{"GPU-309d93a2-b61f-60c2-d5d3-9a79f26e7468":{"evict_threshold_percent":40}}}'
   ```
2. **main_server 가 `--reload` 로 떠 있으면 `app.py` 편집으로 매니저가 재시작**될 수 있습니다.
   ComfyUI/llama 는 `start_new_session` 로 분리돼 살아남지만 원치 않는 재시작을 피하려면 확인 후 직접 재시작하세요.
3. 디스크의 구버전 설정 키 (`llama_router_default_model` 등) 는 로더가 무시하므로 지장 없으나,
   `"llama_inject_missing_model": false` 기준 유지는 권장됩니다.

### 미구현 (의도적)

- `/arbiter/llama/{path}` 프록시는 `arbiter.llama_port()` 단일 값 사용 → 다중 포트 llama
  게이트 미지원. **지시에 따라 이번 작업 범위 밖**입니다.
- Comfy 의 실질적 multi-GPU 미지원 (정책상).

### 롤백

```bash
cd /home/flux/main_server_new
# 기능만 끄기 (권장 — 코드 유지)
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("vram_arbiter_settings.json")
d = json.loads(p.read_text()) if p.exists() else {}
d["enabled"] = False
p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
PY

# Comfy 훅만 비활성화 (삭제하지 않고 이름 변경)
mv /home/flux/ComfyUI/custom_nodes/main_server_vram_arbiter \
   /home/flux/ComfyUI/main_server_vram_arbiter.disabled

# 코드 원복이 필요하면 삭제 대신 백업 후 이동
mkdir -p ~/arbiter_backup_$(date +%m%d_%H%M)
git checkout -- app.py index.html
mv vram_arbiter.py test_arbiter_domain.py ~/arbiter_backup_$(date +%m%d_%H%M)/
```

---

## 7. 진행 내역 (세션별)

### 세션 1 — 3 건 반영 (보고서 기준)

- Comfy 훅을 **차단 대기 + 원본 prompt 반환** 으로 재작성, 매니저에 **홀드 메커니즘** 신설
- UNKNOWN 비율추정 폐기 → resident 정리 우선, `_learn()` max 고수정 → 70/30 완만 반영
- GPU 별 부족량 도입 (당시 GPU 용량 비례 배분 포함)

### 세션 2 — 진단 (코드 미변경)

이전 세션이 "정적 검증까지 마쳤다"고 보고한 부분 중 **도메인화가 선언만 되고 배선되지 않았음**을
확인했습니다.

| 항목 | 상태 |
|---|---|
| `ServiceState.home_gpu / multi_gpu / managed_evict` | 어디서도 대입 안 됨 → 항상 `""/False/True` |
| `GpuPool.evict_threshold_percent` | 파싱·저장만, `admit()`/`_evict_locked()` 미참조 |
| `self.gpu_services` registry | `home_gpu` 항상 `""` → 매번 빈 dict |
| `_unload()` 실패 경로 | `return 0` → 호출부 `.items()` AttributeError |
| `_log_event("evict", gpu=...)` | 존재하지 않는 kwarg → 성공 경로 TypeError |

### 세션 3 — 도메인 단일화 + threshold (요청 크기 %) 도입

`_deficit(pool)`, `_evict_gate()`, `_resolve_uuid()`, `_detect_comfy/_llama` 배선,
`_unload` 반환형 통일, 크로스 GPU 참조 제거.

### 세션 4 (현재) — A~D 완성 + 정책 확정 + 정적 검증

- **A** `_gated_forward` 전 GPU 합산 baseline → 소속 GPU 1장
- **B** `_service_payload` 도메인 필드 (인스턴스 메서드 전환)
- **C** `app.py` `llama_devices` 훅 — **기동 정보 1차 소스** (요청에 따라 반드시 반영)
- **D** `index.html` GPU 별 Evict threshold 입력 + OOM 경고 툴팁
- UNKNOWN 정책 확정 (free 충분하면 미리 내리지 않음), 응답 dict 형상 유지 + scalar 추가
- 정적 하네스 작성 및 실행 → **40건 / 실패 0건**

하네스 작성 과정에서 제가 만든 오류 4건 (`va.int`, `world()` 초기상태로 인한 `reuse` 조기반복,
`SpyLock` 과 `Condition._is_owned` 충돌, 오타) 과, 가짜 풀로 실제 UUID 를 해석해 **의미 없는
FAIL** 이 나던 9번 체크를 발견해 수정했습니다. 최종 상태만 유효합니다.

---

## 8. 다음 단계 (권장 순서)

1. `gpus` 설정에 GPU0/GPU1 threshold 값 기록 (예: GPU0 50 / GPU1 40)
2. 매니저 재시작 시점 결정 → 훅 등록 로그 확인
   (`[vram-arbiter] on_prompt 훅 등록 완료 (동기 대기 상한 900초)`)
3. GPU0 단독 llama 재실행 후 → 8188/8189 에서 평소대로 워크플로우
4. `gpu_arbiter_events.json` 에서 확인:
   - `evict` — `GPU{n} N MB 반환 (도메인 ...)`
   - `evict_skipped` — threshold 미만으로 resident 유지
   - `wait` — 같은 도메인 ACTIVE 완료 대기
   - **다른 GPU 의 서비스 unload 이벤트가 섞이면 안 됩니다**
5. 학습된 `learned_peak_mb` 에 서비스 id 별 값이 쌓이는지 확인 (첫 실행 후 공존 판단 정밀화)
