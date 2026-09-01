# GPU VRAM Arbiter — 멀티 GPU 단순화(도메인 독립) 진행 상황

> 작성 시점: 2025-08-29 · 대상: `/home/flux/main_server_new`
> 상태: **코드 반영 완료, 정적 검증 40/40 통과. 런타임(재시작/unload/실제 eviction)은 아직 미진행.**

---

## 1. 목표 (요청)

Arbiter의 멀티 GPU 설계를 단순화. `main_server`가 ComfyUI / llama.cpp를 직접 실행하므로
각 서비스가 어느 GPU/포트에 속하는지 이미 정확히 알고 있다. 그것을 scheduling 기준으로 쓴다.

```
GPU0 domain                     GPU1 domain
  ComfyUI  :8188  <->            ComfyUI  :8189
  llama.cpp :8080                 llama.cpp :8081
```

- GPU0/GPU1 Arbiter가 **둘 다 ON이어도** 서로 완전히 독립. 각 도메인의 request는 **자기 소속
  managed 서비스만** 대상으로 판단/eviction. 상대 GPU의 free/margin/lock/victim을 참조하지 않는다.
- GPU 간 free 합산, 용량 비례 expected 분배, global multi-GPU deficit, cross-GPU victim 최적화 **불필요**.
- ComfyUI는 실질적 multi-GPU를 고려하지 않는다.
- **llama.cpp를 GPU0+GPU1 양쪽에 걸쳐 실행하는 인스턴스는 eviction 관리에서 제외** (공유 요구 시나리오 아님).
- **GPU별 사용자 설정: `evict_threshold_percent`** 추가.
- Arbiter가 인식하는 request/victim은 main_server가 관리하는 지원 대상(ComfyUI, llama.cpp)만.
  기타 CUDA 프로세스는 NVML free에는 반영되지만 제어하지 않는다.

---

## 2. 판단 로직 (확정된 정책)

### 2.1 소속 GPU(`home_gpu`)는 기동 정보로 정한다 — NVML 추정이 아님

| 서비스 | 1차 소스 | 폴백 |
|---|---|---|
| ComfyUI | `comfyui_settings.json > gpu_device` (UUID) | NVML 실측이 정확히 1장일 때만 |
| llama.cpp | `last_run.json > gpuDevices` (UUID, 디스크라 재시작 후에도 유효) | `services["llama"].device`(메모리) → NVML |

- 토큰 표식(UUID / index / `CUDA1` / `cuda:0` / PCI busId)은 `VramArbiter._resolve_uuid()`가
  pool UUID 하나로 통일.
- **NVML은 실제 free/used 읽기와 검증용.** 소속 판단의 1차 소스가 아니다.

### 2.2 `evict_threshold_percent` = **요청 크기 %** (GPU 사용률이 아님)

> 이번 managed request의 **예상 VRAM**이 해당 GPU **총 VRAM**의 몇 %인지.
> 예) 24GB GPU + threshold 50 → 예상 12GB 이상인 요청만 "큰 요청".

판단 순서 (`admit()` ①②③):

```
① free >= expected + safety_margin
     → 그대로 공존, eviction 없음 (threshold·UNKNOWN 무관)
② free 부족
     → expected / total * 100 >= threshold
          → 같은 home_gpu의 RESIDENT managed peer eviction 허용
     → threshold 미만
          → resident eviction 하지 않음 (요청 자체는 OOM 날 수 있음 — 사용자 선택, UI에 명시)
③ threshold = 0 → free 부족 시 항상 eviction (기존 동작)
```

### 2.3 UNKNOWN (예상 VRAM 미학습) 정책 — 확정안

- **free 충분 → UNKNOWN이어도 eviction 없음.** "추정 불가"라는 이유만으로 상주를 미리 내리지 않는다.
- **free 부족 + UNKNOWN → 큰 요청으로 간주 → threshold 충족으로 취급 → 같은 home_gpu의 RESIDENT managed peer eviction.**
- threshold 미만의 **KNOWN** 요청은 free 부족이어도 resident를 강제로 내리지 않는 현재 의미를 유지.

### 2.4 멀티 GPU llama (GPU0+GPU1 걸쳐 실행)

- `multi_gpu=True`, `managed_evict=False`.
- **victim으로도, requester로도 Arbiter를 태우지 않는다** (`admit` → `passthrough`,
  `reason=multi-gpu-instance-excluded`). 어느 도메인 registry에도 들지 않는다.

---

## 3. 적용된 변경

### 3.1 `vram_arbiter.py`

| 항목 | 내용 |
|---|---|
| `_deficit_per_gpu(uuids)` **삭제** | → `_deficit(pool, expected) -> int` (항상 **단일** pool. GPU 간 합산/비례 배분 제거) |
| `_evict_gate(pool, expected, *, force_open)` | 요청 크기 % 게이트. `force_open=True`(UNKNOWN)면 threshold 적용 생략 |
| `_request_percent(pool, expected)` | `expected/total*100`. 용량 미확인 시 `-1.0`(게이트가 0.0으로 닫히지 않게) |
| `admit()` | `home=service.home_gpu`, `pool=self.pools[home]`. lock/refresh/판정 전부 `[home]`. ①②③ 순서. OFF GPU·미지원·multi-GPU·소속 미확정 → `passthrough` |
| `_evict_locked(home_uuid, ...)` | victim 조건 `state.home_gpu == home_uuid and state.managed_evict`. (기존 `any(u in enabled_uuids for u in state.gpu_uuids)` 제거 — 교차 GPU 후보 원천 차단) |
| `_active_blockers(home_uuid, ...)` | 같은 도메인 ACTIVE만 대기 대상 |
| `_unload(service, home_uuid) -> int` | **반환형 Dict→int 통일**. 소속 GPU 1장만 관찰. 실패 시 `return 0` |
| `_resolve_uuid(token)` **신설** | UUID/index/CUDA1/PCI → pool UUID |
| `_detect_comfy` | 소속 = `comfy_device` 단일 강제. `gpu_uuids=[home]`, `multi_gpu=False` |
| `_detect_llama` | 소속 = `llama_devices(port)` 기동 정보. 멤버 2장↑ → `multi_gpu=True`, `managed_evict=False` |
| `_llama_file_size_mb(home_uuid)` | 용량 95% 상한을 **소속 1장**으로 재단 (기존 GPU0+GPU1 합산 기준 삭제) |
| `rebuild_registry()` **신설** | `gpu_services` = `{uuid: [home_gpu==uuid and managed_evict]}`. 테스트 가능하게 분리 |
| `summary()`/`status()` | 필터 `state.home_gpu == uuid`, `evict_threshold_percent`·`managed_services` 노출 |
| `_service_payload` | `@staticmethod` → 인스턴스 메서드. `home_gpu/home_index/multi_gpu/managed_evict/domain_note` 추가 |
| `_gated_forward` | baseline/delta를 **전 GPU 합산 → 소속 GPU 1장** (`gate_home`) |
| `/evict/{id}` | home 기준, `_unload` int 반환에 맞춰 응답 정리 |

### 3.2 `app.py`

- `_arbiter_llama_devices(port)` **신설** + `configure(llama_devices=...)`.
  - 1차 `last_run.json > gpuDevices`(포트 일치 확인) → 2차 `services["llama"].device` → `[]`.

### 3.3 `index.html`

- GPU별 `Evict __ %` 입력 (`setArbiterThreshold` → `POST /api/gpu-arbiter/settings`).
- `arbiterThresholdHelp()` 툴팁: 판단 4단계 + **"3번에서 요청은 OOM 날 수 있음, 의도된 동작"**
  + "GPU 사용률이 아닙니다" + "GPU별 독립" 명시.
- 멀티 GPU 서비스 `제외` 배지 (`svc.domain_note` 노출).
- estimate 소스 라벨 `첫 실행 · free 충분하면 eviction 없음`으로 정정.

### 3.4 `test_arbiter_domain.py` (신규, 정적 하네스)

- NVML/네트워크/프로세스 **미접촉**. `read_gpu_snapshot`/`read_process_map`/`_unload`/`_log_event` 스텁·no-op.
- `_PoolLock.__enter__`만 관측해 **어느 GPU lock이 실제로 잡혔는지** 확인(제품 코드 왜곡 없음).
- `app.py`의 `_arbiter_llama_devices`를 **AST로 추출해 실제 코드 그대로** 실행 검증.

---

## 4. 검증 결과

### 4.1 컴파일
```
.venv/bin/python -m py_compile vram_arbiter.py app.py   → OK (A~D 반영 후)
.venv/bin/python -m py_compile ... test_arbiter_domain.py → OK
```

### 4.2 정적 domain 독립성 테스트 — **총 40건 중 실패 0건, EXIT=0**

| 요청 검증 항목 | 결과 | 실제 출력 |
|---|---|---|
| **GPU0 req → GPU1 victim 0건** | PASS | `unloaded=[('llama-8080', GPU0)]`, `acquired=[GPU0]` (GPU1 lock 미획득) |
| **GPU1 req → GPU0 victim 0건** | PASS | `unloaded=[('llama-8081', GPU1)]`, `acquired=[GPU1]` |
| **multi-GPU llama → 양쪽 제외** | PASS ×5 | requester `reason=multi-gpu-instance-excluded`, `acquired=[]`; victim 측 양 도메인 후보 제외 |
| **threshold = request 크기 %** | PASS ×4 | `pct=32.6%`→`load-no-evict`+`home_deficit_mb=7048` / `pct=52.9%`→`evicted+load` / `threshold=0`→항상 |
| **free 충분 → threshold 이상이어도 eviction 없음** | PASS | `pct=52.9%`인데 free 20GB → `action=load`, peer `RESIDENT` 유지 |
| UNKNOWN + free 충분 → eviction 없음 | PASS | `action=load`, `unloaded=[]` |
| UNKNOWN + free 부족 → 큰 요청 간주 | PASS | `pct=16.7% < 50%`여도 `evicted+load` |
| 응답 형상 호환 | PASS | `free_mb`/`deficit_mb` = dict, `home_free_mb`/`home_deficit_mb` scalar 병존 |
| `_resolve_uuid` | PASS ×6 | UUID/`1`/`CUDA1`/`cuda:0`/PCI 해석, `CUDA9`·`""`→`""` |
| C: `_arbiter_llama_devices` | PASS ×7 | last_run UUID 2개 반환, 포트 불일치→`[]`, 디스크 우선, multi-GPU 판정 |
| `_active_blockers` 도메인 한정 | PASS ×2 | GPU0은 GPU0 ACTIVE만, GPU1은 GPU1 ACTIVE만 |

---

## 5. 현재 머신 실제 상태 (참고)

- `nvidia-smi`: GPU0 = CMP 170HX 64GB (`GPU-6cd658de-...`), GPU1 = RTX 3090 24GB (`GPU-309d93a2-...`).
- **실행 중** `llama-server :8080`은 `--device CUDA1,CUDA0 --split-mode layer --tensor-split 7,1` →
  `last_run.json gpuDevices` 2개 → **`multi_gpu=True`로 자동 관리 제외**. (지시대로 재시작하지 않음.)
- ComfyUI: `main`→GPU0, `gpu1`→GPU1 (UUID 설정). 포트 8188/8189.

---

## 6. 아직 안 한 것 / 주의

1. **런타임 테스트 미진행** — llama/Comfy 재시작, 모델 load/unload, 실제 eviction. (요청대로 보류)
2. `vram_arbiter_settings.json`에 `evict_threshold_percent` 키가 **없으면 기본 0** = "부족 시 항상 eviction"
   (기존 동작과 동일). GPU1에 40을 넣으려면 UI 입력 또는
   `POST /api/gpu-arbiter/settings {"gpus":{"GPU-309d93a2-...":{"evict_threshold_percent":40}}}`.
3. **`--reload` 주의** — main_server가 `--reload`로 떠 있으면 `app.py` 편집 시 **매니저 프로세스는
   재시작**될 수 있음. ComfyUI/llama는 `start_new_session`으로 분리돼 살아남지만, 원치 않는 시점의
   재시작을 피하려면 확인 후 재시작.
4. **설정 파일 구버전 키** — `llama_inject_missing_model` 등. 로더가 미인식 키는 무시해 동작에 지장 없음.
5. `/arbiter/llama/{path}` 프록시는 이번 작업에서 **다중 포트 확장 안 함** (요청대로). GPU별 domain은
   기동 정보로만 구성. Comfy 훅 경로에는 영향 없음.

---

## 7. 롤백 (glob 삭제 대신 백업/이동)

```bash
cd /home/flux/main_server_new
# 기능만 끄기 (권장 — 코드 유지)
python3 - <<'PY'
import json,pathlib
p=pathlib.Path("vram_arbiter_settings.json")
d=json.loads(p.read_text()) if p.exists() else {}
d["enabled"]=False
p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
PY
# Comfy 훅만 비활성화 (이름 변경, 삭제 아님)
mv /home/flux/ComfyUI/custom_nodes/main_server_vram_arbiter \
   /home/flux/ComfyUI/main_server_vram_arbiter.disabled
# 코드 원복
mkdir -p ~/arbiter_backup_$(date +%m%d_%H%M)
git checkout -- app.py index.html
mv vram_arbiter.py ~/arbiter_backup_$(date +%m%d_%H%M)/
rm -f test_arbiter_domain.py
```

---

## 8. 재현 명령

```bash
cd /home/flux/main_server_new
.venv/bin/python -m py_compile vram_arbiter.py app.py && echo "py_compile OK"
.venv/bin/python test_arbiter_domain.py   # 40건, EXIT=0 기대
```
