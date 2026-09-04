# 디스크 용량 정리 대상 리포트

> 작성일: 2026-09-04 · 기준: `df -h` / `du -xh --max-depth=2` / `find` / md5sum 비교
> 경로 표기는 이 기기 마운트 기준. 삭제는 각 항목 확인 후 진행할 것.

## ⚠️ 스캔 범위 안내

- **`/mnt/hdd14tb`** — 로컬 WDC 14TB (WUH721414ALE6L4, NTFS) · 13T 중 **11T 사용 (85%)**
- **`/mnt/nas`** — 192.168.1.119 NAS homes 볼륨 (CIFS) · 13T 중 **12T 사용 (96%)**
- **`/mnt/hdd12tb` (물리 12TB HDD)** — `fstab`에 UUID `4ED28987D28973CD`로 등록되어 있으나 **현재 디스크 미연결**. 연결 후 재스캔 필요.

---

## 0. 시스템 SSD (`/`) — 정리 실행 완료 ✅

사용량 **184G → 138G**, 여유 **32G → 79G** (+47G 확보)

| 삭제 대상 | 확보량 |
|---|---|
| `/var/crash/_usr_bin_python3.12.1000.crash` (앞 200KB는 `main_server_new/logs/python3.12_crash_head_20260904.txt`에 백업) | 8.4G |
| `~/Documents/Codex/2026-08-27/…qwen3-8/work` (미사용 venv 3종 + vllm 빌드) | 28G |
| HF 캐시 blobs (`MiniMax-H3-Acc-LoRAs` 3.1G + `Qwen3.8-27B-DFlash2-W4A16` 1.2G) | 4.3G |
| `~/.cache/google-chrome-headless` (헤드리스 세션 잔여물) | 1.3G |
| `~/.triton` + `~/.cache/torch_extensions` + `~/.local/share/uv` + `.cache/{pnpm,selenium,thumbnails}` | ~1.4G |
| `~/llama.cpp-qwen38-mtp-backup-20260829` (원본과 동일 커밋 확인) | 1.0G |
| `/opt/llama/.src` 구버전 소스빌드 b10628 / b10665 / b10675 | 2.3G |
| apt 캐시·리스트, journal vacuum(100M), `/tmp` 잔여 빌드물, 설치완료 .deb 3종 | ~2.3G |

**의도적 보존 (사용자 지정):** `~/ComfyUI/temp` 3.8G · `~/ComfyUI/output` 3.4G · `~/.cache/codex-runtimes` 1.8G · `~/.unsloth` 9.2G · `~/.codex` 1.3G

**유보 (판단 대기):** `~/.cache/google-chrome` 1.5G (크롬 실행 중) · `~/llama.cpp-qwen38-mtp-opt` 1.6G (동일 커밋, `qwen38-mtp-ple-pp` 브랜치) · `~/llama.cpp-qwen38-mtp-qsa-experiment` 1.1G (미커밋 21개 파일) · 현역 venv 일체 (`tabbyAPI-exl3` 6.1G — main.py 실행 중, `qwen38-vllm-upstream` 8.0G, `qwen38-27b-cmp170hx64` 8.1G, `ComfyUI` 7.8G)

---

## 1. 14TB HDD (`/mnt/hdd14tb`)

### 🔴 1단계 — 폐기물 / 재다운로드 가능 (~210G)

| 경로 | 크기 | 비고 |
|---|---|---|
| `/mnt/hdd14tb/miyoomini.img` | **119G** | 2022년 VMware 디스크 이미지. `vmware/` 디렉터리는 비어 있음 |
| `/mnt/hdd14tb/Qwen3.5-122B-A10B-UD-Q4_K_XL-00002-of-00003.gguf` | 49.7G | 3분권 합계 **68G** (00001=10.9M, 00003=18.6G 포함). 구 모델, 현역(Qwen3.8-Flash-Next)에 대체. NVMe 모델볼륨에도 없음 |
| `/mnt/hdd14tb/MMVCServerSIO_win_onnxgpu-cuda_v.1.5.3.6a.zip` | 3.2G | 압축본. 해체본 `MMVCServerSIO` 폴더가 이미 존재 |
| `/mnt/hdd14tb/MMVCServerSIO_win_onnxgpu-cuda_v.1.5.3.8a.zip` | 3.2G | 〃 |
| `/mnt/hdd14tb/RVC-beta-v2-0618.7z` | 4.8G | 압축본. 해체본 `voice/RVC-beta-v2-0618` (252G) 존재 |
| `/mnt/hdd14tb/ComfyUI_windows_portable.7z` | 2.6G | Windows 시절 포트러블. 현행 ComfyUI는 NVMe 볼륨 구동 |
| Windows 설치 잔여물 | ~5G | `RipX_630.exe` 1.6G · `S5___V26.zip` 84M · `hitomi_downloader_GUI.zip` 85M · `guitar.zip` 19M · Adobe AE/MediaEncoder/Premiere RePack torrent 3개 · `HoaxEliminator7.15.zip` · `MalwareZero.zip` · `MSIAfterburnerSetup.zip` · `UVR_v5.5.1_setup.exe` · `VoicemeeterSetup.exe` · `InstallGoldWave676.exe` · `I3GSvcManager.exe` · `VBCABLE_Driver_Pack43`(+zip) · LUT zip 3종 (`33 Cube LUTs E-E` 외) |
| `/mnt/hdd14tb/System Volume Information` | 14M | Windows 시스템 디렉터리($MRCBT) |
| 최상위 흩어짐 파일 | ~270M | `01 (enhanced).wav` 78M · `VID_20160409_152445.mp4` · `hubert_base.pt` 190M · `ko_KR.json` |
| 빈 디렉터리 | ~0 | `$RECYCLE.BIN` · `BaiduNetdiskDownload` · `NPKI` · `msdownld.tmp` 등 (삭제 전 최종 확인) |

### 🟠 2단계 — 판단 필요 (크기 순)

| 경로 | 크기 | 내용 / 판단 근거 |
|---|---|---|
| **`temp/`** | **6.0T** | DFL 계열 작업물: `DeepFaceLab_NVIDIA_RTX3000_series` **5.6T** (약 20만 파일) · `DeepFaceLab` 187G · `df_workspace` 131G · `faceswap` 68G. 2023~24년 DeepFaceLab/DL 프로젝트 — 접었다면 **이 드라이브 최대 정리 대상** |
| **`voice/`** | **1.3T** | `train_utili` 627G · `RVC-beta-v2-0618` 252G · `MMVC_old` 157G · `Mangio-RVC-v23.7.0_dev` 122G · `voice_data` 97G. RVC/MMVC 학습+출력물. 학습 데이터셋은 재생산 가능하나 **학습된 .pth 모델은 비복제성** → 선별 필요 |
| **`sub_drive/`** | **1.1T** | 전부 모델 컬렉션: `models/Stable-diffusion` 573G (safetensors 134 + ckpt 4) · `models/ckpt` 483G (ckpt 122) · VAE/업스케일러 소규모. 현행 ComfyUI(NVMe) 모델과 **중복 미검증** — 이름+크기(+해시) 크로스체크 후 정리 |
| `replacer/` | 379G | SD-webUI 클론 (safetensors 154 + pth 8 + pt 5) |
| `hitomi_downloader_GUI/` | 368G | twitter 112G · pixiv 73G · **`hitomi_downloaded_youtube_old` 59G** (신규 `youtube` 21G와 패턴 겹침 → _old 중복 추정, 검증 필요) · pornhub 계열 ~20G |
| `Lora+hyper/` | 317G | `complte_Lora` 271G — 2023년식 구형 .pt LoRA. 현 파이프라인 사용 여부 확인 |
| `stable-diffusion-webui/` | 214G | 웹UI + models 208G (safetensors 26 + ckpt 26) |
| `github/` | 168G | 3번째 SD-webUI 클론 (safetensors 18 + ckpt 15 + pt 8 + pth 7) |
| `text/` | 161G | 거의 전부 `one-click-installers-main` 160G |
| `musubi-tuner/` | 142G | 비디오 생성 모델 프로젝트 (출력/데이터셋) |
| `DF_LIVE/` | 86G | `DeepFaceLive_NVIDIA` |
| `ai-toolkit/` | 78G | `output` 57G 포함. dreambooth/LoRA 학습 결과 |
| `24_01_24_backup/` | 70G | **R-Studio류 복구 덤프** (2024-01): `손실된 파일 (1441711)` 47G · `(1463344)` 9.1G · `(1460846)` 7.4G · `(1442120)` 5.4G · `#1 (NTFS)1.82 TB` ~1G. **복구 데이터일 수 있어 개별 확인 후** |
| `huggingface_cache/` | 66G | `models--Lightricks--LTX-2.3` + `models--Lightricks--gemma-3-12b-it-qat-q4_0-unquantized`. **HF 재다운로드 가능 → 폐기 무방** |
| `hitomi_2/` | 60G | `missav` 23G + pornhub 계열 |
| `llama.cpp/` | 50G | Windows 시절 구 소스+빌드. `/opt/llama`, `~/llama.cpp*` 계열과 별개 구판 |
| `voice_backup/` + `voice_backup_2/` | 1.1G + 612M | 두 폴더 간 **MD5 동일 파일 3개 (총 173M)**: `RALO_V1.pth` · `merged_gyun+vRize660e.pth` · `peymon_v1_e450_s7650.pth`. 나머지는 각기 다른 보컬 데이터 |

### ✅ 검증 결과 메모

- `liyuu_merge_test.pt` vs `liyuu_merge_test2.pt` — 크기 동일(151,268,049B)이나 **MD5 상이 → 중복 아님** (모두 유지)

### ⚠️ SD-webUI 4클론 비교 — 미실행

`stable-diffusion-webui`(214G) · `github/…`(161G) · `replacer/…`(379G) · `replacer2/…`(9.8G)

서로 다른 모델 컬렉션(safetensors 26/18/154/—개)이라 단순 복사가 아님. 모델 파일 단위 이름+크기(+해시) 크로스체크를 수행해야 실제 교차 중복량 파악 가능.

---

## 2. NAS (`/mnt/nas`, 192.168.1.119/homes)

| 경로 | 크기 | 비고 |
|---|---|---|
| `/mnt/nas/Flux/Photos` | **1.8T** | 현역 계정 · 899,082개 파일 (1,762.7 GiB) · **유지** |
| `/mnt/nas/Flus/Photos` | **1.3T** | **구 계정** (2022~24, Nikon NEF 중심 · 26,624개 · 1,272.7 GiB) |
| `/mnt/nas/kjh1004net/Photos` | 117G | **타 사용자 계정** — 삭제 전 해당 사용자 확인 필수 |
| `/mnt/nas/admin` · `jihye` · `m3` · `m3_test` · `simura` | 0 | 전부 빈 공유 폴더 |
| `/mnt/nas/Flux/Drive` · `/mnt/nas/Flus/Drive` | 0 | 빈 디렉터리 |
| `/mnt/nas/Flux/.local`(89M) · `.cache`(28M) · `.Maildir` · sieve · `.DS_Store` | ~120M | 미미 |

### Flus vs Flux 미러 검증 결과 — **미러가 아님**

| 비교 기준 | 동일 파일 수 | 크기 |
|---|---|---|
| 전체 경로 + 크기 | 4개 | ~0 |
| 파일명(basename) + 크기 | 6,980개 | 83.7 GiB |

→ `Flus/Photos`의 **약 1.19T는 고유 콘텐츠**. 단순 미러 삭제가 아니라 2022~24년 Nikon RAW 라이브러리의 보존 필요 여부 판단 후 결정.

---

## 📊 정리 여력 요약

| 구분 | 예상 확보량 |
|---|---|
| 14TB · 1단계 (폐기물) | ~210G |
| 14TB · 2단계 고신뢰 (`temp` DFL 6T, `huggingface_cache` 66G, `musubi-tuner` 142G) | ~6.3T |
| 14TB · 2단계 판단형 (`voice` 1.3T, `sub_drive` 1.1T, webUI×4 ~980G, hitomi 계열 430G, `Lora+hyper` 317G, 복구백업 70G, `ai-toolkit` 78G, `text` 161G) | ~4.5T (중복 검증 시 추가) |
| NAS · `Flus/Photos` 1.3T + `kjh1004net` 117G (권한/의미 확인 필요) | ~1.4T |

**14TB만 현실적으로 전 정리하면 사용량 11T → 0.3~1T 수준** 축소 가능.

## ⚠️ 보존 권장 (삭제 금지)

- `24_01_24_backup/` — 복구로 되찾은 데이터. 삭제 전 개별 파일 확인 필수
- `voice/` 내 학습된 `.pth` 모델, `voice_backup*` — 비복제 결과물
- NAS `Flux/Photos` (현역 라이브러리), `kjh1004net/Photos` (타 사용자 자산)
- SSD 현역: `tabbyAPI-exl3`(main.py 실행 중), `main_server_new`(app.py 실행 중), 각 라우트 venv

## 📋 다음 액션 옵션

1. **14TB 1단계만 실행** → ~210G 즉시 확보
2. **`temp/` (DFL) 포함 실행** → +6.0T
3. **4개 웹UI 모델 크로스체크 수행** (수 시간 소요, 완료 후에만 모델 중복 삭제)
4. **물리 12TB HDD 연결 → `/mnt/hdd12tb` 재스캔**
