# 디스크 정리 / 중복·폐기물 분석 리포트

- 작성일: 2026-09-04
- 대상: 메인 SSD(`/`), 로컬 14TB HDD(`/mnt/hdd14tb`), NAS 홈 볼륨(`/mnt/nas`)
- 조사 방법: `du` 트리 스캔(최대 깊이 2~3), 2GB+ 대용량 파일 전수 추출, 디렉터리 간 파일명+크기 정렬 비교, MD5 샘플 검증, 실행 중 프로세스 경로 교차 확인(`ps`/`/proc`)
- 주의: 본 리포트의 **14TB·NAS 항목은 아직 아무것도 삭제되지 않음** (리스트업만). 삭제 실행은 항목별 승인 후.

---

## 0. 상태 개요 (작성 시점)

| 볼륨 | 용량 | 사용 | 여유 | 사용률 | 비고 |
|---|---|---|---|---|---|
| `/` (sda2, Samsung 860 EVO 250G ext4) | 228G | 139G | 77G | 65% | 오전 정리 전 184G(86%) → 47G 확보 완료 |
| `/mnt/hdd14tb` (sdb3, WDC WUH721414ALE6L4 NTFS) | 13T | 11T | 2.1T | 85% | 이번 분석 대상 |
| `/mnt/nas` (CIFS //192.168.1.119/homes) | 13T | 12T | 534G | 96% | "12TB"로 호출되던 볼륨 |
| `/mnt/hdd12tb` (fstab 등록, UUID `4ED28987D28973CD`) | — | — | — | — | **물리 디스크 미연결** (lsblk/blkid 미검출, 마운트 안 됨) |

> `fstab`에 12TB HDD 항목이 존재하나 현재 시스템에 디스크가 붙어 있지 않아 스캔 불가. 연결 후 재분석 필요.

---

## 1. 메인 SSD(`/`) — 정리 완료 내역 (참고 기록)

2026-09-04 오전 실행. **사용자 지정 보존**(`~/ComfyUI/temp` `~/ComfyUI/output` `~/.cache/codex-runtimes` `~/.unsloth` `~/.codex`) 제외 후 진행.

| 삭제 항목 | 확보량 | 근거 |
|---|---|---|
| `/var/crash/_usr_bin_python3.12.1000.crash` (앞 200KB를 `main_server_new/logs/python3.12_crash_head_20260904.txt`로 백업 후 삭제) | 8.4G | python3.12 단일 크래시 덤프 |
| `~/Documents/Codex/2026-08-27/…/work` (미사용 torch/nvidia venv 3종 + vllm 빌드) | 28G | 실행 중 프로세스 참조 없음 확인 |
| HF 캐시 blobs (`MiniMax-H3-Acc-LoRAs` 3.1G + `DFlash2-W4A16` 1.2G) | 4.3G | ComfyUI models에서 symlink 미사용 확인 |
| `~/.cache/google-chrome-headless` (scoped_dir 잔여) | 1.3G | 헤드리스 세션 잔여물 |
| `~/.triton` `~/.cache/torch_extensions` `~/.local/share/uv` 등 재빌드 캐시 | ~1.4G | 재생성 가능 |
| `~/llama.cpp-qwen38-mtp-backup-20260829` | 1.0G | 원본과 HEAD 동일(99acf2324)+clean 확인 |
| `/opt/llama/.src` 구버전 b10628/b10665/b10675 | 2.3G | 설치본은 별도 유지 |
| apt 캐시/리스트, journal→100M vacuum, `/tmp` 빌드 잔여, 설치완료 .deb 3종 | ~2.3G | 정상 캐시 정리 |

- 결과: 184G→138G 사용 (현재 139G, 세션 중 증가분 소폭) — **+47G 확보**
- 판단 보류로 의도적으로 유지한 항목:
  - `~/.cache/google-chrome` 1.5G (Chrome 실행 중)
  - `~/llama.cpp-qwen38-mtp-opt` 1.6G (qwen38-mtp-ple-pp 브랜치 작업물)
  - `~/llama.cpp-qwen38-mtp-qsa-experiment` 1.1G (미커밋 변경 21개 파일)
  - 현역 venv 전면: `~/tabbyAPI-exl3`(6.1G, main.py 실행 중), `~/qwen38-vllm-upstream`(8.0G), `~/qwen38-27b-cmp170hx64`(8.1G), `~/ComfyUI/venv`(7.8G)

---

## 2. 14TB HDD (`/mnt/hdd14tb`) — 총 사용 11T

### 2.1 🔴 1단계: 폐기물·재다운로드 가능 (~210G, 삭제 위험 낮음)

| 경로 | 크기 | 근거/비고 |
|---|---|---|
| `miyoomini.img` | 119G | 2022-05 VMware 디스크 이미지 단일 파일. 인접 `vmware/` 폴더는 비어 있음 |
| `Qwen3.5-122B-A10B-UD-Q4_K_XL-{00001,00002,00003}-of-00003.gguf` | ~64G | 2026-02 다운로드 분할 GGUF 3개. 현역 모델(Qwen3.8-Flash-Next)로 교체됨, NVMe 모델볼륨에도 사본 없음 |
| `voice/RVC-beta-v2-0618` 외 최상위 `RVC-beta-v2-0618.7z` | 4.5G | 압축 원본과 해체본이 각각 존재 → 압축분 폐기 가능 |
| `MMVCServerSIO_win_onnxgpu-cuda_v.1.5.3.6a.zip` | 3.0G | 해체 폴더(`MMVCServerSIO`, `MMVCServerSIO_17A`)가 이미 존재 |
| `MMVCServerSIO_win_onnxgpu-cuda_v.1.5.3.8a.zip` | 3.0G | 〃 |
| `ComfyUI_windows_portable.7z` | 2.4G | 구 Windows 포터블. 현역 ComfyUI는 NVMe 볼륨 |
| Windows 설치 파일 잔여물 | ~6G | `RipX_630.exe`(1.5G), `MMVCServerSIO_...zip` 외 exe/zip: GoldWave, I3GSvcManager, UVR_setup, Voicemeeter, HoaxEliminator, MalwareZero, MSIAfterburnerSetup.zip, MalwareZero.zip, VBCABLE_Driver_Pack43(+zip), S5___V26.zip, guitar.zip, hitomi_downloader_GUI.zip, e-e/vlt LUT zip, Adobe RePack torrent 3개 |
| 최상위 흩어진 미디어 | ~260M | `01 (enhanced).wav`(74M), `hubert_base.pt`(181M), `VID_20160409_152445.mp4`, `ko_KR.json`, `O1CN01...jpg` |
| `System Volume Information` | 14M | Windows 시스템 잔여($MRCBT) |
| 빈/불필요 디렉터리: `$RECYCLE.BIN` `BaiduNetdiskDownload` `NPKI` `msdownld.tmp` | ~0 | 삭제 전 `ls`로 실제로 비어 있음 재확인 필요 |

### 2.2 🟠 2단계: 대용량 판단 항목 (크기 내림차순)

크기 단위는 `du` 기준(GiB), 깊이 2 전체 스캔 결과 값.

| 경로 | 크기 | 내용 | 삭제 전 확인할 점 |
|---|---|---|---|
| **`temp/DeepFaceLab_NVIDIA_RTX3000_series`** | **5.6T** | DeepFaceLab 학습 워크스페이스(약 20만 파일) | DFL 프로젝트 종료 여부. 단일 최대 정리 대상 |
| `temp/DeepFaceLab` | 187G | 구버전 설치+프로젝트 | 〃 |
| `temp/df_workspace` | 131G | DFL 세션 작업공간 | 〃 |
| `temp/faceswap` | 68G | facefusion 실험물 | 〃 |
| `temp/` 소계 | **6.0T** | 위 항목 전체 + 기타 설치 잔여물 | |
| **`voice/`** 소계 | **1.3T** | `train_utili` 627G, `RVC-beta-v2-0618`(해체본) 252G, `MMVC_old` 157G, `Mangio-RVC-v23.7.0_dev` 122G, `voice_data` 97G | RVC/MMVC 학습 산출물. **학습 결과 보컬 모델은 복원 불가** → 학습 출력(`*.pth`, `*.index`)과 임시 데이터셋 분리 선별 |
| **`sub_drive/`** | **1.1T** | SD 모델 아카이브: `models/Stable-diffusion` 573G(safetensors 134+ckpt 4), `models/ckpt` 483G(ckpt 122), VAE 3G 외 | 현재 ComfyUI(NVMe `ComfyUI-models` 164G) 모델 라이브러리와 이름+크기 크로스체크 후 중복 제거 |
| **`replacer/stable-diffusion-webui`** | **379G** | webui 클론(safetensors 154, pth 8, pt 5, `cache/cache2.json` 등 잔여) | 구 실험 잔여로 추정. 삭제 후보 1순위 |
| **`hitomi_downloader_GUI/`** | **368G** | twitter 112G, pixiv 73G, **youtube_old 59G**, youtube 21G, insta 23G, pornhub 13G | `youtube_old` vs `youtube` 중복 추정(미검증, 이름 패턴상). 트위터/픽시브 소스는 원본 사이트 존속 여부 확인 |
| **`Lora+hyper/`** | **317G** | `complte_Lora` 271G (2023년식 .pt LoRA 일체) | 현행 safetensors 파이프라인 전환 후 미사용이면 전체 후보 |
| **`stable-diffusion-webui/`** | **214G** | models 208G(safetensors 26 + ckpt 26) | 4개 webui 벌 비교 대상 (2.3절) |
| **`github/`** | **168G** | 내부 `stable-diffusion-webui` 161G(models: safetensors 18, ckpt 15, pt 8, pth 7), `sd-scripts`, `kohya_ss`, `train_network.ps1` | 학습용 클론. 현재 `~/kohya*`/`musubi` 계열과 기능 중복 |
| **`text/`** | **161G** | `one-click-installers-main` 160G | AI 앱 원클릭 인스톨러 모음(다운로드 가능 자료) → 후보 |
| **`musubi-tuner/`** | **142G** | 비디오 튜너 프로젝트 작업물 | 현재 학습 진행 중 여부 확인 |
| **`DF_LIVE/DeepFaceLive_NVIDIA`** | **86G** | DeepFaceLive 라이브 스왑판 | 진행 여부 기준 — `temp/`의 DFL 계열과 동시 정리 대상 |
| **`ai-toolkit/`** (output 포함) | **78G** | `output` 57G = LoRA/dreambooth 학습 결과 | 결과 LoRA를 어디 쓰는가 확인 |
| **`24_01_24_backup/`** | **70G** | 복구 소프트웨어 덤프(2024-01): `손실된 파일 (1441711)` 47G, `(1463344)` 9.1G, `(1460846)` 7.4G, `(1442120)` 5.4G, `#1 (NTFS)1.82 TB(1)` 902M 등 | 복구 성공 데이터일 가능성 → 개별 파일 내용 확인 필수. 삭제 신중 |
| **`huggingface_cache/hub`** | **66G** | `models--Lightricks--LTX-2.3` + `gemma-3-12b-it-qat-q4_0` | HF 재다운로드 가능 → 안전 후보 |
| **`hitomi_2/`** | **60G** | missav 23G, pornhub 계열 약 10G | `hitomi_downloader_GUI` 이전 다운로드 트리의 잔여분으로 추정 |
| `llama.cpp/` | 50G | 2023-era 소스 빌드 | `/opt/llama`(설치본 6.4G)·`~/llama.cpp*` 계열과 중복. 삭제 후보 |

### 2.3 검증 결과 — 중복 의심 항목 판정

| 검증 항목 | 결과 |
|---|---|
| `liyuu_merge_test.pt` vs `liyuu_merge_test2.pt` | 크기 동일(144M)이나 **MD5 상이 → 중복 아님. 모두 유지** |
| `voice_backup/`(1.1G) vs `voice_backup_2/`(612M) | 파일명 충돌 3개 모두 **MD5 일치 → 완전 복제(173M)**: `RALO_V1.pth`, `merged_gyun+vRize660e.pth`, `peymon_v1_e450_s7650.pth`. 나머지 파일은 서로 다름(개별 보컬). 한 벌로 병합 가능 |
| SD-webui 4벌 (`root` 214G / `github/…` 161G / `replacer/…` 379G / `replacer2/…` 9.8G) | 벌마다 모델 구성 상이(st 26 / 18 / 154 / 0). **단순 복제 아님 → 모델 단위 이름+크기+해시 크로스체크 없이는 아무것도 삭제 금지** |
| 압축파일 vs 해체폴더 (RVC 7z, MMVC zip×2) | 해체 폴더가 이미 존재 확인 → 압축분만 폐기 가능 |
| Qwen3.5-122B GGUF 3분할 | NVMe 모델볼륨(`/mnt/main-server-models`) 대비 사본 없음. 완전 로컬 원본 → 삭제 시 영구 소실 (HF 재다운 가능한 공개모델인지 확인 후) |
| 14TB 내 2GB+ 대형파일 전수 추출 | 전수 목록 생성 완료: `/tmp/big_14tb.tsv` |

### 2.4 예상 확보량 (14TB)

| 시나리오 | 확보 예상 |
|---|---|
| 1단계만 (완전 안전) | **~210G** |
| 1단계 + DFL 계열(`temp/` 일괄) | **~6.2T** |
| 전체 고신뢰 후보(2단계 중 temp/voice 일부/manual 설치물/캐시/구 모델선) | **~7~8T** |
| webui 4벌 모델까지 크로스체크 해체 | 최대 **~10T** (현재 비어 있는 2T 포함 ~12T 여유 확보) |

---

## 3. NAS 홈 볼륨 (`/mnt/nas`) — 총 사용 12T/13T (96%)

| 경로 | 크기 | 파일 수 | 비고 |
|---|---|---|---|
| `Flux/Photos` | **1.8T** | 899,082 | 현역 계정. 유지 |
| `Flus/Photos` | **1.3T** | 26,624 | 구 계정(Nikon NEF/DSC_ 중심, 2022~24) |
| `kjh1004net/Photos` | 117G | — | 타 사용자 계정 — 본인 확인 필수 |
| `Flux/.local`+`.cache`+잡파일 | ~120M | — | .DS_Store 외에는 무해한 잔여물 |
| `admin/` `jihye/` `m3/` `m3_test/` `simura/` | 0 | — | 완전 비어 있음 |
| `Flux/Drive` `Flus/Drive` | 0 | — | 빈 Drive 폴더 |

### 검증 결과 — Flus ⇄ Flux는 미러 아님
- 전체 경로+크기 완전 일치: **4개 파일뿐**
- basename+크기 일치: **6,980개 / 83.7G**만 중첩
- 결론: `Flus/Photos`의 약 **1.19T는 고유 콘텐츠**. "Flux로 이사하고 남은 빈 틈"이 아니라 별도 프로젝트성 아카이브
- 따라서 **무조건 삭제 대상 아님**. 83.7G 완전 복제분은 이동·병합 후보로 분리 가능

### 예상 확보량
- `Flus/Photos` 정리(Flux 병합 또는 아카이빙): 최대 **1.3T** (NAS 여유 534G → 약 1.8T)
- 빈 계정 폴더 제거: 용량 효과 0 (정비 목적). 루트 `.DS_Store` 및 잡파일 제거는 무해

---

## 4. 물리 12TB HDD (`/mnt/hdd12tb`)

- `fstab`에 마운트 설정은 존재하나 **하드디스크 미연결 상태**
- lsblk에서 14TB HDD(WDC) 외에 12TB급 물리 디스크 자체가 미검출. UUID `4ED28987D28973CD`도 blkid 미검출
- **연결 후 재스캔 필요** — 현재 본 리포트에서 빠진 유일한 볼륨

---

## 5. 권장 실행 순서

1. ✅ (완료) 메인 SSD 47G 정리
2. 14TB 1단계 (~210G) 즉시 실행 가능 — 승인 시 절차화
3. 14TB `temp/`(DFL) 처리 의사 결정 — 6T, 단일 최대
4. 4개 webui 모델 크로스체크 스크립트 실행 (백그라운드 잡 권장, 소요 1~3시간)
5. `voice/` 보컬 모델 .pth/.index 백업 정책 수립 후 학습 캐시성 디렉터리 정리
6. NAS: Flus 병합 검토 (복제 83.7G 제거부터), 빈 공유 폴더 정리
7. 12TB 물리 디스크 연결 → 동일 pipeline 재실행

---

## 부록: 원시 데이터 산출물 위치

- 14TB depth-2 du 전체: `/tmp/du_hdd14tb.txt` (420행)
- NAS depth-2 du: `/tmp/du_nas.txt`
- 14TB 2GB+ 파일 전체 목록: `/tmp/big_14tb.tsv`
- NAS 사진 트리 목록: `/tmp/nas_flus_photos.tsv` (26,624행), `/tmp/nas_flux_photos.tsv` (899,082행), 정렬본 `/tmp/{flus,flux}_sorted.tsv`
- 크래시 덤프 앞부분 백업: `main_server_new/logs/python3.12_crash_head_20260904.txt`