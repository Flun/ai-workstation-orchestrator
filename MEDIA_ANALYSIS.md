# Local Audio / Video Text

`/av-text`는 업로드한 audio/video를 로컬 GPU에서 분석한다. 메인 FastAPI 프로세스는 파일 업로드, ffmpeg 전처리, 작업 상태, 캐시와 결과 병합만 담당한다. 모델은 역할별 별도 프로세스에서 lazy load된다.

## 구성

| 역할 | 기본 포트 | 모델 | 기본 상태 |
| --- | ---: | --- | --- |
| ASR | 8902 | Qwen3-ASR-1.7B + ForcedAligner, Whisper large-v3, Raon-Speech-9B | Qwen/Whisper 활성, Raon 비활성 |
| Audio caption | 8903 | Qwen3-Omni-30B-A3B-Captioner | 비활성(설정에서 활성화) |
| Video analysis | 8904 | Qwen3-Omni-30B-A3B-Instruct | 비활성(설정에서 활성화) |

서비스는 `127.0.0.1`에만 bind된다. 설정은 git에서 제외된 `media_analysis_settings.json`에 저장한다. 이 서버의 GPU 배치는 ASR `0`, Caption `1`, Video `0,1`이며 UI에서 바꿀 수 있다. Video Instruct BF16은 Talker를 꺼도 cold preload 직후 실측 약 73.2GB(`nvidia-smi` 예약량 약 75.0GB)라 64GB 한 장에는 들어가지 않는다. 한 장+CPU offload보다 두 장 분산이 훨씬 빠르다.

## 설치와 실행

```bash
./setup_media_analysis.sh
./start_manager.sh
```

웹에서 `Audio / Video Text`를 열고 필요한 서비스를 시작한다. 모델은 최초 요청 때 `/mnt/main-server-models/model/media-analysis` 아래 캐시에 내려받는다.

수동 실행 예:

```bash
CUDA_VISIBLE_DEVICES=0 /home/flux/media-analysis-env/bin/python media_analysis_service.py \
  --role asr --config media_analysis_settings.json --host 127.0.0.1 --port 8902
```

Captioner는 GPU1 한 장, Video 모델은 두 GPU가 보이도록 실행한다.

```bash
CUDA_VISIBLE_DEVICES=1 /home/flux/media-analysis-env/bin/python media_analysis_service.py \
  --role caption --config media_analysis_settings.json --host 127.0.0.1 --port 8903

CUDA_VISIBLE_DEVICES=0,1 /home/flux/media-analysis-env/bin/python media_analysis_service.py \
  --role video --config media_analysis_settings.json --host 127.0.0.1 --port 8904
```

## API

분석 요청은 `multipart/form-data`로 파일과 옵션을 전송하며 즉시 `job_id`를 반환한다.

```bash
curl -F file=@sample.mp4 -F mode=detailed -F asr_model=qwen3-asr \
  -F language=ko -F timestamps=true http://127.0.0.1:8999/api/media/analyze
curl http://127.0.0.1:8999/api/media/jobs/JOB_ID
curl -O http://127.0.0.1:8999/api/media/jobs/JOB_ID/download/srt
```

주요 endpoint:

- `POST /api/media/transcribe`
- `POST /api/media/audio-caption`
- `POST /api/media/analyze`
- `GET /api/media/jobs/{job_id}`
- `POST /api/media/jobs/{job_id}/cancel`
- `GET /api/media/jobs/{job_id}/download/{json|txt|srt|vtt|context}`
- `GET /api/media-analysis/status`
- `POST /api/media-analysis/services/{asr|caption|video}/{start|stop|unload}`

내부 inference service는 공통으로 `GET /health`, `GET /models`, `POST /load`, `POST /unload`를 제공한다. UI의 `서버`는 가벼운 API 프로세스만 시작하고 `GPU 로드`는 선택 모델을 실제 VRAM에 미리 올린다.

## 결과와 캐시

ASR transcript가 대사의 source of truth다. Captioner의 출력은 non-speech/ambient context에만 사용한다. 결과에는 `llm_context`가 포함되어 기존 OpenAI-compatible LLM 요청의 사용자 prompt 앞에 붙일 수 있다.

`15/30/60초` 설정은 전체 파일 길이 제한이 아니라 처리 window 크기다. 긴 파일은 끝까지 반복 분할하고 각 결과에 원본 기준 timestamp offset을 적용해 합친다. 예를 들어 5분 파일에서 60초, overlap 0이면 5개 chunk가 생성된다. overlap 2초이면 경계 문맥 보존을 위해 6개가 될 수 있으며 ASR transcript 경계의 중복 토큰은 병합 단계에서 제거한다.

캐시는 `SHA256(file) + model/options` 키로 `cache/media/<sha256>/`에 저장한다. 결과 파일은 `media_analysis_jobs/results/<job_id>/`에 생성된다. 업로드와 임시 ffmpeg 파일은 성공, 실패, 취소 시 모두 정리된다.

웹 화면은 업로드한 로컬 파일을 `<audio>`/`<video>` 기본 컨트롤로 즉시 재생한다. 분석 결과는 카드 히스토리로 기본 30일 보관하며, 사용자가 영구 저장으로 고정하거나 즉시 삭제할 수 있다. 원본 미디어는 서버에 장기 보관하지 않으므로 과거 결과 카드에서는 transcript/caption과 다운로드 파일만 다시 연다.

## Benchmark

```bash
python tools/benchmark_asr.py sample.mp4 --language ko \
  --models qwen3-asr whisper-large-v3
```

모델 load time, inference time, audio duration, RTF, VRAM과 transcript를 JSON으로 출력한다.

### 이 서버에서 확인한 값 (2026-09-20)

11.04초 한국어 샘플 영상(`KRAFTON/Raon-Speech` 저장소의 `ko_1.wav`를 MP4로 mux)을 같은 옵션으로 처리했다. 아래 load 값은 모델 파일 다운로드가 끝난 뒤의 cold process load다.

| Backend | load | inference | RTF | model VRAM |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-ASR + ForcedAligner | 12.434s | 1.488s | 0.1348 | 5.312GB |
| Whisper large-v3 | 2.747s | 1.090s | 0.0987 | 3.531GB |
| Raon-Speech-9B | 10.740s | 1.688s | 0.1529 | 16.855GB |

Captioner는 같은 오디오에서 load 41.251초, 생성 80.221초, model VRAM 59.680GB를 기록했다. Caption 결과의 작은 배경음·녹음환경 묘사는 실제 근거보다 과도할 수 있으므로, Combined 결과에서도 대사는 전문 ASR만 사용한다.

### Video Full 경로 (2026-09-21)

Full의 빠른 기본값은 `Qwen3-ASR + Qwen3-Omni Video(visual-only)`다. 대사는 ASR이 담당하고 Omni는 0.5fps로 추출한 프레임만 설명한다. 비언어적 배경음이 필요할 때만 UI의 `배경음까지 별도 분석`을 켜 Captioner를 순차 실행한다. 11.04초 샘플 end-to-end는 cold load 포함 76.621초였고 ASR 추론은 1.500초, Video 체크포인트 load는 해당 실행에서 약 33초였다. 별도 cold preload 실측은 54.295초였다. Video 프로세스는 작업 뒤 유지하므로 다음 분석부터 load 시간은 생략된다. preload 직후 Video 프로세스 점유는 두 GPU 합계 73.225GB였고, ASR까지 포함한 `nvidia-smi` 예약량은 GPU0 약 40.8GB + GPU1 약 40.9GB였다.

실제 `:8999` 업로드, ffmpeg 추출, Qwen alignment, UTF-8 transcript, JSON/TXT/SRT/VTT/context 다운로드와 headless Chromium 결과 렌더링까지 확인했다. 같은 파일·옵션의 캐시 재요청은 0.103초였다.

## 라이선스와 제한

- Qwen3-ASR / ForcedAligner: Apache-2.0
- Qwen3-Omni Captioner: 모델 카드의 `license_name`은 Apache-2.0
- Raon-Speech-9B: CC BY-NC 4.0. 상업 용도로 사용하면 안 되며 기본 비활성이다.
- Qwen3-Omni Captioner 자체는 안정적인 event timestamp를 제공하지 않는다. 화면에 표시되는 시간은 ffmpeg chunk 범위이며 모델이 생성한 timestamp가 아니다.
- 취소는 ffmpeg와 아직 시작하지 않은 단계를 즉시 멈춘다. 이미 GPU `generate()` 안에 들어간 요청은 backend 호출이 반환된 뒤 정리된다.
