# 터미널 창 탭화 + 웹 인터랙티브 터미널 구현 가능성 분석

> 2026-09-01 · floating memo/terminal 패널 구현 이후 후속 기능 검토 문서
> 결론부터: **둘 다 구현 가능함.** 탭 UI는 소규모, 인터랙티브 웹 터미널은 tmux + WebSocket + xterm.js 조합이 적절하고,
> "다른 PC/모바일을 넘어가도 연속성 유지" 조건은 tmux가 본질적으로 보장한다.

---

## 1. 요구사항 재정의

| 요구 | 해석 |
|---|---|
| 터미널로그 창에 탭 기능 | 기존 single-pane 로그 패널을 탭 컨테이너로 확장 (main_server 로그 + 서비스 로그들) |
| 추가 탭에서 서버 PC OS 터미널 조작 | 브라우저 안에서 실제 셸(bash)을 타이핑·사용하는 웹 터미널 |
| 다른 PC, 모바일을 넘어가도 연속성 | 접속 장비를 바꿔도 **같은 터미널 세션**이 그대로 (cwd, 실행 중인 프로그램, 스크롤백 유지) |

## 2. 현재 환경 사실 (검증됨)

| 항목 | 상태 |
|---|---|
| main_server | user systemd 서비스 (`systemctl --user`, `Restart=always`, `KillMode=process`), stdout/stderr → **user journal** |
| user journal 용량 | 706MB (액세스 로그가 계속 누적 — §5 참고) |
| `loginctl linger` | **enabled** → 사용자 세션/대몬이 로그아웃 후에도 유지 |
| tmux | **3.4 설치됨** (`/usr/bin/tmux`), 현재 활성 세션 없음 |
| 프론트 아키텍처 | Vue3 + Tailwind 모두 CDN(unpkg/cdn.tailwindcss) — xterm.js도 동일 방식으로 추가 가능, 빌드 시스템 없음 |
| 서버 | FastAPI(uvicorn) — **WebSocket 네이티브 지원**, 별도 데몬 불필요 |
| 노출 범위 | `0.0.0.0:8999`, 인증 없음. Tailscale은 deepseek_harness 서비스에만 사용, 메인 대시보드는 **LAN 노출** |
| 기존 유사 기능 | `/api/open_terminal/{target}` (서버 OS에 **로컬 GUI 터미널 창** 띄움 — 웹이 아님), `/api/logs/{name}` (서비스 stdout 파일) |

## 3. Phase 1 — 탭 UI (난이도: 낮음)

현 terminal 패널 헤더 아래 탭 바를 추가:

```
┌ Main Server 터미널 ─────────────────── [자동스크롤][일시정지][×]
│ [ main_server ] [ ComfyUI ] [ llama ] [ vLLM ] [ + ]
│ 09-01 17:02:11  INFO: ...
```

- **main_server 탭**: 이번에 구현한 `/api/terminal/main` (journal).
- **서비스 로그 탭**: 기존 `/api/logs/{name}`(comfyui, llama, vllm, unsloth, bot, watcher...) 재사용 —
  지금 서비스 카드의 "Console" 모달이 이미 쓰는 엔드포인트.
- 구현 방식: `terminal.tabs = [{id, label, source, lines, autoScroll}]`, 활성 탭만 렌더/폴링.
  탭 닫기(x)는 그 탭의 폴링만 정지.
- 기존 서비스 Console **모달은 유지** (전체 화면 상세 뷰). 플로팅 탭 창은 빠른 확인용.

## 4. Phase 2 — 웹 인터랙티브 터미널 (난이도: 중, 권장: tmux)

### 4.1 아키텍처

```
[브라우저 xterm.js] ──WebSocket──▶ FastAPI /ws/terminal/{session}
                                          │
                                   pty.openpty()
                                          │
                                   tmux new -A -s main  (attach)
```

1. **WS 엔드포인트**: `@app.websocket("/ws/terminal/{session}")`
   - 수신 메시지: `{"type":"stdin","data":"..."}` / `{"type":"resize","cols":C,"rows":R}`
   - 전송: `{"type":"stdout","data":"..."}` (pty master fd에서 읽은 raw byte)
2. **pty 브리지** (Python 표준 라이브러리, 추가 의존성 0):
   ```python
   master, slave = pty.openpty()
   fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
   proc = subprocess.Popen(["tmux", "new-session", "-A", "-s", name, "-x", str(cols), "-y", str(rows)],
                           stdin=slave, stdout=slave, stderr=slave, close_fds=True)
   ```
   uvicorn 이벤트 루프에서 `loop.add_reader(master, ...)`로 비동기 읽기.
3. **tmux가 연속성 요구를 만족시키는 이유** (핵심):
   - `tmux new-session -A -s main`는 세션 없으면 만들고, 있으면 **재접속**.
   - 클라이언트가 끊겨도(detach) 세션·실행 중인 프로그램·스크롤백·cwd는 **서버에서 그대로 유지**.
   - 다른 PC/모바일에서 다시 접속하면 **같은 화면에서 이어서** 사용 → 요구조건 그대로.
   - 여러 기기 동시 접속도 가능 (모두 같은 세션에 attach).
   - dashboard 재시작, 심지어 user 재로그인에도 영향 없음 (linger=on, tmux server는 user@1000 하의 독립 데몬).
   - 단, **전원 재부팅 시만 소멸** — 필요 시 main_server startup 훅에서
     `tmux has-session || tmux new -d -s main`으로 부팅 시 자동 재생성 가능 (1줄).
4. **프론트**: xterm.js를 기존 Vue3/Tailwind CDN 스택과 동일한 방식으로 추가
   (`<script>` 2개: xterm + fit addon). 터미널 탭이 활성일 때만 WS를 열고, 탭/패널이 닫히면 disconnect.

### 4.2 resize 처리
- xterm fit addon이 브라우저 리사이즈/orientation change 시 `resize` 메시지 발송 →
  `tmux resize-window -t {session} {cols}x{rows}`.
- 여러 기기 크기가 다르면 마지막 resize한 기기를 기준으로 동기화 (tmux 표준 동작).

### 4.3 보안 검토 (중요)

- 현재 대시보드(8999)는 **인증이 전혀 없음** — LAN 내 누가든 서비스 시작/정지,
  GPU 전제, OS 재부팅(`/api/panic`)까지 이미 가능. 즉 웹 셸 탭은 **새로운 신뢰 레벨이 아니라
  기존 동일 권한을 셸 형태로 묶는 것**이다. 단독 LAN 사용 전제라면 현 상태와 위험이 동일.
- 다만 향후 Tailscale/ WAN으로 대시보드를 노출할 계획이 있다면, WS 핸드셰이크에
  `?token=<비밀값>` 검증을 한 줄 단위로 추가 가능 (settings json에 토큰 저장,
  UI에 최초 1회 표시). 권장: **이 기능 도입 시점에 선택적으로 추가**.
- 명령은 flux 사용자 권한으로 실행. `sudo -n` 무임 sudoers 항목이 이미 다수 존재
  (fan/gpu-tune/170tune) — 셸에서 `sudo` 사용도 그 범위 내에서 이미 열려 있는 권한과 동일.

### 4.4 대안 비교

| 안 | 연속성 | 추가 노출 | 판단 |
|---|---|---|---|
| **tmux + WS 브리지 (권장)** | ✅ (부팅 전까지) | 없음 (8999 기존 포트) | ✅ |
| ttyd / gotty 등 오픈소스 데몬 | ✅ (ttyd는 tmux 연동 가능) | 별도 포트 + 별도 데몬 | ✕ 주 stack에 통합 안 됨 |
| tmux 없는 순수 pty spawn | ❌ 탭 닫히면 죽음 | 없음 | 요구조건 불만족 |
| 기존 `/api/open_terminal` 확장 | ❌ 로컬 GUI 터미널만 | 없음 | 원격 불가 |

## 5. 주의사항

1. **user journal 706MB**: uvicorn 접근 로그가 2초마다 쌓인다.
   `~/.config/systemd/user/...`와 무관하게 user journal은
   `systemctl --user edit`로 `LogRateLimitIntervalSec=0` 등 없이도
   `[Journal] SystemMaxUse=` (user scope: `~/.config/systemd/user/journald.conf`)
   로 상한 걸 수 있다. 웹 터미널 도입 시 접근 로그가 journal 속도를 좌우하므로
   상한 설정(예: 1G) 권장.
2. **동시 접속 시 입력 공유**: 같은 tmux window에 2개 클라이언트가 attach되면
   양쪽 화면이 같고 키 입력도 함께 들어간다. "하나의 터미널을 기기 간 이어 쓰는"
   요구사항과 정확히 일치. 기기마다 **별도** 세션이 원망되면
   세션명을 `{device}` keyed 로 만들 수 있으나, 그때는 "연속성"이 기기 당 세션 단위로
   나뉜다 — 기본은 단일 공용 세션 권장.
3. **모바일 UX**: xterm.js는 터치 입력 지원하지만 하드웨어 키보드가 없는 모바일에서는
   on-screen keyboard + xterm 가상 키보드 조합이 불편할 수 있음.
   명령 입력용으로는 충분, 대화형 TUI(vim 등)는 PC 권장.
4. **tmux 세션 네이밍/청소**: 기본 `main` 세션은 계속 두는 것이 연속성의 본체.
   필요 시 패널에 "세션 리셋" 버튼(`tmux kill-session`)을 넣을 수 있다.

## 6. 권장 진행 순서 (추정 규모)

| 단계 | 내용 | 규모 |
|---|---|---|
| 1 | 탭 바 UI (main_server + 서비스 로그 탭, `/api/logs` 재사용) | 소 (동일 파일 내 ~150줄) |
| 2 | `/ws/terminal/{session}` + pty/tmux 브리지 | 중 (~120줄, 표준 라이브러리만) |
| 3 | xterm.js 탭 + resize/재접속 처리 | 중 (~100줄, CDN 3개) |
| 4 (선택) | WS 토큰 게이트, 부팅 시 tmux 자동 재생성 | 극소 (각 ~10줄) |

4단계 합쳐도 기존 코드의 아키텍처(CDN 프론트 + FastAPI + user systemd)를
벗어나지 않고, 추가 데몬/포트/의존성이 모두 0이다. → **추천: Phase 1 + 2(단계 1~3) 진행.**
