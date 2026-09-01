# 터미널 창 탭화 + 웹 인터랙티브 터미널 구현 가능성 분석

> 2026-09-01 · floating memo/terminal 패널 구현 이후 후속 기능 검토 문서
> 결론부터: **둘 다 구현 가능함.** 탭 UI는 소규모, 인터랙티브 웹 터미널은 tmux + WebSocket + xterm.js 조합이 적절하고,
> "다른 PC/모바일을 넘어가도 연속성 유지" 조건은 tmux가 본질적으로 보장한다.
>
> **2026-09-01 이후 구현 완료** (본 문서의 권장안 그대로 반영):
> - 터미널 패널 탭 바: `Main Server`(로그) + 추가/제거 가능한 tmux OS 터미널 탭
> - `GET/POST/DELETE /api/terminal/sessions` + `WS /ws/terminal/{name}?cols=&rows=` (pty ↔ `sudo tmux attach` 브리지)
> - 세션은 루트 셸로 생성(`sudo -n tmux new-session`, flux의 전수 NOPASSWD sudo 이용),
>   history-limit 20000 / xterm scrollback 20000 (상단 로그 복사용)
> - 기기 간 연속성: 세션 상태는 tmux(루트) 서버에 → 어떤 디바이스에서 접속해도 같은 세션에 붙고,
>   패널 재열림 시 서버 세션 목록으로 탭 복원, WS 단절 시 2초 간격 자동 재연결
> - 탭 제거 = `tmux kill-session` (작업 포함 완전 종료)
> - 헤더 Terminal 버튼으로 와이드 모드(min(1100px,94vw)×88vh) 열기, 플로팅 버튼은 컴팩트,
>   헤더의 확대/축소 버튼으로 전환(전환 시 재접속 — 아래 resize 제한 때문에)
>
> **구현 중 발견한 tmux/xterm 특성 (중요):**
> 1. **attach 이후 resize 시 tmux가 화면을 지운다.** 그래서 xterm fit을 먼저 수행해
>    최종 cols/rows를 WS URL 파라미터로 보내, pty를 *붙기 전*에 올바른 크기로 만든다.
>    attach 후 resize(브라우저 창 리사이즈 등)는 계속 전달되지만 화면 리셋을 감수.
>    wide/compact 전환은 대신 재접속(새 크기 attach → 전체 화면 다시 그린다) 방식으로 처리.
> 2. **systemd user 서비스 환경은 TERM=dumb** → attach 프로세스에
>    `TERM=xterm-256color` env를 명시해서 준다 (안 주면 tmux가 attach를 거부).
> 3. **bracketed paste**: bash(5.x)가 인터랙티브 셸 시작 시 2004h를 켜고, xterm.js의
>    `paste()`(클립보드 Ctrl+V 포함)가 이 모드를 존중해 `\x1b[200~...\x1b[201~`로 감싼다.
>    즉 붙여넣은 텍스트 끝의 CR은 자동 실행되지 않고(모든 최신 터미널과 동일한 안전 동작),
>    사용자가 Enter를 누르면 실행된다. 키보드 입력(Ctrl+C 등)은 key 이벤트 경로라 영향을 받지 않는다.
> 4. **Vue in-DOM 템플릿 함정**: (a) `<button>` 안에 `<button>`은 브라우저 파서가
>    재구성해 v-for 스코프가 깨진다 — 닫기 버튼은 `<span role="button">` 사용.
>    (b) `<template v-for>`는 HTML `<template>` 요소가 content 프래그먼트로 격리돼 Vue가
>    아예 못 본다 — div v-for + computed 필터로 대체.
>    (c) 패널 `v-if`가 false로 되면 호스트 DOM이 사라지므로 xterm 인스턴스는 폐기 후
>    재열림 때 재생성해야 한다(재생성 시 tmux 전체 화면 redraw로 상태가 온다).
> 5. **xterm.js는 CDN(jsdelivr)이 들쭉날쭉 끊겨** 조용히 실패하는 사례가 있어서
>    pin된 버전을 `vendor/`에 들여와 앱이 `/vendor/{xterm.js,addon-fit.js,xterm.css}`로 직접 서빙.

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

## 7. 구현 노트 — 웹 터미널 탭 (3~4라운드, 2026-09-01)

### 7.1 tmux redraw 모델 → 클라이언트 스크롤백은 절대 안 찬다 (핵심)
- tmux 클라이언트 프로토콜은 **화면 차이만** 보내므로(`/x1b[K`, insert-line,
  명시적 커서 이동), 브라우저 xterm의 스크롤백에는 히스토리가 쌓이지 않는다.
  `cat`/로그가 화면을 밀어도 xterm `buffer.active.length`는 항상 == rows,
  `viewportY`는 0에 고정. 히스토리는 **서버 쪽**(tmux, history-limit)에만 있다.
- **스크롤백 뷰어**(4라운드): 위로 스크롤 → 서버 `tmux capture-pane -p -e
  -S -N -E -2` (WS `{"type":"history","lines":20000}`) → 읽기전용 xterm
  오버레이에 SGR 색상 그대로 렌더. 여기서 드래그/선택으로 복사 가능.
  뷰어는 자체 scrollback 20000이라 더 오래된 것까지 내부 스크롤로 볼 수 있고,
  화면 끝에서 아래로 스크롤(또는 ESC, 또는 "현재 터미널로 ↓" 버튼)로 복귀.
  `-E -2`는 마지막 2줄(라이브 프롬프트 영역)을 라이브 화면에 남긴다.
- 모바일(휠 없음)은 헤더의 스크롤백 버튼(역시 역사 시계 아이콘)으로 열 수 있다.

### 7.2 xterm 휠 → 화살표 키 변환 (사용자 보고 버그의 정체)
- xterm.js 5.5.0: 스크롤백이 비어 있으면(`buffer.hasScrollback`==false) 휠을
  `getLinesScrolled()`로 변환해 **UP/DOWN 화살표 키를 앱에 전송**한다
  (옛 xterm의 스크롤=키보드로 내보내기 관행). tmux 환경에서는 스크롤백이 항상
  비어 있으므로, 위로 스크롤 = UP 화살표 = bash readline 히스토리 내비게이션
  → "이전 명령이 입력줄에 나타남" + 로그 선택 복사 불가.
- 대응: `attachCustomWheelEventHandler`가 **false를 반환하면 xterm 내장 로직
  전체가 스킵**되고 이벤트는 버블링(= veto-only API, true 반환은 "스킵"이
  아님에 주의). 현재: 위로 스크롤(false 반환)→ 버블 → `.term-host`의
  `@wheel`이 preventDefault + 뷰어 오픈. 아래 스크롤은 내장 통과(바닥에서
  getLinesScrolled=0이라 무해).

### 7.3 init 버스트 2중 발송 버그 (4라운드 발견, 수정됨)
- `\x1b=` 필터를 추가하던 중 else 분기에서 `head = data` (별칭) 후
  `data = head + data` → **4KB 윈도우 안의 모든 read가 2배로 복제되어
  발송**되는 버그를 만든 적 있다. 결과: 클라이언트가 첫 버스트(\x1b[?1049h
  \x1b[22;0;0t \x1b[?1h \x1b= \x1b[H \x1b[2J ...)를 2회 수신 +
  \x1b=가 필터 후에도 1회 생존(복제본이 원본 객체를 가리켜 필터 전 바이트였기
  때문). pty의 os.read 로그(188/523/70/523)와 클라이언트 수신(1420) 불일치로
  추적. 수정: else 분기에서 `data = b""`로 재할당.
- 현재 남아있는 필터의 역할: attach 초기 4KB에서 `\x1b=`(DECKPAM)만 제거.
  tmux는 세션 크기 ≠ 클라이언트 크기면 resize 때문에 **init 버스트를 2회**
  보내므로, 그중 application-keypad 모드 설정을 걷어내는 것. read 경계에서
  ESC와 '='가 갈라질 수 있어 1바이트 캐리. 버스트 이후의 mode 변화(앱이
  스스로 설정/해제)는 그대로 통과. (참고: xterm.js 5.5.0은 DECKPAM을 키
  매핑에 쓰지 않음 — 8개 참조는 저장/셋팅만 — 실제 영향은 미미하지만
  정상화하는 것은 무해.)

### 7.4 xterm 키 매핑 관련 오해 정리 (소스 검증, vendor/xterm.js)
- `evaluateKeyboardEvent(e, applicationCursorKeys, isMac, macOptionIsMeta)` —
  **applicationKeypad 인자가 아예 없음**. 본체 알파 키는 switch default로
  `o.key = e.key` (1자 그대로 전달). `ESC O S` 같은 keypad 이스케이프는 F1-F4
  (case 112-115)와 커서 키(37-40, DECCKM 시)뿐.
- **테스트 함정**: CDP `Input.dispatchKeyEvent`에 keyCode를 안 주면
  (또는 charCode를 주면) Linux Chrome은 `keyCode` = **문자 코드**(s→115)
  로 보고, 115는 F4 case에 걸려 `ESC O S`로 변환됐다 → 'seq 1 120'이
  'e 1 120'으로 깨지는 **테스트 아티팩트**였다(실사용 키보드는 VK 코드
  83을 써서 해당 없음). CDP 키 입력 시 `windowsVirtualKeyCode`는
  대문자 charCode(=VK 코드)로.

### 7.5 alternate screen
- tmux attach 시마다 `\x1b[?1049h` → 라이브 콘텐츠는 xterm
  `buffer.alternate` (buffer.active === alternate). 버퍼 검증 코드는
  alternate를 스캔해야 하고, **마지막 줄은 tmux 상태줄**(프롬프트는 0줄
  부근). "마지막 비어있지 않은 줄" 스캔은 상태줄에 걸린다 — 특징 문자열
  마커 스캔 사용.

### 7.6 기타 동작 검증
- Ctrl+C: 브라우저 → WS [3] → pty → tmux → pane bash, 전 구간 검증
  (trap INT 발화, sleep 30 킬, ^C 화면 출력). 2개 클라이언트 검증기
  (attach/clear)는 크기 flap/재그림 때문에 결과 해석에 주의.
- WS 단절 시 자동 재연결(패널 열려 있으면 2초 간격, 재연결 시 tmux가 전체
  화면을 다시 그려 어떤 기기에서도 현재 상태 복원).
- resize는 attach 전에 `?cols=&rows=`로 pty를 최종 크기로 만들어
  (attach 후 resize면 tmux가 화면을 지움), 이후 크기 변화는 재연결로.

### 7.7 5라운드 버그 두 건 (스크롤백 늘어짐 / 패널 재개방 검정 화면)
1. **뷰어 사다리꼴(줄이 대각으로 늘어짐)**: `capture-pane -p`의 출력이 **LF만**
   (\r 없이) 찍는다. xterm은 LF만으로는 커서를 0열로 안 되돌리므로 각 줄이
   전 줄이 끝난 x열에서 시작해 사다리꼴으로 쌓였다. → 클라이언트가 쓰기 전
   `\r\n` 정규화(`replace(/\r\n/g,'\n').replace(/\n/g,'\r\n')`).
   tmux 3.4에서 `capture-pane -E -N`의 음수 좌표는 -S 기준 상대 위치로 해석
   되는데(버전 간 불일치, 실측: `-S -20000 -E -2`가 히스토리+화면 60줄 중 33줄만
   반환) → `-E`를 빼고 `-S -N`만 사용해 전체(히스토리+현재 화면)를 캡처한다.
   capture-pane은 pane만 잡아서 tmux 상태줄은 포함되지 않는다.
2. **패널 닫기→플로팅 재개방 시 기존 탭 검정 화면**: xterm이 생성 직후
   호스트 레이아웃이 안 잡혀 있으면 fit이 기본 80x24를 주고, 그 크기로 pty를
   만든 뒤 실제 크기로 resize가 follow → tmux가 resize 시 화면을 지우는
   문제(3라운드에서 발견한 것의 재발). → ① attach 전에 fit해서 재측정,
   레이아웃이 안 잡혔으면 50ms 간격으로 최대 30회 재시도(마지막 알려진
   크기 _lastCols/Rows 폴백), ② attach 후 크기가 실제로 바뀌면(창 드래그,
   와이드/컴팩트 전환) resize 메시지를 보내지 말고 **300ms 디바운스 후 새
   크기로 재접속** — attach 시 tmux가 전체 화면을 다시 그려 항상 정상화.

### 7.8 6라운드 — 기본 사용자 root→flux, xterm 마우스-모드 휠 삼켜짐
1. **기본 사용자 전환**: web 터미널 세션이 `sudo -n tmux`(root)로 돌다가
   서비스 사용자(flux) 자신의 tmux 서버(`/tmp/tmux-1000/default`)로 전환.
   앱 프로세스가 이미 flux(user systemd)이므로 sudo 자체가 불필요해졌고,
   pipx/venv 등 사용자 공간 도구(`hf` 등)가 PATH로 바로 잡혀
   "알 수 없는 명령어"가 원천 해소. root 작업은 셸 안에서 `sudo`로
   (flux는 전수 NOPASSWD). 기존 root tmux 상의 web 세션은 소멸(스크래치).
2. **마우스 모드 + 휠 = 뷰어 사망(재현)**: flux의 `~/.tmux.conf`에
   `set -g mouse on` → tmux가 attach 시 SGR 마우스 모드(wheel 비트 포함)
   설정 → xterm이 **`.xterm` 루트에 wheel 리스너를 달고 무조건
   `stopPropagation`(cancel(e,true))** — custom wheel handler 거부(veto)
   체크도 없이. 그래서 버블 단계의 `.term-host @wheel`이 절대 이벤트를 못 받음
   (root tmux는 mouse off라 4~5라운드 테스트가 통과했던 이유).
   → **`.term-host` 리스너를 `@wheel.capture`(하강 경로)로 변경**, 위로 휠만
   `preventDefault+stopPropagation`하고 뷰어를 열기. 아래 휠(정상 스크롤,
   마우스 리포트)은 그대로 xterm에 전달 — 마우스 기능(vim 등)은 유지.
   캡처/버블 단계 탐침(모든 경로 요소에 probe)으로 절단 지점을 확인.
