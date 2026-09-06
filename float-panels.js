/* 플로팅 터미널 + 메모 위젯 — 서브페이지(/media, /model-hub, /vast,
 * /infrastructure, /dataset)에서 index.html과 같은 플로팅 버튼을 제공한다.
 *
 * - 자립형 Vue 3 앱(페이지의 window.Vue 사용)으로, 각 페이지의 #app 앱과
 *   분리되어 #float-panels-root에 마운트된다.
 * - 첫 탭 = 메인과 동일한 "Main Server" 로그(2s 폴링, /api/terminal/main).
 *   이후 tmux OS 터미널 탭은 서버 세션(/ws/terminal, /api/terminal/sessions)을
 *   공유해서 메인 페이지/다른 기기에서 열었던 터미널을 이어 쓴다.
 * - 검증된(4~6라운드) 동작을 그대로 반영: attach 전 크기 재측정+디퍼,
 *   크기 변경 시 디바운스 재접속(resize 메시지 대신), capture-pane LF→CRLF
 *   정규화, tmux 마우스 모드 대응 .capture 휠 가로채기, ESC 순서, 자동 재연결.
 */
(function () {
    'use strict';
    if (window.__floatPanelsLoaded) return;
    window.__floatPanelsLoaded = true;

    const root = document.getElementById('float-panels-root');
    if (!root || !window.Vue) {
        if (!window.Vue) console.error('[float-panels] window.Vue 없음 — 페이지가 Vue를 로드해야 한다');
        return;
    }

    // xterm 벤더 스크립트(localhost 서빙, pin) — 없으면 주입
    function ensureXterm() {
        if (window.Terminal && window.FitAddon) return Promise.resolve();
        return new Promise((resolve, reject) => {
            if (!document.querySelector('link[data-fp-xterm]')) {
                const link = document.createElement('link');
                link.rel = 'stylesheet'; link.href = '/vendor/xterm.css';
                link.setAttribute('data-fp-xterm', '1');
                document.head.appendChild(link);
            }
            let pending = 0;
            const done = () => { if (--pending === 0) (window.Terminal && window.FitAddon) ? resolve() : reject(new Error('xterm 로드 실패')); };
            for (const src of ['/vendor/xterm.js', '/vendor/addon-fit.js']) {
                if (document.querySelector('script[src="' + src + '"]')) continue;
                const s = document.createElement('script');
                s.src = src;
                s.onload = done; s.onerror = () => reject(new Error(src + ' 로드 실패'));
                pending++;
                document.head.appendChild(s);
            }
            if (pending === 0) (window.Terminal && window.FitAddon) ? resolve() : reject(new Error('xterm 로드 실패'));
        });
    }

    // 모바일 가로 오버플로우 차단(메인 페이지와 동일) — fixed 버튼이 밀려남 방지
    const style = document.createElement('style');
    style.textContent = 'html, body { overflow-x: hidden; } html, body { overflow-x: clip; }';
    document.head.appendChild(style);

    const THEME = {
        background: '#0c0c0e',
        foreground: '#d4d4d8',
        cursor: '#818cf8',
        cursorAccent: '#1e1b4b',
        selectionBackground: '#2d2d44',
    };

    function newLogTab() {
        return { id: 'main', label: 'Main Server', kind: 'log', closable: false, term: null, fit: null, ws: null, connected: false };
    }
    function newTermTab() {
        return {
            id: '', label: '', kind: 'term', closable: true,
            term: null, fit: null, ws: null, connected: false,
            _lastCols: 0, _lastRows: 0, _attachRetries: 0,
            _resizeTimer: null, _attachAt: 0,
        };
    }

    const app = Vue.createApp({
        data() {
            return {
                terminal: {
                    show: false,
                    tabs: [newLogTab()],
                    activeTab: 'main',
                    lines: [],
                    autoScroll: true,
                    paused: false,
                    timer: null,
                    shownLines: 0,
                    pending: 0,
                    history: { open: false, term: null, loading: false, lines: 0, fetchedAt: '' },
                },
                memo: {
                    show: false,
                    content: '',
                    saving: false,
                    lastSaved: '',
                    _saveTimer: null,
                },
            };
        },
        computed: {
            termTabs() {
                return this.terminal.tabs.filter(t => t.kind === 'term');
            },
            activeTermTab() {
                const t = this.terminal.tabs.find(x => x.id === this.terminal.activeTab);
                return (t && t.kind === 'term') ? t : null;
            },
        },
        methods: {
            // ---------- 패널 개/닫기 ----------
            openTerminalPanel() {
                this.terminal.show = true;
                this.terminal.paused = false;
                this.terminal.autoScroll = true;
                this.terminal.shownLines = 0;
                this.terminal.pending = 0;
                this.fetchLogs();
                this.startPolling();
                this.refreshSessions().then(() => {
                    this.$nextTick(() => {
                        for (const tab of this.terminal.tabs) {
                            if (tab.kind === 'term') {
                                this.ensureTerm(tab);
                                this.attachWs(tab);
                                this.fitTab(tab);
                            }
                        }
                    });
                });
            },
            closeTerminalPanel() {
                this.terminal.show = false;
                this.stopPolling();
                if (this.terminal.history.open) this.closeViewer();
                // xterm 인스턴스는 v-if가 없앤 호스트에 묶여 재사용 불가 — 폐기 후
                // 다음 열림에 재생성. tmux 세션은 서버에 유지(연속성).
                for (const tab of this.terminal.tabs) this.disposeTab(tab);
            },
            toggleTerminalPanel() {
                if (this.terminal.show) this.closeTerminalPanel();
                else this.openTerminalPanel();
            },
            toggleMemo() {
                if (this.memo.show) this.closeMemo();
                else this.openMemo();
            },

            // ---------- Main Server 로그 탭 (메인 페이지와 동일 소스) ----------
            startPolling() {
                this.stopPolling();
                this.terminal.timer = setInterval(() => {
                    if (this.terminal.show && !this.terminal.paused && !document.hidden) this.fetchLogs();
                }, 2000);
            },
            stopPolling() {
                if (this.terminal.timer) { clearInterval(this.terminal.timer); this.terminal.timer = null; }
            },
            async fetchLogs() {
                try {
                    const res = await fetch('/api/terminal/main?lines=400&t=' + Date.now());
                    if (!res.ok) return;
                    const data = await res.json();
                    const lines = (data.lines || []).map(l =>
                        (typeof l === 'string') ? { t: '', s: l } : { t: l.t || '', s: l.s || '' }
                    );
                    // 사용자가 스크롤을 올려 읽는 중이면 리스트를 교체하지 않고
                    // 새 줄 수만 세어 둔다 — 선택/복사가 흔들리지 않는다.
                    if (!this.terminal.autoScroll) {
                        this.terminal.pending = Math.max(0, lines.length - this.terminal.shownLines);
                        return;
                    }
                    this.terminal.shownLines = lines.length;
                    this.terminal.pending = 0;
                    this.terminal.lines = lines;
                    this.$nextTick(() => {
                        const el = this.$el.querySelector('#fp-terminal-log');
                        if (el) el.scrollTop = el.scrollHeight;
                    });
                } catch (e) {}
            },
            // 터미널 스크롤백 뷰어와 동일한 라이브 추적 규칙:
            // 위로 스크롤하면 추적 해제(읽기 고정), 바닥까지 내리면 자동 재개.
            onMainLogScroll(event) {
                const el = event.target;
                const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= 24;
                if (atBottom === this.terminal.autoScroll) return;
                this.terminal.autoScroll = atBottom;
                if (atBottom) this.fetchLogs();   // 바닥에 닿으면 즉시 최신 반영
            },
            resumeMainLogLive() {
                this.terminal.autoScroll = true;
                this.fetchLogs();
            },
            togglePaused() {
                this.terminal.paused = !this.terminal.paused;
                if (!this.terminal.paused) this.fetchLogs();
            },

            // ---------- 세션 동기화(연속성) ----------
            async refreshSessions() {
                try {
                    const res = await fetch('/api/terminal/sessions');
                    if (!res.ok) return;
                    const data = await res.json();
                    const names = new Set((data.sessions || []).map(s => s.name));
                    // 사라진 세션(타 기기에서 종료/재부팅) 정리 — 'main' 로그 탭은 유지
                    for (const tab of [...this.terminal.tabs]) {
                        if (tab.kind !== 'term') continue;
                        if (!names.has(tab.id)) {
                            this.disposeTab(tab);
                            this.terminal.tabs.splice(this.terminal.tabs.indexOf(tab), 1);
                            if (this.terminal.activeTab === tab.id) this.terminal.activeTab = 'main';
                        }
                    }
                    // 새로 생긴 세션 추가
                    for (const s of (data.sessions || [])) {
                        if (!this.terminal.tabs.some(t => t.id === s.name)) {
                            const tab = newTermTab();
                            tab.id = s.name; tab.label = s.name;
                            this.terminal.tabs.push(tab);
                        }
                    }
                } catch (e) {}
            },
            async addTermTab() {
                try { await ensureXterm(); } catch (e) {
                    alert('터미널 라이브러리(xterm.js) 로드 실패: ' + e.message + '\n페이지를 새로고침해 주세요.');
                    return;
                }
                try {
                    const res = await fetch('/api/terminal/sessions', { method: 'POST' });
                    const data = await res.json().catch(() => ({}));
                    if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
                    const tab = newTermTab();
                    tab.id = data.name; tab.label = data.name;
                    this.terminal.tabs.push(tab);
                    this.switchTab(data.name);
                } catch (e) {
                    alert('터미널 탭 생성 실패: ' + e.message);
                }
            },
            switchTab(id) {
                const tab = this.terminal.tabs.find(t => t.id === id);
                if (!tab) return;
                this.terminal.activeTab = id;
                if (tab.kind === 'term') {
                    this.$nextTick(() => {
                        this.ensureTerm(tab);
                        this.attachWs(tab);
                        this.fitTab(tab);
                        if (tab.term) tab.term.focus();
                    });
                }
            },
            async closeTermTab(id) {
                const tab = this.terminal.tabs.find(t => t.id === id);
                if (!tab || !tab.closable) return;
                if (!confirm(`터미널 "${id}"를 닫으려면 세션이 종료됩니다.\n그 터미널에서 실행 중인 작업도 함께 사라집니다.\n\n계속할까요?`)) return;
                try { await fetch('/api/terminal/sessions/' + encodeURIComponent(id), { method: 'DELETE' }); } catch (e) {}
                this.disposeTab(tab);
                const idx = this.terminal.tabs.indexOf(tab);
                if (idx >= 0) this.terminal.tabs.splice(idx, 1);
                if (this.terminal.activeTab === id) {
                    this.terminal.activeTab = 'main';
                }
            },
            disposeTab(tab) {
                if (tab.kind !== 'term') return;
                if (tab._resizeTimer) { clearTimeout(tab._resizeTimer); tab._resizeTimer = null; }
                if (tab.ws) { try { tab.ws.onclose = null; tab.ws.close(); } catch (e) {} tab.ws = null; }
                if (tab.term) { try { tab.term.dispose(); } catch (e) {} tab.term = null; tab.fit = null; }
                tab._attachRetries = 0;
                tab.connected = false;
            },

            // ---------- xterm + WS ----------
            ensureTerm(tab) {
                if (tab.term) return;
                if (!window.Terminal || !window.FitAddon) return;
                const host = this.$el.querySelector('.fp-term-host[data-term="' + tab.id + '"]');
                if (!host) return;
                const term = new window.Terminal({
                    cursorBlink: true,
                    fontSize: 12.5,
                    fontFamily: "'JetBrains Mono', monospace",
                    scrollback: 20000,
                    allowProposedApi: true,
                    theme: THEME,
                });
                const fit = new window.FitAddon.FitAddon();
                term.loadAddon(fit);
                term.open(host);
                tab.term = term;
                tab.fit = fit;
                term.onData((d) => {
                    if (tab.ws && tab.ws.readyState === WebSocket.OPEN) {
                        tab.ws.send(new TextEncoder().encode(d));
                    }
                });
                // 마우스 모드 꺼진(tmux mouse off) 환경의 내장 휠(화살표키 리포트)
                // 방어용 거부. 켜져 있는(mouse on) 환경의 가로채기는 .capture가 담당.
                term.attachCustomWheelEventHandler((event) => this.onLiveCustomWheel(tab, event));
                // ESC: 뷰어 열려 있으면 거기서 소비(셸 주입 방지). xterm은 ESC를
                // stopPropagation 하기 때문에 전역 핸들러도 직접 막는다.
                term.attachCustomKeyEventHandler((e) => {
                    if (e.type === 'keydown' && e.key === 'Escape' && this.terminal.history.open) {
                        e.preventDefault();
                        e.stopPropagation();
                        this.closeViewer();
                        return false;
                    }
                    return true;
                });
                if (host.clientWidth > 10 && host.clientHeight > 10) {
                    try { fit.fit(); } catch (e) {}
                }
                this.attachWs(tab);
            },
            attachWs(tab) {
                if (tab.ws) { try { tab.ws.onclose = null; tab.ws.close(); } catch (e) {} tab.ws = null; }
                // attach 전에 최종 크기를 재야 한다 — pty를 그 크기로 만들어
                // (서버 TIOCSWINSZ) attach 직후 resize로 tmux가 화면을 지우는
                // 일을 막는다. 레이아웃이 안 잡히면 디퍼 재시도.
                if (tab.term && tab.fit) { try { tab.fit.fit(); } catch (e) {} }
                let cols = 0, rows = 0;
                if (tab.term && tab.term.cols > 1 && tab.term.rows > 1) {
                    cols = tab.term.cols; rows = tab.term.rows;
                    tab._lastCols = cols; tab._lastRows = rows;
                } else if (tab._lastCols > 1 && tab._lastRows > 1) {
                    cols = tab._lastCols; rows = tab._lastRows;
                }
                if (!cols || !rows) {
                    if (this.terminal.show && (tab._attachRetries || 0) < 30) {
                        tab._attachRetries = (tab._attachRetries || 0) + 1;
                        setTimeout(() => {
                            if (!this.terminal.show || !tab.term) return;
                            if (tab.ws && tab.ws.readyState <= WebSocket.OPEN &&
                                tab._attachAt && Date.now() - tab._attachAt < 2000) return;
                            this.attachWs(tab);
                        }, 50);
                        return;
                    }
                }
                tab._attachRetries = 0;
                const proto = location.protocol === 'https:' ? 'wss' : 'ws';
                let url = proto + '://' + location.host + '/ws/terminal/' + encodeURIComponent(tab.id);
                if (cols > 1 && rows > 1) url += '?cols=' + cols + '&rows=' + rows;
                const ws = new WebSocket(url);
                tab._attachAt = Date.now();
                ws.binaryType = 'arraybuffer';
                tab.ws = ws;
                ws.onopen = () => {
                    tab.connected = true;
                    tab._lastCols = tab.term ? tab.term.cols : 0;
                    tab._lastRows = tab.term ? tab.term.rows : 0;
                    this.fitTab(tab);
                    if (tab.term) tab.term.focus();
                };
                ws.onmessage = (ev) => {
                    if (typeof ev.data === 'string') {
                        try {
                            const m = JSON.parse(ev.data);
                            if (m.type === 'history') {
                                this.renderViewer(tab, m);
                            } else if (m.type === 'err' && tab.term) {
                                tab.connected = false;
                                tab.term.writeln('\r\n\x1b[31m' + m.message + '\x1b[0m');
                            } else if (m.type === 'closed' && tab.term) {
                                tab.connected = false;
                                tab.term.writeln('\r\n\x1b[33m[연결이 닫혔습니다]\x1b[0m');
                            }
                        } catch (e) {}
                    } else if (tab.term) {
                        tab.term.write(new Uint8Array(ev.data));
                    }
                };
                ws.onclose = () => {
                    if (tab.ws !== ws) return;
                    tab.connected = false;
                    // 패널이 열려 있으면 2초 간격 자동 재연결
                    if (this.terminal.show) {
                        setTimeout(() => {
                            if (this.terminal.show && tab.ws === ws && !tab.connected && tab.term) {
                                this.attachWs(tab);
                            }
                        }, 2000);
                    }
                };
                ws.onerror = () => {};
            },
            fitTab(tab) {
                if (!tab.term || !tab.fit) return;
                requestAnimationFrame(() => {
                    try {
                        tab.fit.fit();
                        const c = tab.term.cols, r = tab.term.rows;
                        if (c === tab._lastCols && r === tab._lastRows) return;
                        tab._lastCols = c; tab._lastRows = r;
                        if (!this.terminal.show) return;
                        // 크기 변경 시 resize 메시지 대신 300ms 디바운스 후 재접속 —
                        // attach 시 tmux가 전체 화면을 다시 그려 항상 정상화된다.
                        if (tab._resizeTimer) clearTimeout(tab._resizeTimer);
                        tab._resizeTimer = setTimeout(() => {
                            tab._resizeTimer = null;
                            if (this.terminal.show && tab.term) this.attachWs(tab);
                        }, 300);
                    } catch (e) {}
                });
            },
            onResize() {
                for (const tab of this.terminal.tabs) {
                    if (tab.term && this.terminal.activeTab === tab.id) this.fitTab(tab);
                }
            },

            // ---------- 스크롤백 뷰어 ----------
            onLiveCustomWheel(tab, event) {
                if (this.terminal.history.open) return true;
                // tmux 환경에서 브라우저 스크롤백은 비어 있으므로 위로 스크롤은
                // 전부 스크롤백 뷰어로 (false 반환 → xterm 내장 스킵)
                return !(event.deltaY < 0);
            },
            // .capture 필수: tmux 마우스 모드(mouse on)에서 xterm이 .xterm 루트에서
            // 모든 wheel을 stopPropagation한다 — 버블로는 못 오므로 하강 경로에서
            // 가로챈다. 위로만 stopPropagation(아래=정상 스크롤/마우스 유지).
            onHostWheel(event) {
                const tab = this.activeTermTab;
                if (!tab || !tab.term) return;
                if (this.terminal.history.open) return;
                if (event.deltaY < 0) {
                    event.preventDefault();
                    event.stopPropagation();
                    this.openViewer(tab);
                }
            },
            openViewer(tab) {
                if (!tab) tab = this.activeTermTab;
                if (!tab || tab.kind !== 'term') return;
                if (this.terminal.history.open) return;
                if (!tab.ws || tab.ws.readyState !== WebSocket.OPEN) {
                    this.attachWs(tab); // 연결이 살아나면 사용자가 다시 요청
                    return;
                }
                this.terminal.history = { open: true, term: null, loading: true, lines: 0, fetchedAt: '' };
                tab.ws.send(JSON.stringify({ type: 'history', lines: 20000 }));
            },
            renderViewer(tab, msg) {
                const h = this.terminal.history;
                if (!h.open) return;
                if (!window.Terminal || !window.FitAddon) {
                    this.terminal.history = { open: false, term: null, loading: false, lines: 0, fetchedAt: '' };
                    return;
                }
                let text = '';
                try {
                    const bin = atob(msg.data || '');
                    const bytes = new Uint8Array(bin.length);
                    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                    text = new TextDecoder().decode(bytes);
                } catch (e) {}
                // capture-pane -p는 LF만(0x0A) 찍는다 — xterm은 LF만으로는 커서를
                // 0열로 안 되돌려서 줄이 사다리꼴로 늘어진다. CRLF로 정규화.
                text = text.replace(/\r\n/g, '\n').replace(/\n/g, '\r\n');
                const host = this.$el.querySelector('.fp-history-host');
                if (!host) {
                    this.$nextTick(() => this.renderViewer(tab, msg));
                    return;
                }
                const cols = (tab.term && tab.term.cols > 2) ? tab.term.cols : 80;
                const term = new window.Terminal({
                    cols, rows: 50,
                    scrollback: 20000,
                    disableStdin: true, disableCursor: true, cursorBlink: false,
                    fontSize: 12.5,
                    fontFamily: "'JetBrains Mono', monospace",
                    theme: THEME,
                });
                const fit = new window.FitAddon.FitAddon();
                term.loadAddon(fit);
                term.open(host);
                h.term = term;
                term.attachCustomKeyEventHandler((e) => {
                    if (e.type === 'keydown' && e.key === 'Escape') {
                        e.preventDefault();
                        e.stopPropagation();
                        this.closeViewer();
                        return false;
                    }
                    return true;
                });
                try { fit.fit(); } catch (e) {}
                term.write(text || '(스크롤백이 없습니다)', () => {
                    try { fit.fit(); } catch (e) {}
                    term.scrollToBottom();
                    h.loading = false;
                    h.lines = text ? text.split(/\r?\n/).length : 0;
                    h.fetchedAt = new Date().toTimeString().slice(0, 8);
                });
            },
            closeViewer() {
                const h = this.terminal.history;
                if (h.term) { try { h.term.dispose(); } catch (e) {} }
                this.terminal.history = { open: false, term: null, loading: false, lines: 0, fetchedAt: '' };
                const tab = this.activeTermTab;
                if (tab && tab.term) {
                    this.$nextTick(() => { try { tab.term.focus(); } catch (e) {} });
                }
            },
            onViewerWheel(event) {
                const h = this.terminal.history;
                if (!h.term) return;
                const b = h.term.buffer.active;
                const atBottom = b.viewportY >= b.length - h.term.rows - 1;
                if (event.deltaY > 0 && atBottom) {
                    event.preventDefault();
                    this.closeViewer();
                }
            },

            // ---------- 메모 ----------
            async openMemo() {
                this.memo.show = true;
                try {
                    const res = await fetch('/api/memo');
                    if (res.ok) {
                        const data = await res.json();
                        if (this.memo.content === '' && data.memo) this.memo.content = data.memo;
                    }
                } catch (e) {}
                this.$nextTick(() => {
                    const el = this.$el.querySelector('.fp-memo-area');
                    if (el) el.focus();
                });
            },
            closeMemo() { this.memo.show = false; },
            async saveMemoNow() {
                this.memo.saving = true;
                clearTimeout(this.memo._saveTimer);
                try {
                    await fetch('/api/memo?memo=' + encodeURIComponent(this.memo.content), { method: 'POST' });
                    this.memo.lastSaved = new Date().toTimeString().slice(0, 8);
                } catch (e) {}
                this.memo.saving = false;
            },
            debouncedSaveMemo() {
                this.memo.saving = true;
                clearTimeout(this.memo._saveTimer);
                this.memo._saveTimer = setTimeout(() => this.saveMemoNow(), 1000);
            },
            async clearMemo() {
                if (!confirm('메모를 지울까요?')) return;
                this.memo.content = '';
                await this.saveMemoNow();
            },
            closeAll() {
                this.closeMemo();
                this.closeTerminalPanel();
            },

            // ESC: 뷰어(열려 있으면) → 메모 순서. 터미널 패널의 ESC는 셸에
            // 가므로(커스텀 키 핸들러가 true 반환) 여기에서 안 닫는다.
            onKeydown(event) {
                if (event.key !== 'Escape') return;
                if (this.terminal.history.open) { this.closeViewer(); return; }
                if (this.memo.show) { this.closeMemo(); return; }
            },
        },
        async mounted() {
            window.addEventListener('resize', this.onResize);
            window.addEventListener('keydown', this.onKeydown);
            try { await ensureXterm(); } catch (e) {
                console.warn('[float-panels] xterm 로드 실패:', e.message);
            }
            this.refreshSessions();
        },
        beforeUnmount() {
            window.removeEventListener('resize', this.onResize);
            window.removeEventListener('keydown', this.onKeydown);
            this.stopPolling();
            for (const tab of this.terminal.tabs) this.disposeTab(tab);
        },
        template: `
<div class="fp-root">
    <!-- 바깥 클릭 감지 오버레이(투명) -->
    <div v-if="memo.show || terminal.show" class="fixed inset-0 z-30" @click.self="closeAll()"></div>

    <!-- 플로팅 버튼 스택: 터미널(위) / 메모(아래) -->
    <div class="fixed bottom-6 right-6 z-40 flex flex-col items-center gap-3">
        <button @click="toggleTerminalPanel" title="Main Server 터미널" aria-label="Main Server 터미널 열기"
            class="w-12 h-12 rounded-full border shadow-lg flex items-center justify-center transition"
            :class="terminal.show ? 'bg-emerald-500/15 border-emerald-500/60 text-emerald-300' : 'bg-zinc-800 border-zinc-700 hover:border-zinc-500 hover:bg-zinc-700 text-zinc-300'">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m4 17 6-5-6-5M12 19h8"></path></svg>
        </button>
        <button @click="toggleMemo" title="메모" aria-label="메모 열기"
            class="w-12 h-12 rounded-full border shadow-lg flex items-center justify-center transition"
            :class="memo.show ? 'bg-indigo-500/15 border-indigo-500/60 text-indigo-300' : 'bg-zinc-800 border-zinc-700 hover:border-zinc-500 hover:bg-zinc-700 text-zinc-300'">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
        </button>
    </div>

    <!-- 메모 패널 -->
    <div v-if="memo.show" role="dialog" aria-label="메모"
        class="fixed bottom-36 right-6 w-[calc(100vw-3rem)] sm:w-96 bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl z-40 flex flex-col overflow-hidden">
        <div class="flex items-center gap-2.5 bg-zinc-950 px-4 py-3 border-b border-zinc-800">
            <span class="w-2 h-2 rounded-full bg-indigo-500"></span>
            <h3 class="font-semibold text-zinc-200 text-xs uppercase tracking-wider">메모</h3>
            <span class="text-[10px] text-zinc-600 font-mono">{{ memo.content.length }}자</span>
            <button @click="closeMemo" class="ml-auto text-zinc-500 hover:text-zinc-200 transition" title="닫기 (Esc)">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
        </div>
        <textarea class="fp-memo-area w-full h-72 bg-transparent text-[13px] p-4 outline-none text-zinc-300 resize-none font-mono leading-relaxed selection:bg-indigo-500/30"
            v-model="memo.content" @input="debouncedSaveMemo"
            @keydown.ctrl.s.prevent="saveMemoNow" @keydown.meta.s.prevent="saveMemoNow"
            placeholder="포트, 프롬프트, 간단한 설정 메모를 적어두세요. 입력 후 1초면 서버에 자동 저장됩니다. (Ctrl+S 즉시 저장)"></textarea>
        <div class="bg-zinc-950 px-4 py-2.5 text-[10px] text-zinc-500 flex items-center justify-between gap-2 border-t border-zinc-800">
            <div class="flex items-center gap-1.5 min-w-0">
                <span class="w-1.5 h-1.5 rounded-full shrink-0" :class="memo.saving ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'"></span>
                <span class="truncate" v-if="memo.saving">서버 저장 중...</span>
                <span v-else-if="memo.lastSaved">서버에 저장됨 · {{ memo.lastSaved }}</span>
                <span v-else>서버 자동 저장 대기</span>
            </div>
            <div class="flex items-center gap-1.5 shrink-0">
                <button @click="saveMemoNow" class="px-2.5 py-1 rounded-md border border-zinc-700 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-medium">저장</button>
                <button @click="clearMemo" class="px-2.5 py-1 rounded-md border border-zinc-800 text-zinc-500 hover:text-rose-400 hover:border-rose-500/40 font-medium">지우기</button>
            </div>
        </div>
    </div>

    <!-- 터미널 패널(컴팩트 플로팅) -->
    <div v-if="terminal.show" role="dialog" aria-label="터미널"
        class="fixed bottom-36 right-6 w-[calc(100vw-3rem)] h-[68vh] max-h-[48rem] sm:w-[40rem] bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl z-40 flex flex-col overflow-hidden">
        <!-- 헤더 -->
        <div class="flex items-center gap-3 bg-zinc-950 px-4 py-3 border-b border-zinc-800 shrink-0">
            <span class="w-2.5 h-2.5 rounded-full shrink-0"
                :class="terminal.activeTab === 'main' ? (!terminal.paused ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500') : (activeTermTab && activeTermTab.connected ? 'bg-emerald-500' : 'bg-amber-500')"></span>
            <h3 class="min-w-0 truncate font-mono text-sm font-medium text-zinc-200">터미널</h3>
            <span class="text-[10px] text-zinc-600 font-mono shrink-0 hidden sm:inline">
                {{ terminal.activeTab === 'main' ? terminal.lines.length : '' }}줄 · {{ terminal.tabs.length }}탭
            </span>
            <div class="ml-auto flex items-center gap-2.5 shrink-0">
                <label v-if="terminal.activeTab === 'main'" class="flex items-center gap-1.5 text-[11px] text-zinc-400 cursor-pointer hover:text-zinc-200 transition select-none" title="스크롤을 위로 올리면 자동 추적이 꺼지고, 바닥까지 내리거나 이 체크박스를 켜면 다시 라이브 추적이 재개됩니다">
                    <input type="checkbox" v-model="terminal.autoScroll" @change="terminal.autoScroll && fetchLogs()" class="w-3.5 h-3.5 rounded bg-zinc-800 border-zinc-700 text-emerald-500 focus:ring-emerald-500/50"> 자동 스크롤
                </label>
                <button v-if="terminal.activeTab === 'main'" @click="togglePaused" class="text-[11px] font-medium px-2.5 py-1 rounded-md border transition"
                    :class="terminal.paused ? 'bg-amber-500/15 border-amber-500/40 text-amber-300' : 'bg-zinc-800 border-zinc-700 text-zinc-300 hover:bg-zinc-700'">
                    {{ terminal.paused ? '재개' : '일시정지' }}
                </button>
                <span v-if="activeTermTab" class="flex items-center gap-1.5 text-[10px] font-mono" :class="activeTermTab.connected ? 'text-emerald-400' : 'text-amber-400'">
                    <span class="w-1.5 h-1.5 rounded-full" :class="activeTermTab.connected ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'"></span>
                    {{ activeTermTab.connected ? '연결됨' : '재연결 중...' }}
                </span>
                <button v-if="activeTermTab" @click="openViewer()" class="p-1 text-zinc-500 hover:text-zinc-200 transition"
                    title="스크롤백 보기 — 현재 화면 위로 스크롤해도 되는 전체 히스토리(서버 기록)" aria-label="스크롤백 보기">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12a9 9 0 1 0 9-9 9.75 9 0 0 0-6.74 2.74L3 8"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3v5h5M12 7v5l3 3"></path></svg>
                </button>
                <button @click="closeTerminalPanel" class="text-zinc-500 hover:text-zinc-200 transition" title="닫기">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                </button>
            </div>
        </div>

        <!-- 탭 바: Main Server 로그(첫째, 닫을 수 없음) + tmux 탭 + 추가 -->
        <div class="flex items-center gap-1.5 px-3 py-2 bg-zinc-950/80 border-b border-zinc-800 overflow-x-auto shrink-0">
            <button v-for="tab in terminal.tabs" :key="'fp-tab-'+tab.id" @click="switchTab(tab.id)"
                class="group shrink-0 flex items-center gap-1.5 pl-2.5 pr-1.5 py-1.5 rounded-md text-[11px] font-medium border transition"
                :class="terminal.activeTab === tab.id ? 'bg-zinc-800 border-zinc-600 text-zinc-100' : 'bg-transparent border-transparent text-zinc-500 hover:text-zinc-300 hover:border-zinc-800'"
                :title="tab.kind === 'log' ? 'main_server 로그' : ('tmux 세션 ' + tab.id)">
                <span class="w-1.5 h-1.5 rounded-full shrink-0" :class="tab.kind === 'log' ? 'bg-sky-500' : (tab.connected ? 'bg-emerald-500' : 'bg-amber-500')"></span>
                <span class="max-w-[10rem] truncate">{{ tab.label }}</span>
                <span v-if="tab.closable" role="button" tabindex="0" @click.stop="closeTermTab(tab.id)" @keydown.enter.stop="closeTermTab(tab.id)"
                    class="hidden group-hover:inline-flex p-0.5 rounded text-zinc-500 hover:text-zinc-200 hover:bg-zinc-700" title="탭 닫기(세션 종료)">
                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                </span>
            </button>
            <button @click="addTermTab" title="OS 터미널 탭 추가" aria-label="OS 터미널 탭 추가"
                class="shrink-0 p-1.5 rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800 transition ml-auto">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.5v15m7.5-7.5h-15"></path></svg>
            </button>
        </div>

        <!-- 본문: 로그 뷰 / xterm 호스트 -->
        <div class="flex-1 min-h-0 relative">
            <div v-show="terminal.activeTab === 'main'" id="fp-terminal-log" class="absolute inset-0 overflow-y-auto bg-[#0c0c0e] p-3 sm:p-4 font-mono text-[11px] leading-[1.65]" @scroll.passive="onMainLogScroll">
                <div v-if="!terminal.lines.length" class="text-zinc-600">로딩 중...</div>
                <div v-for="(line, i) in terminal.lines" :key="i" class="flex gap-2.5 whitespace-pre-wrap break-words">
                    <span class="shrink-0 select-none text-zinc-600/90">{{ line.t }}</span>
                    <span class="min-w-0 text-zinc-300">{{ line.s }}</span>
                </div>
            </div>
            <!-- 스크롤백 뷰어의 "현재 터미널로 ↓"와 같은 라이브 복귀 버튼 -->
            <button v-if="terminal.activeTab === 'main' && !terminal.autoScroll" @click="resumeMainLogLive" class="absolute bottom-4 right-4 z-10 flex items-center gap-2 rounded-md border border-emerald-500/40 bg-zinc-900/90 px-2.5 py-1 text-[11px] font-medium text-emerald-300 shadow-lg backdrop-blur transition hover:bg-zinc-800 hover:border-emerald-500/60">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                <span>{{ terminal.pending ? '새 로그 ' + terminal.pending + '줄' : '라이브 로그' }}</span> ↓
            </button>
            <div v-for="tab in termTabs" :key="'fp-host-'+tab.id" :data-term="tab.id" v-show="terminal.activeTab === tab.id"
                @wheel.capture="onHostWheel"
                class="fp-term-host absolute inset-0 bg-[#0c0c0e] p-2.5"></div>
            <!-- 스크롤백 뷰어 오버레이 -->
            <div v-if="terminal.history.open" @wheel="onViewerWheel" class="absolute inset-0 z-10 bg-[#0c0c0e] flex flex-col">
                <div class="flex items-center gap-2.5 px-3 py-1.5 bg-zinc-950 border-b border-zinc-800 shrink-0">
                    <svg class="w-3.5 h-3.5 text-zinc-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12a9 9 0 1 0 9-9 9.75 9 0 0 0-6.74 2.74L3 8"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3v5h5"></path></svg>
                    <span class="text-[11px] font-medium text-zinc-300 shrink-0">스크롤백</span>
                    <span class="text-[10px] font-mono text-zinc-600 truncate">{{ terminal.history.lines }}줄 · {{ terminal.history.fetchedAt }} 스냅샷</span>
                    <span v-if="terminal.history.loading" class="text-[10px] font-mono text-amber-400 animate-pulse shrink-0">불러오는 중...</span>
                    <button @click="closeViewer" class="ml-auto shrink-0 text-[11px] font-medium px-2 py-0.5 rounded-md border border-zinc-700 bg-zinc-800 text-zinc-300 hover:bg-zinc-700 transition">현재 터미널로 ↓</button>
                </div>
                <div class="fp-history-host flex-1 min-h-0"></div>
            </div>
        </div>
    </div>
</div>
`,
    });
    app.mount(root);
})();
