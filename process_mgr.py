import os
import signal
import subprocess
import threading
import time

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _pgid_of(pid):
    """프로세스의 그룹 ID(pgid). 프로세스가 이미 죽거나 읽지 못하면 None."""
    try:
        import psutil
        return psutil.Process(pid).pgid()
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read().decode("ascii", "replace")
        # comm(프로세스 이름)에 공백이 들어갈 수 있으므로 마지막 ')' 기준으로 분리.
        fields = stat.rsplit(")", 1)[1].split()
        # state(0) ppid(1) pgrp(2) session(3) ...
        return int(fields[2])
    except Exception:
        return None


def _children_of(pid):
    """프로세스의 모든 자손 PID (실패하면 빈 목록)."""
    try:
        import psutil
        return [child.pid for child in psutil.Process(pid).children(recursive=True)]
    except Exception:
        return []

class Service:
    """한 서비스(ComfyUI / llama / bot / watcher) 프로세스 관리.

    Linux: start_new_session으로 완전 분리(headless), 로그는 파일로.
    Windows: CREATE_NO_WINDOW로 창 숨김.
    """

    def __init__(self, name, stop_timeout=6, kill_timeout=2):
        self.name = name
        self.pid = None
        self.started_at = None
        self.device = None
        self.phase = "stopped"
        self.generation = 0
        self._process = None
        self._lock = threading.RLock()
        self.stop_timeout = stop_timeout
        self.kill_timeout = kill_timeout
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_file = os.path.join(LOG_DIR, f"{name}.log")
        self._pidfile = os.path.join(LOG_DIR, f"{name}.pid")

    # ---- 상태 ----
    def read_pidfile(self):
        try:
            with open(self._pidfile) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _pid_alive(self, pid):
        try:
            import psutil
            if not psutil.pid_exists(pid):
                return False
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            pass
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _process_tree_alive(self, pid):
        """Return whether the process and its tree still own live work.

        Own-session processes (start_new_session) are checked through their
        process group, which also covers descendants outliving the parent.
        External processes do not lead their own group (it may belong to a
        user terminal session), so their own liveness is the check instead.
        """
        if os.name == "nt":
            return self._pid_alive(pid)
        if self._pid_alive(pid):
            return True
        try:
            os.killpg(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _reap(self):
        if self._process is not None:
            try:
                self._process.poll()
            except Exception:
                pass

    def running(self):
        with self._lock:
            pid = self.read_pidfile()
            if pid and self._process_tree_alive(pid):
                self.pid = pid
                if self.started_at is None:
                    try:
                        import psutil
                        self.started_at = psutil.Process(pid).create_time()
                    except Exception:
                        pass
                if self.phase == "stopped":
                    self.phase = "starting"
                return True
            self._reap()
            self.pid = None
            self.started_at = None
            self._process = None
            if self.phase != "stopping":
                self.phase = "stopped"
            return False

    def info(self):
        if self.running():
            uptime = time.time() - self.started_at if self.started_at else 0
            return {
                "running": True, "pid": self.pid, "uptime": round(uptime),
                "phase": self.phase, "generation": self.generation,
                "device": list(self.device) if isinstance(self.device, (list, tuple)) else self.device,
            }
        return {
            "running": False, "pid": None, "uptime": 0,
            "phase": self.phase, "generation": self.generation,
        }

    # ---- 시작/종료 ----
    def start(self, cmd, cwd=None, env=None, device=None):
        # Keep the liveness check, spawn, and PID publication under one lock.
        # Stop can therefore never observe a spawned-but-untracked process.
        with self._lock:
            if self.phase == "stopping":
                raise RuntimeError(f"{self.name} 서비스가 종료 중입니다")
            if self.running():
                raise RuntimeError(f"{self.name} 서비스가 이미 실행 중입니다 (PID {self.pid})")
            self.generation += 1
            self.phase = "starting"
            os.makedirs(LOG_DIR, exist_ok=True)
            env_full = dict(os.environ)
            if env:
                env_full.update(env)
            if device is not None:
                if isinstance(device, (list, tuple)):
                    visible_devices = ",".join(str(item).strip() for item in device if str(item).strip())
                else:
                    visible_devices = str(device).strip()
                if visible_devices:
                    env_full["CUDA_VISIBLE_DEVICES"] = visible_devices
            log_f = open(self.log_file, "a", encoding="utf-8", errors="replace")
            try:
                log_f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작: {' '.join(cmd)} =====\n")
                log_f.flush()
                creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                p = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env_full,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=(os.name != "nt"),
                    creationflags=creationflags,
                )
            except Exception:
                self.phase = "stopped"
                raise
            finally:
                log_f.close()
            self._process = p
            self.pid = p.pid
            self.started_at = time.time()
            self.device = list(device) if isinstance(device, (list, tuple)) else device
            try:
                with open(self._pidfile, "w") as f:
                    f.write(str(p.pid))
            except Exception:
                # Publication is part of start. Never leave an untracked child
                # behind if the pidfile cannot be persisted.
                try:
                    self._signal_tree(p.pid, force=True)
                except (OSError, ProcessLookupError):
                    pass
                self._wait_for_exit(p.pid, self.kill_timeout)
                self._clear_run(p.pid)
                self.phase = "stopped"
                raise
            return p.pid

    def _wait_for_exit(self, pid, timeout):
        deadline = time.monotonic() + timeout
        while self._process_tree_alive(pid) and time.monotonic() < deadline:
            self._reap()
            time.sleep(0.1)
        self._reap()
        return not self._process_tree_alive(pid)

    def _signal_tree(self, pid, force):
        """프로세스 트리에 시그널 전송.

        자신이 만든 새 세션(start_new_session, pgid == pid)이면 그룹 전체를
        죽여도 안전합니다. manager 밖에서 떠서 남의 세션/그룹(사용자 터미널
        등)에 속한 프로세스는 그 그룹을 건드리면 안 되므로, 목표 프로세스와
        그 자손만 죽입니다.
        """
        if os.name == "nt":
            args = ["taskkill", "/PID", str(pid), "/T"]
            if force:
                args.append("/F")
            subprocess.run(args, capture_output=True, creationflags=NO_WINDOW)
            return
        sig = signal.SIGKILL if force else signal.SIGTERM
        pgid = _pgid_of(pid)
        if pgid == pid:
            try:
                os.killpg(pid, sig)
                return
            except (OSError, ProcessLookupError):
                pass
        # 자손부터(가장 깊은 쪽) 시그널을 보낸 뒤 목표를 죽입니다.
        for child in sorted(_children_of(pid), reverse=True):
            try:
                os.kill(child, sig)
            except (OSError, ProcessLookupError):
                pass
        try:
            os.kill(pid, sig)
        except (OSError, ProcessLookupError):
            pass

    def _clear_run(self, pid):
        # Completion of an old stop must never erase a newer run's pidfile.
        if self.read_pidfile() == pid:
            try:
                os.remove(self._pidfile)
            except OSError:
                pass
        if self.pid == pid:
            self.pid = None
            self.started_at = None
            self._process = None
            self.device = None

    @staticmethod
    def _resolve_targets(pids):
        """명시적 종료 대상 PID 목록을 정규화합니다 (무효/중복 제거)."""
        targets = []
        for item in pids or []:
            try:
                item = int(item)
            except (TypeError, ValueError):
                continue
            if item > 0 and item not in targets:
                targets.append(item)
        return targets

    def _enter_stopping(self, targets):
        """종료 전 상태 전이를 lock 안에서 수행합니다.

        - 명시적 대상이 있고 어느 하나라도 살아 있으면 그 대상이 1호 번호가 됩니다.
          pidfile 없이 manager 밖에서 뜬 프로세스도 여기서 추적 대상이 됩니다.
        - 명시적 대상이 이미 모두 끝났으면 즉시 stopped 처리합니다.
        - 명시적 대상이 없으면 기존처럼 pidfile 기준으로 판단합니다.

        이전 종료 시도가 "stopping"으로 멈춰 있어도 재진입을 허용합니다:
        실패한 stop을 다시 누르면 재시도해야 교착에서 벗어납니다.
        """
        with self._lock:
            if targets:
                if not any(self._process_tree_alive(t) for t in targets):
                    for t in targets:
                        self._clear_run(t)
                    self.phase = "stopped"
                    return None
                self.pid = targets[0]
                if self.started_at is None:
                    self.started_at = time.time()
                self.generation += 1
                self.phase = "stopping"
                return targets[0]
            if not self.running():
                return None
            self.generation += 1
            self.phase = "stopping"
            return self.pid

    def stop(self, pids=None):
        """SIGTERM(→SIGKILL)으로 프로세스 트리를 종료합니다.

        pids를 주면 pidfile과 무관하게 해당 프로세스들(예: manager 밖에서
        직접 띄운 인스턴스)을 종료 대상으로 삼습니다.
        """
        targets = self._resolve_targets(pids)
        primary = self._enter_stopping(targets)
        if primary is None:
            return True if targets else False
        scope = targets or [primary]
        for target in scope:
            try:
                self._signal_tree(target, force=False)
            except (OSError, ProcessLookupError):
                pass
        if all(self._wait_for_exit(t, self.stop_timeout) for t in scope):
            self._finish_stop(scope)
            return True
        for target in scope:
            try:
                self._signal_tree(target, force=True)
            except (OSError, ProcessLookupError):
                pass
        if all(self._wait_for_exit(t, self.kill_timeout) for t in scope):
            self._finish_stop(scope)
            return True
        # If even SIGKILL could not be verified, retain PID + stopping.
        return False

    def force_kill(self, pids=None):
        """SIGKILL로 프로세스 트리를 즉시 종료합니다. pids는 stop()과 같습니다."""
        targets = self._resolve_targets(pids)
        primary = self._enter_stopping(targets)
        if primary is None:
            return True if targets else False
        scope = targets or [primary]
        for target in scope:
            try:
                self._signal_tree(target, force=True)
            except (OSError, ProcessLookupError):
                pass
        if all(self._wait_for_exit(t, self.kill_timeout) for t in scope):
            self._finish_stop(scope)
            return True
        return False

    def _finish_stop(self, scope):
        with self._lock:
            for target in scope:
                self._clear_run(target)
            self.phase = "stopped"


def tail(path, n=300):
    """파일 끝에서 n줄 읽기 (바이트 기반으로 안전하게)."""
    if not os.path.exists(path):
        return []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > 1_000_000:
                f.seek(size - 1_000_000)
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return lines[-n:]
    except Exception:
        return []


def find_process(pattern):
    """패턴(정규식, cmdline 매칭)에 해당하는 PID 목록 — 우리가 관리 안 하는 외부 프로세스 감지용.

    psutil 기반이라 Windows/Linux 모두 동일하게 동작합니다.
    """
    import re

    found = []
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
            except (psutil.AccessDenied, psutil.ZombieProcess, psutil.NoSuchProcess):
                continue
            if not cmdline:
                continue
            try:
                if re.search(pattern, cmdline):
                    found.append(proc.pid)
            except re.error:
                continue
    except Exception:
        return []
    return found
