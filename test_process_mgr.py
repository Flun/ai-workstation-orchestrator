import os
import subprocess
import sys
import threading
import time

import psutil

from process_mgr import Service


def _service(tmp_path, name="test-service"):
    service = Service(name, stop_timeout=0.25, kill_timeout=1)
    service.log_file = str(tmp_path / f"{name}.log")
    service._pidfile = str(tmp_path / f"{name}.pid")
    return service


def _spawn_external(tmp_path, script_body, name="test-service-ext"):
    """pidfile 없이, 자신의 세션/그룹이 아닌 프로세스 트리를 띄웁니다.

    manager가 만든 새 세션이 아니라 bash 래퍼(테스트 프로세스의 그룹에 속한)가
    자식으로 sleeper를 실행한 형태 — 터미널에서 직접 nohup으로 띄운 ComfyUI의
    그룹 관계와 동일합니다. (wrapper, child)를 반환합니다.
    """
    script = tmp_path / f"{name}.py"
    script.write_text(script_body, encoding="utf-8")
    wrapper = subprocess.Popen(
        ["bash", "-c", f"{sys.executable} {script} & wait"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            children = psutil.Process(wrapper.pid).children()
        except psutil.NoSuchProcess:
            return wrapper, None
        if children:
            return wrapper, children[0]
        time.sleep(0.02)
    return wrapper, None


def test_stop_is_available_immediately_after_spawn(tmp_path):
    service = _service(tmp_path)
    pid = service.start([sys.executable, "-c", "import time; time.sleep(30)"])

    assert service.phase == "starting"
    assert service.stop() is True
    assert service.phase == "stopped"
    assert not service.running()
    assert not psutil.pid_exists(pid)
    assert not os.path.exists(service._pidfile)


def test_stop_falls_back_to_sigkill_and_verifies_exit(tmp_path):
    service = _service(tmp_path)
    pid = service.start([
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ])
    time.sleep(0.1)

    assert service.stop() is True
    assert not psutil.pid_exists(pid)
    assert service.info()["phase"] == "stopped"


def test_start_stop_generation_prevents_overlapping_run(tmp_path):
    service = _service(tmp_path)
    service.start([
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ])
    time.sleep(0.1)
    first_generation = service.generation
    result = []

    stopper = threading.Thread(target=lambda: result.append(service.stop()))
    stopper.start()
    deadline = time.monotonic() + 1
    while service.phase != "stopping" and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        service.start([sys.executable, "-c", "import time; time.sleep(30)"])
        started_during_stop = True
    except RuntimeError:
        started_during_stop = False
    stopper.join(timeout=3)

    assert not started_during_stop
    assert result == [True]
    assert service.generation > first_generation

    second_pid = service.start([sys.executable, "-c", "import time; time.sleep(30)"])
    assert service.running()
    assert service.stop() is True
    assert not psutil.pid_exists(second_pid)


def test_stop_external_process_without_pidfile(tmp_path):
    """pidfile 없는 외부 프로세스도 명시적 pids로 종료됩니다.

    래퍼/목표는 테스트 프로세스의 프로세스 그룹에 속합니다(외부 nohup 형태).
    목적이 죽고도 테스트가 계속 진행되면, 남의 그룹이 통째로 죽지 않았다는
    뜻입니다(기존 killpg 방식이라면 이 테스트 프로세스 자체가 종료됐을 것입니다).
    """
    service = _service(tmp_path, name="test-service-ext")
    wrapper, child = _spawn_external(
        tmp_path, "import time\ntime.sleep(30)\n", name="ext-sleeper")
    assert child is not None and child.pid != _pgid_of_service(child)
    try:
        assert service.read_pidfile() is None
        # pidfile가 없는 상태에서 pids 없이 호출하면 기존 계약대로 False.
        assert service.stop() is False

        assert service.stop(pids=[child.pid]) is True
        assert service.phase == "stopped"
        assert not psutil.pid_exists(child.pid)
    finally:
        _cleanup(wrapper, child)


def test_force_kill_external_process_ignoring_sigterm(tmp_path):
    """외부 프로세스가 SIGTERM을 무시해도 SIGKILL 폴백으로 종료됩니다."""
    service = _service(tmp_path, name="test-service-ext2")
    wrapper, child = _spawn_external(
        tmp_path,
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(30)\n",
        name="ext-ignorer",
    )
    assert child is not None
    try:
        assert service.force_kill(pids=[child.pid]) is True
        assert service.phase == "stopped"
        assert not psutil.pid_exists(child.pid)
    finally:
        _cleanup(wrapper, child)


def test_stop_pids_already_dead_is_idempotent(tmp_path):
    """이미 끝난 대상에 대한 stop(pids)는 재시도/버튼 리클릭이 안 막히도록 True."""
    service = _service(tmp_path, name="test-service-ext3")
    wrapper = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wrapper.wait(timeout=10)
    assert service.stop(pids=[wrapper.pid]) is True
    assert service.phase == "stopped"


def _pgid_of_service(child):
    from process_mgr import _pgid_of
    return _pgid_of(child.pid)


def _cleanup(*procs):
    for proc in procs:
        if proc is None:
            continue
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass
