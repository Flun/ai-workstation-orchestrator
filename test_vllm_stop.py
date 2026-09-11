from pathlib import Path

import app
from process_mgr import Service


def _isolated_service(tmp_path):
    service = Service("vllm-stop-test", stop_timeout=0.1, kill_timeout=0.1)
    service.log_file = str(tmp_path / "vllm.log")
    service._pidfile = str(tmp_path / "vllm.pid")
    return service


def test_stale_cidfile_is_removed_and_not_reported_running(tmp_path, monkeypatch):
    service = _isolated_service(tmp_path)
    cidfile = tmp_path / "vllm-stale.cid"
    cidfile.write_text("a" * 64)
    service.container_id_file = str(cidfile)
    monkeypatch.setitem(app.services, "vllm", service)
    monkeypatch.setattr(app, "_docker_container_active", lambda _container_id: False)
    monkeypatch.setattr(app, "_docker_vllm_container_ids", lambda: set())
    monkeypatch.setattr(app, "_vllm_detected_pids", lambda: set())

    assert app._vllm_container_ids() == set()
    assert not Path(cidfile).exists()


def test_stop_succeeds_when_runtime_is_already_gone(tmp_path, monkeypatch):
    service = _isolated_service(tmp_path)
    monkeypatch.setitem(app.services, "vllm", service)
    monkeypatch.setattr(app, "_vllm_container_ids", lambda: set())
    monkeypatch.setattr(app, "_vllm_detected_pids", lambda: set())
    monkeypatch.setattr(app, "_vllm_listener_pids", lambda: set())
    monkeypatch.setattr(app, "_stop_vllm_systemd_unit", lambda: None)

    assert app._vllm_stop(force=True) is True
    assert service.phase == "stopped"
