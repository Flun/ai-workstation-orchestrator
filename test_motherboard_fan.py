"""Regression checks using temporary settings and a mocked hardware helper.

Run with: python -m unittest test_motherboard_fan -v
"""
import ast
import atexit
import concurrent.futures
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import motherboard_fan as fan


class FanMappingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Path(temporary.name) / "settings.json"
        override = patch.object(fan, "SETTINGS_FILE", self.settings)
        override.start()
        self.addCleanup(override.stop)

    def save(self, uuid="gpu0", channel="pwm3", **extra):
        return fan.save_settings({"gpu_uuid": uuid, "channel_id": channel,
                                  "enabled": True, "fan_role": "gpu_hbm", **extra})

    def test_sequential_save_reload_and_update_preserve_other_gpu(self):
        first = self.save()["gpu_profiles"]["gpu0"]
        self.save("gpu1", "pwm4")
        self.assertEqual(fan.load_settings()["gpu_profiles"]["gpu0"], first)
        self.save("gpu1", "pwm5", min_percent=50)
        profiles = fan.load_settings()["gpu_profiles"]
        self.assertEqual(profiles["gpu0"], first)
        self.assertEqual(profiles["gpu1"]["channel_id"], "pwm5")
        self.assertEqual(profiles["gpu1"]["min_percent"], 50)

    def test_legacy_file_migrates_when_second_gpu_is_saved(self):
        legacy = copy.deepcopy(fan.DEFAULTS)
        legacy.update(gpu_uuid="gpu0", channel_id="pwm3", enabled=True)
        self.settings.write_text(json.dumps(legacy))
        self.save("gpu1", "pwm4")
        self.assertEqual(fan.load_settings()["gpu_profiles"]["gpu0"]["channel_id"], "pwm3")

    def test_stale_client_map_cannot_erase_or_revert_other_gpu(self):
        stale = self.save()
        self.save("gpu1", "pwm4")
        stale.update(channel_id="pwm5")
        saved = fan.save_settings(stale)
        self.assertEqual(saved["gpu_profiles"]["gpu1"]["channel_id"], "pwm4")
        self.assertEqual(saved["gpu_profiles"]["gpu0"]["channel_id"], "pwm5")

    def test_cpu_only_save_preserves_gpu_profiles(self):
        self.save()
        profiles = self.save("gpu1", "pwm4")["gpu_profiles"]
        saved = fan.save_settings({"cpu": {"enabled": True, "channel_id": "pwm1"}})
        self.assertEqual(saved["gpu_profiles"], profiles)

    def test_disabling_one_gpu_preserves_the_other(self):
        self.save()
        self.save("gpu1", "pwm4")
        self.save("gpu1", "pwm4", enabled=False)
        profiles = fan.load_settings()["gpu_profiles"]
        self.assertTrue(profiles["gpu0"]["enabled"])
        self.assertFalse(profiles["gpu1"]["enabled"])

    def test_duplicate_gpu_and_cpu_channels_rejected_without_writing(self):
        self.save()
        before = self.settings.read_bytes()
        for values in ({"gpu_uuid": "gpu1", "enabled": True, "channel_id": "pwm3"},
                       {"cpu": {"enabled": True, "channel_id": "pwm3"}},
                       {"gpu_uuid": "gpu0", "min_percent": 101}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fan.save_settings(values)
            self.assertEqual(self.settings.read_bytes(), before)

    def test_concurrent_saves_do_not_lose_updates(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda i: self.save(f"gpu{i}", f"pwm{i + 3}"), range(2)))
        self.assertEqual(set(fan.load_settings()["gpu_profiles"]), {"gpu0", "gpu1"})

    def controller(self):
        controller = fan.FanController()
        atexit.unregister(controller.close)
        controller.helper = Mock()
        controller.helper.request.side_effect = lambda request: {"percent": request.get("percent", 0)}
        conflict = patch.object(fan, "_fan_control_running", return_value=False)
        conflict.start()
        self.addCleanup(conflict.stop)
        return controller

    def test_runtime_controls_both_gpus_with_independent_temperatures(self):
        self.save()
        self.save("gpu1", "pwm4")
        controller = self.controller()
        controller.tick([{"uuid": "gpu0", "temp_memory": 40}, {"uuid": "gpu1", "temp_memory": 85}])
        requests = [call.args[0] for call in controller.helper.request.call_args_list]
        self.assertIn({"command": "set", "id": "pwm3", "percent": 40}, requests)
        self.assertIn({"command": "set", "id": "pwm4", "percent": 100}, requests)
        self.assertEqual(controller._profile_state["gpu:gpu0"]["last_percent"], 40)
        self.assertEqual(controller._profile_state["gpu:gpu1"]["last_percent"], 100)
        self.save("gpu1", "pwm4", enabled=False)
        controller.reconfigure()
        controller.helper.request.reset_mock()
        controller.tick([{"uuid": "gpu0", "temp_memory": 40}])
        self.assertTrue(controller._runtime["profiles"]["gpu:gpu0"]["active"])
        self.assertFalse(controller._runtime["profiles"]["gpu:gpu1"]["active"])

    def test_reconfigure_releases_replaced_channel(self):
        self.save()
        controller = self.controller()
        controller.tick([{"uuid": "gpu0", "temp_memory": 40}])
        self.save(channel="pwm5")
        controller.reconfigure()
        controller.helper.request.assert_any_call({"command": "reset", "id": "pwm3"})

    def endpoint(self, controller, snapshot):
        # Load only this route, avoiding app startup/services and real hardware.
        source = ast.parse(Path(fan.__file__).with_name("app.py").read_text())
        route = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name == "motherboard_fans_config")
        route.decorator_list = []
        from fastapi import HTTPException
        scope = {"save_motherboard_fan_settings": fan.save_settings,
                 "validate_motherboard_fan_settings": fan.validate_settings,
                 "motherboard_fan_controller": controller, "_gpu_tuning_snapshot": snapshot,
                 "HTTPException": HTTPException, "STATE": {"hw": {"gpus": []}}}
        exec(compile(ast.Module(body=[route], type_ignores=[]), "app.py", "exec"), scope)
        return scope["motherboard_fans_config"]

    def test_route_keeps_control_active_when_last_edited_gpu_disabled(self):
        self.save()
        self.save("gpu1", "pwm4")
        controller = Mock()
        route = self.endpoint(controller, Mock())
        route({"gpu_uuid": "gpu1", "enabled": False})
        controller.reset.assert_not_called()
        controller.reconfigure.assert_called_once()
        controller.tick.assert_called_once()

    def test_invalid_cmp_selection_does_not_change_saved_settings(self):
        self.save()
        before = self.settings.read_bytes()
        controller = Mock()
        route = self.endpoint(controller, Mock(return_value={"gpus": []}))
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            route({"gpu_uuid": "gpu1", "channel_id": "pwm4", "enabled": True,
                   "fan_role": "cmp170hx_hbm"})
        self.assertEqual(self.settings.read_bytes(), before)
        controller.tick.assert_not_called()

    def test_browser_sends_scoped_payloads_and_restores_selected_gpu(self):
        html = Path(fan.__file__).with_name("index.html").read_text()
        methods = html[html.index("                selectMotherboardFanGpu() {"):
                       html.index("                async setMotherboardFanMode(mode) {")]
        script = "const assert = require('node:assert/strict'); const methods = {" + methods + "};\n" + r'''
        (async () => {
            global.confirm = () => true;
            global.alert = message => { throw new Error(message); };
            const saved = {gpu_uuid:'gpu0', channel_id:'pwm3', enabled:true};
            const settings = {...saved, gpu_profiles:{gpu0:saved}, cpu:{enabled:false}};
            let payload;
            global.fetch = async (url, options) => {
                payload = JSON.parse(options.body);
                return {ok:true, json:async () => ({settings})};
            };
            const context = {...methods, motherboardFans:{settings},
                motherboardFanEdit:{gpu_uuid:'gpu0'}, motherboardCpuFanEdit:{enabled:false}};
            context.selectMotherboardFanGpu();
            assert.equal(context.motherboardFanEdit.channel_id, 'pwm3');
            context.motherboardFanEdit.channel_id = 'pwm5';
            assert.equal(saved.channel_id, 'pwm3');
            await context.saveMotherboardFanConfig();
            assert.equal(payload.channel_id, 'pwm5');
            assert.equal('cpu' in payload, false);
            assert.equal('gpu_profiles' in payload, false);
            await context.saveMotherboardCpuFanConfig();
            assert.deepEqual(payload, {cpu:{enabled:false}});
        })().catch(error => { console.error(error); process.exit(1); });
        '''
        subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
