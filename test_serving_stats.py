"""serving_stats 파서·델타 계산 회귀 테스트 (네트워크 불필요).

Run with: python -m unittest test_serving_stats -v
"""
import time
import unittest

import serving_stats as ss


VLLM_SAMPLE = """
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="m"} 1000.0
vllm:generation_tokens_total{engine="0",model_name="m"} 200.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 400.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1000.0
vllm:request_prefill_time_seconds_sum{engine="0",model_name="m"} 10.0
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:time_to_first_token_seconds_bucket{model_name="m",le="0.5"} 3.0
vllm:time_to_first_token_seconds_bucket{model_name="m",le="1.0"} 8.0
vllm:time_to_first_token_seconds_bucket{model_name="m",le="+Inf"} 10.0
vllm:request_success_total{finished_reason="stop",model_name="m"} 7.0
""".strip()

LLAMA_SAMPLE = """
llamacpp:prompt_tokens_total 500
llamacpp:prompt_seconds_total 2.5
llamacpp:tokens_predicted_total 120
llamacpp:tokens_predicted_seconds_total 4.0
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
""".strip()


class ParseTests(unittest.TestCase):
    def test_counter_and_labels(self):
        samples = ss.parse_prometheus(VLLM_SAMPLE)
        self.assertEqual(ss._sum_value(samples, "vllm:prompt_tokens_total"), 1000.0)
        self.assertEqual(ss._value(samples, "vllm:num_requests_running"), 2.0)
        self.assertEqual(ss._value(samples, "vllm:kv_cache_usage_perc"), 0.25)

    def test_comment_and_bad_lines_ignored(self):
        samples = ss.parse_prometheus("# TYPE x counter\nbroken line here\nx 1.5")
        self.assertEqual([(n, v) for n, _, v in samples], [("x", 1.5)])

    def test_percentile_interpolates(self):
        hist = ss._histogram(ss.parse_prometheus(VLLM_SAMPLE), "vllm:time_to_first_token_seconds")
        self.assertIsNotNone(hist)
        bounds, cum = hist
        self.assertEqual(bounds, [0.5, 1.0, float("inf")])
        p50 = ss._percentile(bounds, cum, 0.5)
        self.assertTrue(0.5 <= p50 < 1.0)
        self.assertIsNone(ss._percentile(bounds, [0.0, 0.0, 0.0], 0.5))

    def test_infinite_bucket_returns_prev_bound(self):
        # p100은 +Inf 버킷에 떨어지므로 직전 상한(1.0)을 반환해야 한다
        bounds, cum = [0.5, 1.0, float("inf")], [3.0, 8.0, 10.0]
        self.assertEqual(ss._percentile(bounds, cum, 0.99), 1.0)


class RateTests(unittest.TestCase):
    def setUp(self):
        ss._prev.clear()

    def test_first_poll_no_rates(self):
        self.assertEqual(ss._rates("k", {"a": 10.0}, 100.0), {})

    def test_delta_rate(self):
        ss._rates("k", {"a": 10.0}, 100.0)
        rates = ss._rates("k", {"a": 30.0}, 105.0)
        self.assertAlmostEqual(rates["a"], 4.0)

    def test_counter_reset_dropped(self):
        ss._rates("k", {"a": 100.0}, 100.0)
        self.assertNotIn("a", ss._rates("k", {"a": 5.0}, 102.0))

    def test_stale_window_dropped(self):
        ss._rates("k", {"a": 10.0}, 100.0)
        self.assertEqual(ss._rates("k", {"a": 30.0}, 100.0 + ss.MAX_WINDOW + 1), {})


class PollShapeTests(unittest.TestCase):
    """요청을 모킹해 poll 함수의 출력 형태와 속도 계산을 확인."""

    def test_vllm_metrics_shape(self):
        ss._prev.clear()
        server = {"key": "t1", "label": "T", "base_url": "http://x/v1"}
        with _mock_response(VLLM_SAMPLE):
            first = ss._poll_vllm(server, 1000.0)
            second = ss._poll_vllm(server, 1010.0)
        self.assertTrue(first["online"])
        self.assertEqual(first["model"], "m")
        # 첫 폴링: 델타 없음
        self.assertNotIn("prefill_tps", first["metrics"])
        # 두 번째(동일 값): 델타 0 → gen wall-clock은 0, prefill은 분모 0으로 미계산
        self.assertEqual(second["metrics"]["decode_tps"], 0.0)
        self.assertNotIn("prefill_tps", second["metrics"])
        self.assertEqual(second["metrics"]["running"], 2.0)

    def test_llama_without_metrics_endpoint(self):
        ss._prev.clear()
        server = {"key": "t2", "label": "T", "base_url": "http://x/v1"}
        with _mock_response(None, props={"model_alias": "mm"}):
            result = ss._poll_llama(server, 1000.0)
        self.assertTrue(result["online"])
        self.assertEqual(result["model"], "mm")
        self.assertIn("metrics_note", result)


class _mock_response:
    """requests.get을 엔드포인트별로 분기해 모킹. payload=None은 404."""

    def __init__(self, prom_payload, props=None):
        self.prom = prom_payload
        self.props = props or {}

    def __enter__(self):
        import unittest.mock as mock

        def fake_get(url, timeout=None):
            resp = mock.Mock()
            if url.endswith("/metrics"):
                if self.prom is None:
                    resp.status_code = 404
                    resp.text = ""
                else:
                    resp.status_code = 200
                    resp.text = self.prom
                resp.raise_for_status = lambda: None
            else:
                resp.status_code = 200
                resp.json = lambda: self.props
            return resp

        self._patch = mock.patch.object(ss.requests, "get", fake_get)
        self._patch.start()

    def __exit__(self, *exc):
        self._patch.stop()


if __name__ == "__main__":
    unittest.main()
