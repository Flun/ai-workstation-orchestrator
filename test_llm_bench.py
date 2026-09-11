"""llm_bench 모듈 스모크 테스트: 가짜 OpenAI 호환 서버(llama.cpp 흉내)로 quick/detailed 실행."""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import llm_bench

llm_bench.STORE_FILE = os.path.join(tempfile.mkdtemp(), "llm_bench.json")

REQUEST_LOG = []
COUNTER = {"prompt_tokens": 0.0}  # vLLM 흉내: 프로세스 생애 누적 카운터 (/metrics 노출)
COUNTER_LOCK = threading.Lock()


class FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj, stream_body=None):
        if stream_body is not None:
            body = stream_body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": "test-llama:latest"}]})
        if self.path == "/props":
            return self._json(200, {
                "model_path": "/models/test.gguf",
                "default_generation_settings": {"n_ctx": 32768, "n_predict": -1},
            })
        if self.path == "/metrics":
            body = f"# TYPE vllm:prompt_tokens_total counter\nvllm:prompt_tokens_total {COUNTER['prompt_tokens']}\n"
            raw = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        return self._json(404, {"error": "no"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/tokenize":
            text = body.get("content") or body.get("prompt") or ""
            cnt = max(1, len(text) // 4)
            return self._json(200, {"tokens": [1] * min(cnt, 16), "token_length": cnt})
        if self.path.endswith("/v1/completions"):
            prompt = body.get("prompt", "")
            max_tokens = int(body.get("max_tokens") or 1)
            p_tok = max(1, len(prompt) // 4)
            REQUEST_LOG.append({"stream": body.get("stream"), "max_tokens": max_tokens,
                                "min_tokens": body.get("min_tokens"), "prompt_len": len(prompt)})
            with COUNTER_LOCK:
                COUNTER["prompt_tokens"] += p_tok
            if body.get("stream"):
                parts = []
                for i in range(max_tokens):
                    ch = {"id": "x", "object": "completion.chunk",
                          "choices": [{"index": 0, "text": "tok ", "finish_reason": None if i < max_tokens - 1 else "length"}]}
                    parts.append("data: " + json.dumps(ch) + "\n\n")
                usage = {"id": "x", "choices": [], "usage": {"prompt_tokens": p_tok, "completion_tokens": max_tokens}}
                parts.append("data: " + json.dumps(usage) + "\n\n")
                parts.append("data: [DONE]\n\n")
                return self._json(200, None, stream_body="".join(parts))
            time.sleep(0.01)
            return self._json(200, {"id": "x", "choices": [{"text": "hi", "finish_reason": "length"}],
                                    "usage": {"prompt_tokens": p_tok, "completion_tokens": max_tokens}})
        return self._json(404, {"error": "no"})


server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
api = f"http://127.0.0.1:{port}/v1"

llm_bench.set_service_discovery(lambda: [{"key": "llama", "label": "llama.cpp (main_server)", "base_url": api, "engine": "llama.cpp"}])

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
app.include_router(llm_bench.router)
client = TestClient(app)

r = client.get("/api/llm-bench")
assert r.status_code == 200, r.text
assert r.json()["cards"] == []

r = client.get("/api/llm-bench/services")
assert r.json()["services"][0]["key"] == "llama", r.text

r = client.post("/api/llm-bench/cards", json={"title": "테스트 카드"})
card = r.json()["card"]
assert card["title"] == "테스트 카드" and card["created_at"]
cid = card["id"]

r = client.patch(f"/api/llm-bench/cards/{cid}", json={"title": "수정된 제목", "api_url": api, "source": "auto", "source_label": "llama.cpp (main_server)"})
assert r.json()["card"]["title"] == "수정된 제목"
assert r.json()["card"]["api_url"].endswith("/v1")

r = client.post(f"/api/llm-bench/cards/{cid}/probe", json={})
probe = r.json()["probe"]
assert probe["ok"], probe
assert probe["engine"] == "llama.cpp", probe
assert probe["max_context"] == 32768, probe
assert probe["model"] == "test-llama:latest"

r = client.post("/api/llm-bench/run", json={"card_id": cid, "kind": "quick"})
assert r.status_code == 200, r.text
job_id = r.json()["job_id"]
for _ in range(100):
    job = client.get(f"/api/llm-bench/jobs/{job_id}").json()
    if job["status"] != "running":
        break
    time.sleep(0.1)
assert job["status"] == "done", job
st = job["stages"][0]
assert st["pp_tok_s"] > 0 and st["decode_tok_s"] > 0 and st["ttft_s"] is not None, st
assert "pp1_tok_s" not in st and "pp2_tok_s" not in st and "warm_pp_tok_s" not in st, st  # 2회/동일프롬프트 측정 폐기 확인
assert any(x.get("min_tokens") == 128 for x in REQUEST_LOG if x["stream"]), REQUEST_LOG
# 단계당 요청은 정확히 1개(단일 스트리밍) — 동일 테스트 중복 측정 없음
assert len(REQUEST_LOG) == 1, REQUEST_LOG
# 첫 가동 cold 플래그: /metrics 누적 카운터가 0이었으므로 이 실행은 cold로 기록
first_res = next(x for x in client.get("/api/llm-bench").json()["results"] if x["kind"] == "quick")
assert first_res["meta"]["server_cold"] is True, first_res["meta"]

# N명 동시 배치: users=3 quick — 단일 1회 + 동시 3회 = 요청 4개만 추가
req_n = len(REQUEST_LOG)
r = client.post("/api/llm-bench/run", json={"card_id": cid, "kind": "quick", "users": 3})
job_nid = r.json()["job_id"]
for _ in range(200):
    jobn = client.get(f"/api/llm-bench/jobs/{job_nid}").json()
    if jobn["status"] != "running":
        break
    time.sleep(0.1)
assert jobn["status"] == "done", jobn
assert len(REQUEST_LOG) - req_n == 4, REQUEST_LOG[req_n:]
stn = jobn["stages"][0]
conc = stn.get("concurrent")
assert conc and conc["ok"], stn
assert conc["users"] == 3 and conc["completed"] == 3, conc
assert conc["per_user_decode_tok_s"] > 0 and conc["total_decode_tok_s"] > 0, conc
assert jobn.get("users") == 3
# 두 번째 실행부터는 트래픽이 있으므로 일반 프리필(server_cold False)
second_res = [x for x in client.get("/api/llm-bench").json()["results"] if x["meta"]["users"] == 3][0]
assert second_res["meta"]["server_cold"] is False, second_res["meta"]

# 동시 실행 거부 확인 (두 번째 실행이 도는 동안) — quick이라 금방 끝나므로 detailed 시작 직후에만 유효:
r = client.post("/api/llm-bench/run", json={"card_id": cid, "kind": "detailed"})
job_id2 = r.json()["job_id"]
r2 = client.post("/api/llm-bench/run", json={"card_id": cid, "kind": "quick"})
assert r2.status_code == 409, r2.text
for _ in range(400):
    job2 = client.get(f"/api/llm-bench/jobs/{job_id2}").json()
    if job2["status"] != "running":
        break
    time.sleep(0.1)
assert job2["status"] == "done", job2
stages = job2["stages"]
targets = [s["target_tokens"] for s in stages]
assert targets == [1024, 2048, 4096, 8192, 16384, 32768, 65536], targets
skipped = [s for s in stages if s.get("skipped")]
assert [s["target_tokens"] for s in skipped] == [32768, 65536], [ (s["target_tokens"], s.get("note")) for s in stages ]

snap = client.get("/api/llm-bench").json()
assert len(snap["results"]) == 3, len(snap["results"])  # quick, quick×3명, detailed
res = snap["results"][0]  # 최신(detailed)
assert res["kind"] == "detailed"
assert res["meta"]["engine"] == "llama.cpp" and res["meta"]["max_context"] == 32768
assert res["meta"]["model"] == "test-llama:latest"
assert "llama.cpp" in res["meta"]["serving"], res["meta"]["serving"]
assert res["stages"][0]["pp_tok_s"] > 0 and res["stages"][0]["decode_tok_s"] > 0
assert res["meta"]["users"] == 1 and res["meta"]["server_cold"] is False
assert res["summary"]["pp_tok_s"] is not None
conc_res = next(x for x in snap["results"] if x["meta"]["users"] == 3)
assert conc_res["summary"]["concurrent"]["users"] == 3, conc_res["summary"]

rid = res["id"]
assert client.delete(f"/api/llm-bench/results/{rid}").json()["ok"]
assert len(client.get("/api/llm-bench").json()["results"]) == 2

assert client.delete(f"/api/llm-bench/cards/{cid}").json()["ok"]
snap = client.get("/api/llm-bench").json()
assert snap["cards"] == [] and snap["results"] == []

server.shutdown()
print("SMOKE OK — quick/detailed/메타데이터/건너뜀/동시대기억부/삭제 전부 통과")
