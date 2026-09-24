#!/usr/bin/env bash
# 외부 LLM 벤치 도구 준비 — lm-evaluation-harness와 llm-inference-bench를
# 메인서버 대시보드(bench_suite 페이지)에서 서브프로세스로 실행할 수 있게
# bench_tools/ 아래에 클론 + 전용 venv를 만든다. 재실행하면 pull/업데이트.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

for repo in EleutherAI/lm-evaluation-harness local-inference-lab/llm-inference-bench; do
  name=$(basename "$repo")
  if [ -d "$name/.git" ]; then
    echo "== $name: git pull"
    git -C "$name" pull --ff-only || echo "   (pull 실패 — 기존 버전 유지)"
  else
    echo "== $name: clone"
    git clone --depth 1 "https://github.com/$repo" "$name"
  fi
done

if [ ! -x .venv/bin/python ]; then
  echo "== venv 생성"
  "$PY" -m venv .venv
fi
.venv/bin/pip -q install -U pip
# llm-inference-bench 런타임(README 명시) + hw 샘플러용 psutil
.venv/bin/pip -q install httpx rich psutil
# lm-eval 코어 + OpenAI 호환 API 백엔드(api) + ifeval 채점기 + 로컬 HF 토크나이저(transformers)
.venv/bin/pip -q install -e "./lm-evaluation-harness[api,ifeval]" transformers

mkdir -p runs
echo
echo "완료. 대시보드 /bench-suite 페이지의 새로고침으로 인식됩니다."
.venv/bin/python - <<'EOF'
import importlib.util, pathlib
ok_lib = pathlib.Path("llm-inference-bench/llm_decode_bench.py").is_file()
ok_lme = pathlib.Path(".venv/bin/lm-eval").is_file()
print(f"  llm-inference-bench: {'OK' if ok_lib else '실패'}")
print(f"  lm-evaluation-harness: {'OK' if ok_lme else '실패'}")
EOF
