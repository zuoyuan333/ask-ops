#!/usr/bin/env bash
# 本机启动脚本（macOS 可双击运行）。读取同目录 .env 后启动 FastAPI 服务。
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
else
  echo "⚠️ 未找到 .env，将使用默认配置（LLM_PROVIDER=echo，不调用模型）"
fi

PY="${PYTHON:-python3}"
HOST="${KB_HOST:-127.0.0.1}"
PORT="${KB_PORT:-9000}"

echo "▶ 启动 AskOps：http://${HOST}:${PORT}"
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
