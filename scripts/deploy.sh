#!/usr/bin/env bash
# 通用部署脚本：rsync 同步到远端并用 systemd 重启服务。
#
# 用法：
#   export DEPLOY_HOST=user@your-server      # 必填
#   export DEPLOY_DIR=/opt/ask-ops           # 可选，默认 /opt/ask-ops
#   export DEPLOY_SERVICE=ask-ops            # 可选，默认 ask-ops
#   bash scripts/deploy.sh
#
# 前置：远端已建好目录、装好依赖、配好 systemd unit（见 docs/DEPLOY.md）。
set -euo pipefail

: "${DEPLOY_HOST:?请先设置 DEPLOY_HOST，例如 user@1.2.3.4}"
DIR="${DEPLOY_DIR:-/opt/ask-ops}"
SVC="${DEPLOY_SERVICE:-ask-ops}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "▶ 同步代码到 ${DEPLOY_HOST}:${DIR}"
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude 'node_modules' \
  --exclude '__pycache__' --exclude 'data/sessions' --exclude 'data/uploads' \
  --exclude 'data/code_index.pkl*' --exclude 'logs' --exclude '.env' \
  "${ROOT}/" "${DEPLOY_HOST}:${DIR}/"

echo "▶ 重启服务 ${SVC}"
ssh "${DEPLOY_HOST}" "sudo systemctl restart ${SVC} && sleep 2 && systemctl is-active ${SVC}"

echo "▶ 健康检查"
ssh "${DEPLOY_HOST}" "curl -fsS -m 8 http://127.0.0.1:\${KB_PORT:-9000}/healthz || true"
echo
echo "✅ 完成"
