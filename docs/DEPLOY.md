# 部署说明

三部分：本机服务 → 隧道 → 网关。任何一部分都可替换成你自己的方案。

## 1. 本机服务（systemd）

```ini
# /etc/systemd/system/ask-ops.service
[Unit]
Description=AskOps AI assistant
After=network.target

[Service]
Type=simple
User=ops
WorkingDirectory=/opt/ask-ops
EnvironmentFile=/opt/ask-ops/.env
ExecStart=/opt/ask-ops/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 9000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now ask-ops
curl -s localhost:9000/healthz
```

## 2. 隧道（Cloudflare Tunnel）

```bash
cloudflared tunnel create askops
cloudflared tunnel route dns askops ai.example.com
```

```yaml
# ~/.cloudflared/config.yml
tunnel: <TUNNEL_ID>
credentials-file: /home/ops/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: ai.example.com
    service: http://127.0.0.1:9000
  - service: http_status:404
```

```bash
sudo cloudflared service install && sudo systemctl enable --now cloudflared
```

> 没有 Cloudflare？任何能把 `ai.example.com` 反代到 `127.0.0.1:9000` 的方案都行（frp / nginx / 内网穿透）。

## 3. 网关（Cloudflare Worker）

```bash
cd worker
npm i -g wrangler        # 或 npx wrangler
wrangler login
# 改 wrangler.toml 里的域名与 KB_ORIGIN
wrangler secret put GATEWAY_API_KEY
wrangler deploy
```

部署后访问 `https://ask.example.com`，首次进入需要访问码（由服务端 `KB_ACCESS_CODE` 或邀请码控制）。

## 4. 日常更新

```bash
DEPLOY_HOST=ops@your-server bash scripts/deploy.sh
```

脚本会 rsync 同步代码（不覆盖 `.env`、会话与索引）、重启 systemd 服务、跑一次健康检查。

## 排障

| 现象 | 检查 |
|---|---|
| 网关 503 `offline` | 隧道没起 / `KB_ORIGIN` 写错 / 本机服务没监听 |
| 首页 401 且返回登录页 | Worker 门禁生效，正常；先完成访问码验证 |
| 问答很慢 | 看 `KB_AGENT_TIME_BUDGET`、模型端点延迟；把 `KB_AGENT_TRACE=true` 打开看每轮工具调用 |
| 检索总是不命中 | 确认 `data/kb/` 有内容，重启服务重建索引 |
