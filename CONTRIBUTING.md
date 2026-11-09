# 贡献指南

## 提交前

1. `python3 -m compileall app` 通过；
2. 本地起服务，`curl localhost:9000/healthz` 返回 `status: ok`；
3. 如果改了工具或提示词，用一条真实问题走一遍（`POST /ask`），确认没有回退成“无工具空答”。

## 代码约定

- Python 3.10+，只用标准库 + `requirements.txt` 里的依赖；
- 工具新增遵守 `docs/PLUGINS.md` 的签名约定，异常一律转文本返回，不向上抛；
- 提示词改动请说明**改之前哪里答错、改之后怎么验证**（这是这个项目最容易被改坏的部分）。

## 绝不提交

- 任何真实密钥、token、Cookie、内网地址、主机名；
- 真实业务数据：订单号、手机号、身份证、卡号、客户名称；
- 内部文档截图（含真实界面数据的）。

提交前可自查：

```bash
grep -rInE "sk-[A-Za-z0-9]{16,}|AKID[A-Za-z0-9]{10,}|1[3-9][0-9]{9}" --include='*' . | grep -v node_modules
```

## 提交信息

`feat: ...` / `fix: ...` / `docs: ...` / `refactor: ...`，一句话说清改了什么、为什么。
