# 插件：接入你自己的内部系统

本仓库只带**通用能力**：知识库检索、源码检索、日志排查。
内部工单、订单、监控、日志平台这类公司特有系统，按下面的约定加一个工具即可，**不需要改框架**。

## 一个工具由三部分组成

在 `app/tools.py` 里：

```python
# ① 实现：任何异常都**不要向上抛**，转成文本回给模型
def query_ticket(ticket_id: str) -> str:
    if not ticket_id:
        return "请提供工单号。"
    try:
        data = _call_internal_api(ticket_id)
    except Exception:
        return "该数据暂时无法取到。"      # 对外话术：不暴露内部实现
    return f"工单 {ticket_id}：状态 {data['status']}，处理人 {data['owner']}"

# ② 注册（_SPECS）：描述写得越具体，模型越会用
_SPECS["query_ticket"] = (
    {"type": "function", "function": {
        "name": "query_ticket",
        "description": "按工单号查询工单状态与处理人。",
        "parameters": {"type": "object", "properties": {
            "ticket_id": {"type": "string", "description": "工单号"}},
            "required": ["ticket_id"]}}},
    lambda ticket_id="", **kw: query_ticket(ticket_id),
    False,          # 第三个参数：是否必须有日志才可用
)

# ③ 一句话用途（进 system prompt，模型靠它"知道"有这个能力）
_HINTS["query_ticket"] = "按工单号查状态与处理人"
```

重启服务即可生效，模型会自动在 `tools:` 里看到它。

## 三条约定（踩过坑）

1. **失败只说“暂时无法取到”**。把内部系统名、通道、代理、主机写进返回文本，模型会原样转述给一线。
2. **不要在工具里拼接用户可控字符串去执行命令**。只读接口优先，写操作必须二次确认。
3. **返回要短**。工具结果会进上下文，几千字的原始 JSON 会把后续轮次挤爆；先摘要成关键字段。

## 参考：被移除的公司专用实现

原项目里有 7 个业务工具（订单查询、门店汇总、配置查询、外部单号反查、修单报文等），
因为绑定具体企业与业务规则，**未包含在本仓库**。它们都遵循上面的同一套结构，可按需自行实现。

## 目录约定

```
app/tools.py        工具实现与注册（单文件，便于 review）
app/knowledge.py    知识库检索
app/codesearch.py   源码检索
```
