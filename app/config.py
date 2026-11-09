"""本机知识库服务配置。

所有配置都支持环境变量覆盖（前缀 KB_），也可以在启动前 export。
建议：敏感项（API Key）不要写死在代码里，用环境变量或 .run/env 文件。
"""
import os
from pathlib import Path

# source/ 目录（本文件在 source/app/ 下，向上两级）
BASE_DIR = Path(__file__).resolve().parent.parent

# ---- 服务监听 ----
HOST = os.getenv("KB_HOST", "127.0.0.1")            # 只监听本机；对外暴露交给 cloudflared tunnel
PORT = int(os.getenv("KB_PORT", "9000"))

# ---- 知识库目录（把 .md / .txt 运维笔记丢进去即可，支持子目录）----
KB_DIR = Path(os.getenv("KB_KB_DIR", str(BASE_DIR / "data" / "kb")))
TOP_K = int(os.getenv("KB_TOP_K", "5"))             # 每次检索返回的片段数

# ---- LLM 提供方 ----
# echo  ：测试模式，不调用任何模型（自测用，返回检索结果摘要）
# ollama：本地模型（数据不出本机），默认 qwen2.5
# openai：任意 OpenAI 兼容端点（DeepSeek / 腾讯混元 / 硅基流动…）
LLM_PROVIDER = os.getenv("KB_LLM_PROVIDER", "echo")

# ollama 相关
OLLAMA_BASE = os.getenv("KB_OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("KB_OLLAMA_MODEL", "qwen2.5:14b")

# openai 兼容相关（2026-10-09 起走自建中转网关 https://ai.ctaigw.cn/v1，
# model=deepseek-v4.1-flash；旧 DeepSeek 官方端点 https://api.deepseek.com/v1 亦可）
# ⚠️ 旧模型名 deepseek-chat / deepseek-reasoner 已于 2026-07-24 停用，勿再使用。
# 腾讯混元 OpenAI 兼容: https://api.hunyuan.cloud.tencent.com/v1（走混元 API Key）
OPENAI_BASE = os.getenv("KB_OPENAI_BASE", "https://ai.ctaigw.cn/v1")
OPENAI_API_KEY = os.getenv("KB_OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("KB_OPENAI_MODEL", "deepseek-v4.1-flash")
# V4 系列默认思考模式会额外产出 reasoning token（更慢更费）；省流量场景置 disabled。
# 需要深度推理时可设 KB_OPENAI_THINKING=enabled（或 high/max 控制强度）。
# ⚠️ 网关侧 v4.1 模型**默认就思考**（实测不传 thinking 也会出 reasoning token），
#    ⇒ 「关思考」必须显式带 thinking:{type:disabled}；llm.py 两条路径都已显式传。
OPENAI_THINKING = os.getenv("KB_OPENAI_THINKING", "disabled")

# 出网代理（可选）。留空 = 直连，并**强制忽略**外部环境里的 http_proxy/https_proxy。
# ⚠️ 为什么默认直连：本机服务常被终端/自动化沙箱注入临时代理（http_proxy=127.0.0.1:<随机端口>），
#    该端口随会话结束即消失；httpx 默认 trust_env=True 会继承它，表现为调用大模型时
#    「Connection refused (Errno 61)」——服务本身没挂，是代理端口没了。
#    DeepSeek 在国内可直连；若你的网络确实必须走代理，设 KB_LLM_PROXY=http://127.0.0.1:10808
LLM_PROXY = os.getenv("KB_LLM_PROXY", "")

# ---- 回答篇幅安全网（去掉空白后计字符数）----
# 定位（2026-09-16 按主人要求调整）：篇幅**不是硬指标**，真正的目标是「去废话、看得懂」——
# 这件事已经在 prompt 里做成「输出前通读做减法」的显式动作，正常回答不会碰到这里。
# 这一层只负责拦「模型彻底写飞」（如把整份日志逐笔抄进答案写成几千字）：
# 超限就调一次模型压措辞（明令禁止删点击路径 / 参数名 / 参数值 / 编号步骤），
# 压不动就**保留原文** —— 宁可长，不可丢内容。
# ⚠️ 别把它当成优化器：实测首稿 1136 字时曾连压三轮、白烧 8 秒，而那一稿其实已经够清楚了。
# 设 0 = 关闭。
ANSWER_MAX_CHARS = int(os.getenv("KB_ANSWER_MAX_CHARS", "1400") or 0)

# ---- 回答简图（tools.draw_diagram）----
# 模型在合适的时候画一张简图（时间线 / 链路 / 判定分支），显示在答案最上方，替代大段文字。
# SVG 由本地模板渲染后落盘，前端按 /diagram/<name> 取（复用 /img 的模式）。
# ⚠️ 不把 SVG 内联进响应体：那会让前端 localStorage 里每条消息膨胀好几 KB。
DIAGRAM_DIR = Path(os.getenv("KB_DIAGRAM_DIR", str(BASE_DIR / "data" / "diagrams")))

# ---- 报障图片上传 ----
# 门店截图（报错弹窗等）存本机，再把「图片 id」随 /ask 一起送模型读图。
# 不直接走 base64 内联的原因：图片要能跨刷新保留、要能随「复制给人工」转交，
# 而且大图内联会把 /ask 请求体撑大。存本机 → 只有 id 进前端 localStorage（几十字节）。
UPLOAD_DIR = Path(os.getenv("KB_UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
UPLOAD_MAX_MB = int(os.getenv("KB_UPLOAD_MB", "5"))     # 单张上限（前端已压缩，这是兜底）
# 图片保留期（2026-11-02 主人定）：**独立于会话的 7 天**，给到 15 天。
# 依据：截图常被「复制给人工」带出去后，运维过几天才回头核对；会话 7 天到期后
# 图片就没人引用了，再留几天做缓冲，但也不能无限涨（一张几百 KB，一年就是几个 G）。
UPLOAD_TTL_DAYS = int(os.getenv("KB_UPLOAD_TTL_DAYS", "15"))

# ---- 会话存储（跨设备续聊，2026-09-22 主人定）----
# 一段报障对话落成 data/sessions/<sid>.json，换台电脑凭「会话 ID」就能接着问 / 继续排查。
# ⚠️ 定调是「轻量级、实时性助手」：**不做归档、不做冷备**，到期即硬删；
#    拉不到就如实告诉用户「会话不存在或已过期」。**不要再加"多留几天"这类旋钮。**
SESSION_DIR = Path(os.getenv("KB_SESSION_DIR", str(BASE_DIR / "data" / "sessions")))
SESSION_TTL_DAYS = int(os.getenv("KB_SESSION_TTL_DAYS", "7"))   # 保留期 —— 唯一的时间旋钮
SESSION_MAX_KB = int(os.getenv("KB_SESSION_MAX_KB", "2048"))    # 单会话上限（默认 2MB），超了从最旧消息截断
SESSION_GC_HOURS = int(os.getenv("KB_SESSION_GC_HOURS", "24"))  # 后台自动清理间隔
GC_DRY_RUN = os.getenv("KB_GC_DRY_RUN", "0") == "1"             # 空跑：只列清单不真删
SESSION_GC_LOG = Path(os.getenv("KB_SESSION_GC_LOG", str(BASE_DIR / "data" / "gc.log")))

# ---- 案例审查 & 教练记录（cases.py）----
# 每次问答的「已解决/未解决」结论落 cases.jsonl；人工在审查页写的纠正落 coach.jsonl。
# JSONL 而非数据库：量小、可直接打开看、坏一行不影响其它行、追加写天然免读改写。
CASES_FILE = Path(os.getenv("KB_CASES_FILE", str(BASE_DIR / "data" / "cases.jsonl")))
COACH_FILE = Path(os.getenv("KB_COACH_FILE", str(BASE_DIR / "data" / "coach.jsonl")))
COACH_TOP_K = int(os.getenv("KB_COACH_TOP_K", "2"))          # 每次提问最多注入几条教练记录
COACH_MIN_SCORE = float(os.getenv("KB_COACH_MIN_SCORE", "0.18"))  # 2-gram 相似度门槛（低于此不注入）

# ---- 网关鉴权 ----
# 设置了之后，Cloudflare Worker 转发请求时必须带 X-API-Key 头。
# 本机自测（localhost 直连）不受影响。
GATEWAY_API_KEY = os.getenv("KB_GATEWAY_API_KEY", "")

# ---- 访问认证（2026-09-22 主人定）----
# 背景：助手跑在总部管理员权限上，内部不需要权限分级，但要挡住**外部人员嗅探数据**。
# 现状问题：网关密钥硬编码在前端 JS 里，F12 就能看到 → 等于没有真正的访问控制。
# 方案：共享口令 → 签发**签名设备令牌** → 网关离线验签 + 设备标识绑定。
#      换浏览器 / 换设备 / 令牌过期 → 重新验证；全站生效（含页面本身）。
# ⚠️ 2026-09-22 晚改动：主人反馈「共享口令不好，每个用户都会来问我要」→ 改成**一人一码**。
# ⚠️ 2026-10-09 起**又加回一条共享长期授权码**（主人要放开内测，见下 KB_ACCESS_CODE）：
#    一人一码的码表改用于「邀请制」的 TOTP 身份；共享码 + 用户自报「员工编号」走另一条路。
ACCESS_CODE = os.getenv("KB_ACCESS_CODE", "")
# 共享**长期**授权码（2026-10-09 主人定：放开内测）—— 用户输「授权码 + 员工编号」即可进站。
# · 留空 = 关闭这条入口（回到只有 TOTP）。
# · 支持**逗号分隔多个**（便于平滑换码：先加新码、等大家都换过来、再去掉旧码）。
# · 「长期」= 不自动过期；要收紧就改这里（改完重启服务）。
# ⚠️ 码要够长（建议 >=16 位随机串）—— 它是长期口令、且是共享的，长短直接决定抗撞能力；
#    另外**别用 6 位纯数字**：那个长度会被判成 TOTP 动态码（见 main.auth_login 的分支判据）。
# ⚠️ 员工编号「只记录不校验」⇒ 承担的是**可追溯**，不是**防冒充**。要防冒充只能回 TOTP。
ACCESS_CODES_FILE = Path(os.getenv("KB_ACCESS_CODES_FILE",
                                   str(BASE_DIR / "data" / "access_codes.json")))   # 一人一码的码表
AUTH_AUDIT_FILE = Path(os.getenv("KB_AUTH_AUDIT_FILE",
                                 str(BASE_DIR / "data" / "auth_audit.jsonl")))      # 谁何时验证过的审计流水
# 邀请链接表（2026-09-22）：一条链接发到群里、大家自助领，24h 失效，可作废。
# 与"专属绑定链接"不同：邀请 token **不含 uid**，谁点谁领，各自拿到一份独立 TOTP 密钥。
INVITES_FILE = Path(os.getenv("KB_INVITES_FILE", str(BASE_DIR / "data" / "invites.json")))

# 令牌吊销名单（2026-09-25）：停用 / 删除用户时把 uid 记进来，
# 由 main.py 的中间件在**每个带身份的请求**上比对 —— 落实「停用即失效、离职即停」。
# ⚠️ 为什么必须有这份显式名单：设备令牌是**离线验签**的（网关只验签名、不回源查库），
#    所以"用户表里已经没有这个人了"并不能让他的令牌失效 —— 签名照样验得过。
REVOKED_FILE = Path(os.getenv("KB_REVOKED_FILE", str(BASE_DIR / "data" / "revoked_uids.json")))
AUTH_SECRET = os.getenv("KB_AUTH_SECRET", "")           # 签名密钥，必须与 Worker 的 AUTH_SECRET 一致
# 管理员列表（2026-09-24 P1）：管理类接口（案例库/教练记录/重载/清理）只允许这些人调用。
# 身份由**网关验签后注入** X-Auth-User 头（前端伪造无效，网关从不透传客户端同名头）。
# ⚠️ 2026-11-02：**uid 或员工编号 都可以写**（判定见 main._check_admin）。
#    默认 `u1,zuoyuan1` 双保险：u1 是左源的历史 uid，zuoyuan1 是他的员工编号 ——
#    哪天用户表重建、uid 变了，靠员工编号 仍然认得出是管理员。
#    ⚠️ 绝不要写 name：用户能自助改名（/auth/name），改名成管理员的名字就提权了。
ADMIN_UIDS = [s.strip() for s in os.getenv("KB_ADMIN_UIDS", "u1,zuoyuan1").split(",") if s.strip()]
AUTH_TTL_DAYS = int(os.getenv("KB_AUTH_TTL_DAYS", "30"))  # 设备令牌有效期（7→15→30，2026-10-12）
# 绑定链接有效期（2026-09-22 主人定：二维码想 1 天过期）
# ⚠️ 注意区分两个"有效期"：
#   · 设备令牌（AUTH_TTL_DAYS）—— 用户验证通过后，这台设备多久免验证；
#   · 绑定链接（AUTH_BIND_TTL_HOURS）—— 那条「扫码绑定」的链接多久内可用。
#   二维码本身不会过期（它只是密钥的载体），能过期的是**分发链接**。
#   让密钥过期会连已绑定的人一起失效 —— 所以只限分发窗口，不动密钥。
AUTH_BIND_TTL_HOURS = int(os.getenv("KB_AUTH_BIND_TTL_HOURS", "24"))
# 公网地址：管理脚本用它拼出可直接发给本人的**绑定链接**（形如 <base>/auth/bind/<token>）
AUTH_PUBLIC_BASE = os.getenv("KB_AUTH_PUBLIC_BASE", "https://ask.example.com").rstrip("/")
AUTH_COOKIE = os.getenv("KB_AUTH_COOKIE", "kb_device")   # 令牌 Cookie 名（HttpOnly）
AUTH_FP_COOKIE = os.getenv("KB_AUTH_FP_COOKIE", "kb_fp") # 指纹 Cookie 名（JS 可读写）

# ---- 源码查询模块（codesearch）----
# 排障/定位类问题会从源码仓库做离线索引检索，把「文件:行 + 代码片段」注入回答，
# 让 AI 基于真实源码作答。索引产物缓存在 CODE_INDEX_FILE，启动秒载，无需每次扫描。
# 开启开关：KB_CODE_ENABLED=1（默认开）。CODE_DIRS 用逗号分隔 src 目录；
# 留空 = 自动探测下方默认候选里「实际存在」的目录。
# 注意：索引构建需要读源码，构建时机=首次启动/显式 /code-rebuild；构建结果落盘后启动只需加载。
CODE_ENABLED = os.getenv("KB_CODE_ENABLED", "1") == "1"
CODE_AUTO_DIRS = [
    # ⭐ 2026-10-22 主人精简：只保留「当前主线版本」，删掉历史版本仓库，减少检索干扰。
    #    已移除：a8_5.7.x/src · a8_55x/src · a8_100x/src（小POS 历史版本）、elo_385/src（大POS 历史版本）
    #    ⚠️ a8_ccb / a8_388 / elo_389 / pos-web-react 本来就不在本列表里（文件同步了但从不索引）。
    "a8_new_ui/src",                                                 # 小 POS 主线（当前）
    "elo_cash/src", "elo/src",                                       # 大 POS 主线（当前）
    "cashsdk/cashier-sdk/src", "cashsdk/sdk/src",                    # 收银台 SDK
    "cashsdk/trade-cashdesk-h5/src",                                 # 收银台 H5（主源码 .jsx/React）
    "cashsdk/trade-cashdesk-h5-plate/src",                           # 码牌（uni-app + .vue）
    "posWeb/src",                                                  # web 后台
    "pos_ht",                                                      # 后台微服务（Java）⚠️ 勿删：
    # 没有它，「接口实现/根因/字段口径」类问题永远查不到后端（如 nextDayRefundCheck、
    # queryStoreIncomeAmount 都在 pos-crm 里）。整仓 3798 个 .java / 27.6 万行。
    "mySpace/toolkit/src",                                          # 运维工具箱（2026-10-08 加入）：
    # 知识库里多篇 SOP 直接指向 toolkit 的脚本（「修单脚本、上送脚本都在 toolkit/src 下」、
    # 《支付单边修复-小程序案例SOP》的取 token / 对账 / 修单 / 上送四件套），
    # 但此前该目录**只同步了文件、没进索引**，助手搜不到 → 只能照 KB 文字复述、无法核对代码。
    # 实测 137 个可索引文件 / 4.3 万行。同步脚本 sync-src-to-ops.command 的 `*/src/***`
    # 模式已覆盖该二级路径，无需额外改同步规则。
]
CODE_DIRS = [d for d in os.getenv("KB_CODE_DIRS", "").split(",") if d.strip()]
CODE_ROOT = os.getenv("KB_CODE_ROOT", "/Users/mac/工作/示例企业/code")
CODE_INDEX_FILE = Path(os.getenv("KB_CODE_INDEX_FILE", str(BASE_DIR / "data" / "code_index.pkl")))
# .java 于 2026-09-11 加入：此前只有前端，导致后端实现类问题 0 命中（详见 KB「隔日退」案例）
# .jsx/.tsx/.vue 于 2026-09-12 加入：收银台 H5 主源码是 .jsx（React，102 个），码牌是 .vue
# （uni-app，24 个）。不加则这两个仓库等于没入库——它们的 src 下几乎只有这两个后缀。
CODE_EXTS = {".ts", ".js", ".html", ".java", ".jsx", ".tsx", ".vue"}
# 命中行打分时，出现频率超过该值的词视为噪音（如 pay/order 等常见词），不计分。
# ⚠️ 该阈值必须随索引规模同比调整：只扫前端时总量 39 万行→2500；加入 pos_ht 后
# 总量约 67 万行，同比放大到 4500；2026-09-12 加收银台 H5 两仓（+约 1.6 万有效行）后
# 总量约 52.7 万行，同比例升至 4800。
# 2026-10-08 加入 mySpace/toolkit/src（+4.3 万行、约 +8.2%）后总量约 57.0 万行，
# 按同一比例升至 5200（4800 × 570107/526891 ≈ 5195，取整）。
# 注意：toolkit 侧多为中文命名的 .js/.html，常见词密度偏高，**下调此值会让这些文件
# 大面积判为噪音**，所以务必跟着规模走、不要凭感觉调。
CODE_DF_NOISE = int(os.getenv("KB_CODE_DF_NOISE", "5200"))
CODE_MAX_HIT_LINES = 40   # 总返回行数上限
CODE_MAX_LINES_PER_FILE = 6  # 同一文件最多返回几行（防单文件刷屏）
# 相对分数门槛：低于「最高分 × 该比例」的命中直接丢弃。
# 作用：真实命中往往断层领先（如 3955 vs 105），同一接口在多个仓库的前端副本
# 会以低分刷进结果列表，把 40 行名额占满。设 0.1 可砍掉这类噪声而不伤主命中。
CODE_SCORE_FLOOR_RATIO = float(os.getenv("KB_CODE_SCORE_FLOOR_RATIO", "0.1"))

# ---- 取证闭环（agent loop）：让模型自己决定查什么、查几轮（2026-09-16 上线）----
# 背景：此前 /ask 是「单向管道」—— 查什么由 Python 的 if/else 与阈值预先决定，模型只能在最后
# 一步改写挑好的素材，既没有取证权、也无法回头补查；表现就是「检索没命中 → 它照样照着编」。
# 实测把 知识库 / 源码 / 日志 包成工具交给模型自主取证后，同一条报障（报障卡 + 34.6 万字日志）：
#   耗时 15.1s → 8.1s；输入 token 2.8 万 → 1.13 万；答案 1099 字（超限）→ 978 字（达标）。
# 三项全面更优（因为「按需查」替代了「整段灌」，且不再一次性吐 1000 字），故替换为主路径。
# ⚠️ 置 0 可一键回退到原单轮流程（llm.answer / _build_messages）：原实现完整保留、一行未删。
AGENT_ENABLED = os.getenv("KB_AGENT_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")

# echo 测试模式不调用模型，Agent 循环没有意义 —— 自动关掉，保证首次运行能看到检索结果
if LLM_PROVIDER == "echo":
    AGENT_ENABLED = False
# 取证轮数上限。⚠️ 必须设限：模型拿到工具会一直查下去 —— 原型实测 6 轮仍不收口，最后把工具
# 原文当答案吐出来。末轮会**不带 tools** 强制收口（等价 tool_choice=none）；
# 光在 prompt 里写「够了就停」是无效的（同一份 prompt 下它照样连查 6 轮）。
# 2026-09-16 由 4 提到 5：加了 draw_diagram 之后，实测 4 轮会被「日志取证」吃满
# （log_overview + 3 次 grep），模型根本没轮次画图 —— 图永远不会出现，还白占一个工具位。
# 多留一轮给它画图，成本约 +1~2 秒；末轮仍是强制收口轮，「查不完」的风险不变。
AGENT_MAX_STEPS = int(os.getenv("KB_AGENT_MAX_STEPS", "5"))
# 单次工具返回的字符上限。⚠️ 这个值直接决定成本：原型 v1 每次返回 3500 字 ×10 次，
# 输入 token 冲到 5.86 万（比整段灌日志还贵）且始终不收口；压到 1300 字后降到 1.05 万。
TOOL_MAX_CHARS = int(os.getenv("KB_TOOL_MAX_CHARS", "1300"))
# 日志落盘目录：agent 的日志工具要「按需查」，日志就必须是个可查对象。
# 按内容 hash 存一份供人工复核（同一份日志重复提交只存一份）；按天自动清理。
AGENT_LOG_DIR = Path(os.getenv("KB_AGENT_LOG_DIR", str(BASE_DIR / "data" / "agent_logs")))
AGENT_LOG_KEEP_DAYS = int(os.getenv("KB_AGENT_LOG_KEEP_DAYS", "7"))
# 是否把本轮取证轨迹（调了哪些工具、各返回多少字）随响应回给前端，供运维复核答案是怎么来的。
AGENT_EXPOSE_TRACE = os.getenv("KB_AGENT_TRACE", "1").strip().lower() not in ("0", "false", "no", "")

# 源码检索词规范化开关（2026-10-05 加）。默认 1 = 规范化（小写 + 去分隔符 + 需要时拆词/拆驼峰）。
# ⚠️ 置 0 会回到「原样传词」的老行为 —— 而索引里的标识符是小写存的，所以
#    `PayMethodsKeyBoard` 这类写法会**零命中**（实测 0 vs 38 条）。留这个开关纯粹是为了
#    让评测集能**干净地隔离**这一项改动的贡献（`--set KB_CODE_NORM=0`），生产不要关。
CODE_NORM = os.getenv("KB_CODE_NORM", "1").strip().lower() not in ("0", "false", "no", "")

# ---------------- 排队等待上限（2026-10-10 评审 ⑧）----------------
# 动机：queue.acquire 原来无限期阻塞 —— 排队第 3 位要等 ~180s，而前端 150s 先爆
# → 用户看到「假超时」，后端还在跑、结果永远到不了（还占并发位）。
# 现在等不到就快速回一条带位次的人话。100s 依据：deep 中位 53s / p90 79s，
# 排 2 位最坏 ~160s —— 诚实告知「等太久」比假装在处理好。
QUEUE_WAIT_TIMEOUT = int(os.getenv("KB_QUEUE_WAIT_TIMEOUT", "100"))

# ---------------- LLM 重试总预算（2026-10-12 第三轮评审 P0）----------------
# ⚠️⚠️ 修补一个**真断裂**：单轮 timeout=180s × (2 重试 + 1) + 4s ≈ **544s**，
#   而 Worker 转发超时只有 210s、前端 240s ⇒ 某轮真挂住时，用户在 210s 看到断连，
#   **后端还会继续跑到 544s**（白占 1/3 并发位）。靠"网关平时快速拒绝"才没暴露。
# 修法：给**整条重试序列**一个总时间上限 —— "单次慢"仍允许（不误杀正常慢轮），
#   只禁止"慢 × 3 次叠加"撑爆外层超时。取 **180s** 依据（模拟验证 + 外层约束）：
#     · 必须 < Worker 转发 210s（Worker 从**请求开始**计时，含排队+前置，故留 30s 余量）
#     · 单次 ≤30s → 3 次尝试（最坏 94s ✅）；单次 60s → 2 次（124s ✅）；
#       单次 ≥90s → 1 次（≤181s ✅）—— 即"已经慢到这个程度就不再叠加"
#   ⚠️ 关键是**第一次尝试无条件放行** —— 否则单次超时 > 预算时一次都不试，等于功能失效。
#   设 0 = 关闭预算检查（回旧行为）。
LLM_RETRY_BUDGET = int(os.getenv("KB_LLM_RETRY_BUDGET", "180") or 0)

# ---------------- deep 档整个问答的时间预算（2026-10-12 下午事故）----------------
# ⚠️ 实测：一次 deep 连查 15 次 query_dtp_to_pos（Mac 中转慢查询），总耗时 **230.3s**
#   —— 而 Worker 转发超时只有 210s ⇒ 用户在 210s 被 Cloudflare 断开、收到「服务不在线」，
#   **后端却跑完了并产出完整答案**（结果永远到不了用户手里，白烧 token）。
# 修法：取证循环里每轮结束都检查"已耗时"，超过预算就强制收口（基于已有证据作答）。
# ⭐ 取 100s（2026-10-12 第二次复盘后从 150 下调）：150 那版实测**仍然超时** ——
#   一次重放 178s 才触发预算（检查在"本轮工具全部跑完"之后，第 3 轮一口气查了 10 个
#   慢单号就冲过线了），收口 12.9s + 自检 24.7s ⇒ 总计 215.7s > 210s。
#   现在：①预算降到 100s；②预算收口场景**跳过自检轮**（已没时间了还复核什么）。
#   新的最坏链 = 100（触发点，实际可能到 ~120） + 收口 13 ≈ 133s ≪ 210s ✅
#   ⭐ 2026-10-14：又加了**精简轮**（-4~23s，随答案长度变），故预算 100 → 110：
#     110 + 收口 13 + 精简 23 ≈ 146 ≪ 210 ✅（自检轮本就在预算内，不另算）
#   正常 deep 中位 53s / p90 79s 不受影响（p90 也够不着 110s）。
#   ⚠️⚠️ 2026-10-14（修 524）**再改回 70s** —— 实测线上 **27.8% 请求超 100s**：
#     该预算只管「取证循环」，而阶段实测 取证 33 + 自检 14 + 精简 39 ≈ 86s（中位），
#     叠加后越过 **CF 免费版 100s 硬墙** ⇒ 524。故预算须显著低于 100s。
#     ⚠️ 真正兜底的是 `TOTAL_DEADLINE_SECS`（85s 总闸门）；本值只管"别把取证拖太久"。
#   设 0 = 关闭（回旧行为）。
AGENT_TIME_BUDGET = int(os.getenv("KB_AGENT_TIME_BUDGET", "70") or 0)

# ---------------- ⭐ 总时间闸门：防撞 Cloudflare 100s 墙（2026-10-14 修 524）----------------
# ⚠️⚠️ 为什么必须有它：`AGENT_TIME_BUDGET` 只管「取证循环」，取证收口后
#   自检轮（中位 14s）与精简轮（中位 39s）还会往上叠 —— 实测线上 **27.8% 请求超 100s**，
#   而 **CF 免费版边缘硬超时就是 100s** ⇒ 直接 524，用户拿到**彻底失败**（连答案都没有）。
# ⇒ 闸门语义：**已耗时超过本值，就不再启动自检/精简**（宁可交没复核的答案，也不撞 524）。
# ⚠️ 取 85s：留 15s 给"生成响应 + 经隧道回传 + CF 边缘转发"的余量。
# ⚠️ 必须 < 100（CF 墙），否则这条闸门形同虚设。
# ⚠️⚠️ 2026-10-14 二次修正：闸门**不只看"已耗时"，还要加上"该阶段预计耗时"** ——
#   因为精简轮自己就要跑 40s+（实测中位 38.6s）。第一版只看已耗时（≥85s 才跳），
#   结果"取证 40s + 自检 10s = 50s"没到线就放行，精简再跑 46s ⇒ 总 96s ⇒ **仍 524**。
#   ⇒ 现在：`已耗时 + EST_POLISH_SECS > TOTAL_DEADLINE_SECS` 就跳过精简。
TOTAL_DEADLINE_SECS = int(os.getenv("KB_TOTAL_DEADLINE_SECS", "85") or 0)

# 各阶段"预计耗时"（秒），用于上面那道闸门做前瞻判断。按线上实测**偏保守**取值：
#   自检轮 中位 14.1s / 均值 27.2s / max 146.7s ⇒ 取 25
#   精简轮 中位 38.6s / 均值 42.2s / max 151.1s ⇒ 取 50（⚠️ 曾取 40，实测有 46.9s 的题漏网）
# ⚠️ 取值宁可偏大：**误跳只是少一道复核，撞 524 是彻底失败（用户什么都拿不到）**。
EST_SELFCHECK_SECS = int(os.getenv("KB_EST_SELFCHECK_SECS", "25") or 0)
EST_POLISH_SECS = int(os.getenv("KB_EST_POLISH_SECS", "50") or 0)

# ⭐⭐ 总时长**硬上限**（2026-10-14 修 524 的真正兜底）
# ⚠️ 与 `TOTAL_DEADLINE_SECS` 的分工：
#   · `TOTAL_DEADLINE_SECS`（软闸门）= 取证结束后，判断"还要不要做自检/精简"；
#   · 本值（硬上限）= **从请求开始计时的绝对上限**，任何阶段发现自己超了就立即收口。
# ⚠️⚠️ 为什么还要硬上限：上面的软闸门**管不到"取证轮"本身** ——
#   实测 A03 取证就跑了 40.4s（第 5 轮单轮 17s），而软闸门只在取证**结束后**才判断。
#   若取证本身拖到 90s，软闸门再怎么跳也来不及。
# ⇒ 硬上限作用于**取证循环每一轮**：超过就强制收口（与 `AGENT_TIME_BUDGET` 同机制，
#   但本值更"绝对"、不受单号数量加成影响）。
# ⚠️ 取 75s：给"收口轮(13s) + 响应回传(15s)"留位，确保 < CF 的 100s 墙。
HARD_DEADLINE_SECS = int(os.getenv("KB_HARD_DEADLINE_SECS", "75") or 0)

# ---------------- 精简轮的最小输入（2026-10-14 修 524）----------------
# ⚠️ 精简轮实测**中位 38.6s / max 151s**，是整条链路最贵的一步，只减 200~400 字。
# ⇒ **短答案不值得为它花 40 秒**（还冒 524 风险）。低于本值直接跳过精简。
# 设 0 = 不设下限（任何长度都精简）。
POLISH_MIN_CHARS = int(os.getenv("KB_POLISH_MIN_CHARS", "500") or 0)

# ---------------- 一轮内工具并发数（2026-10-12）----------------
# ⚠️ 动机：模型一轮常同时发多个独立查询（如 20 个 EXT 单号反查），原来是**串行**执行
#   ⇒ 20 × ~9s = 180s（超预算、超 Worker 210s ⇒ 用户"白等"）。
# 实测（小主机侧，8 个 EXT 反查）：**串行 72.3s vs 并发(5) 16.7s —— 快 4.3 倍**。
# ⚠️ 5 的依据：对端是 Mac 中转代理 + els 查询，并发过高可能触发对端限流；
#   5 实测稳定且提速显著。设 1 = 关闭并发（回串行）。
AGENT_TOOL_WORKERS = int(os.getenv("KB_AGENT_TOOL_WORKERS", "5") or 0)

# ---------------- 精简轮：最终答案前的「废话审查」（2026-10-14 主人要求）----------------
# ⚠️ 动机：主人反馈「回答还是太多了」。此前只有 prompt 里的「输出前通读做减法」和
#   `_enforce_max_chars` 的**字数安全网**在管篇幅，但前者靠模型自觉、后者只拦写飞。
# 做法：在**最终消息产出前**加一轮模型调用，专门删「对用户没用的文字」
#   （复述问题 / 自我描述 / 客套过渡 / 重复表述 / 不必要实现细节），
#   **保留**所有可执行内容（路径/参数/单号/证据行/归属方/诚实声明）。
# ⚠️ 只做减法、不改写不重排；输出过短或三段式/步骤变少 → 回退原文（护栏）。
# 设 0 = 关闭（回旧行为）。
POLISH_ENABLED = bool(int(os.getenv("KB_POLISH_ENABLED", "1") or 0))
# 精简轮的思考档：默认 high —— 判断"哪句是废话"需要理解用户处境，属推理活。
THINKING_POLISH = os.getenv("KB_THINKING_POLISH", "high").strip() or "high"

# ---------------- 答案草稿送入"改写型轮次"的字符上限（2026-10-14）----------------
# ⚠️⚠️ 这是**防静默丢内容**的关键闸门（主人 2026-10-14 质疑暴露）：
#   自检轮 / 精简轮都是「读草稿 → 写回答案」的结构。若把草稿**截断**送入，
#   模型输出的"改写版"必然缺后段，而它自己并不知道 ⇒ 替换后**静默丢内容**，
#   且现有护栏（按原文总长判"删过头"）会因阈值被抬高而放行。
# ⇒ 铁律：`len(draft) > DRAFT_IN_MAX_CHARS` 时**整轮跳过**（宁可不精简，绝不丢内容）。
# ⚠️ 该值必须**大于** `ANSWER_MAX_CHARS`（1400），否则精简轮/自检轮几乎永远不执行。
#   取 8000：是 ANSWER_MAX_CHARS 的 5.7 倍，正常答案（1400~2200 字）远够不着。
DRAFT_IN_MAX_CHARS = int(os.getenv("KB_DRAFT_IN_MAX_CHARS", "8000") or 0)

# ---------------- 线程池上限（2026-10-12 第三轮评审 P1）----------------
# ⚠️ 背景：`/ask` 是**同步函数** ⇒ FastAPI 用 AnyIO 线程池跑（**默认 40**）。
#   而排队（queue.acquire）是**阻塞占一个线程**，不是轻量等待
#   ⇒ 真实容量 = "3 个在跑 + 最多 37 个在等"，第 38 个请求才被拒。
#   更麻烦的是：37 个线程被排队请求占满后，**其他同步接口**（/cases、/admin/*、
#   /upload）也拿不到线程 ⇒ 管理页会跟着卡。
# 取 10 的依据：> MAX_CONCURRENT_ASK(3) 保证"在跑的"排得上；留 7 个给其他接口与
#   短请求。宁可**快速回绝**（已有 queued_out 人话）也不要"默默占线程等 100 秒"。
# ⚠️ 设 0 = 不改（沿用 anyio 默认 40）。
THREAD_POOL_SIZE = int(os.getenv("KB_THREAD_POOL_SIZE", "10") or 0)

# ---------------- 分档：fast / deep（2026-10-05）----------------# 背景（实测 95 个会话 / 389 条轮次日志）：**59% 的会话撞满 5 轮上限被强制收口**，
# 且前 4 轮 out 只有 100~190 token（几乎全是工具调用，没有思考）—— 深度不足是「预算」问题。
# 主人定调：模型仍用 v4 flash 系（2026-10-09 起 = deepseek-v4.1-flash），靠「预算 + 编排」补深度，不引入多智能体。
# 机制：`llm._classify_depth()` 用**本地规则**（零成本、确定性）判定档位，两档各给不同
#      轮数 / 检索宽度 / 思考模式；deep 档再叠加「规划轮」与「自检轮」。
# ⚠️ 把 _FAST 与 _DEEP 设成同一个值 = 等于关闭分档，回到单档行为（便于回滚）。
AGENT_MAX_STEPS_FAST = int(os.getenv("KB_AGENT_MAX_STEPS_FAST", str(AGENT_MAX_STEPS)))
AGENT_MAX_STEPS_DEEP = int(os.getenv("KB_AGENT_MAX_STEPS_DEEP", "12"))
TOP_K_FAST = int(os.getenv("KB_TOP_K_FAST", str(TOP_K)))
TOP_K_DEEP = int(os.getenv("KB_TOP_K_DEEP", "8"))
# 思考模式（DeepSeek V4 专属参数，见 llm._openai_payload）：
#   disabled=省流量；enabled/high/max=开思考（更准但更费、更慢）。只对 v4 系模型生效。
#   ⚠️ 这里 `enabled` 与 `high` 在代码里等价（都映射成 reasoning_effort=high）。
#   2026-10-09 主人定：deep 档 = high（高配，但不设 max 极致）；fast 档保持 disabled。
# ⭐ 2026-10-09 方案 A（主人批准）：**deep 档的中间取证轮不再用 high**。
#   实测依据：deep 一次问答 60~77s，而中间那几轮只是「决定下一步查什么工具」，
#   思考收益很低却每轮都要多等几秒 + 多烧思考 token；真正需要 high 的是**写最终答案**那一轮
#   和自检轮。⇒ 中间轮用 `KB_THINKING_DEEP_MID`，答案轮/自检轮仍用 `KB_THINKING_DEEP`。
#   实测网关档位梯度（同一问题、单次调用）：disabled 5.7s / minimal 13s / low 19.7s / medium 22.9s / high 24.1s
#   ⚠️ 想回退成"全程 high"：把 KB_THINKING_DEEP_MID 设成与 KB_THINKING_DEEP 相同的值即可。
THINKING_DEEP_MID = os.getenv("KB_THINKING_DEEP_MID", "minimal")
# 中间取证轮是否**强制调工具**（tool_choice=required，实测网关支持）。
# ⚠️ 为什么需要它：模型经常在**非最后一轮**就把答案写出来了（实测 8 轮上限时答案出在第 7 轮）。
#    若不强制，那一轮正好用的是"中间档思考"⇒ 答案会被低档思考产出，方案 A 就白做了。
#    强制取证后，"写答案"只会发生在末轮 / 提前收口轮 / 自检轮 —— 这三处都用高档思考。
#    代价：本来想提前收口的轮次会被推着再查一次（中间轮便宜，且有 stall 判据兜底，不会打转）。
#    设 0 = 关掉强制（回到 auto），用于对比验证。
AGENT_DEEP_REQUIRE_TOOL = os.getenv("KB_AGENT_DEEP_REQUIRE_TOOL", "1").strip().lower() not in ("0", "false", "no", "")
THINKING_FAST = os.getenv("KB_THINKING_FAST", OPENAI_THINKING or "disabled")
THINKING_DEEP = os.getenv("KB_THINKING_DEEP", "high")
# 规划轮：deep 档先让模型出一份「取证计划」再进工具循环（少东一榔头西一棒子）。
PLAN_ENABLED = os.getenv("KB_PLAN_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# ⭐ 2026-10-12 思考链路升级（主人："我搞到了免费的，是不是可以改一下思考链路之类的来提升智商"）。
#   原状：**规划轮与自检轮都写死 thinking=disabled** —— 而这两个恰恰是最需要推理的"元认知"环节
#   （规划=决定查什么、自检=判断有没有漏答），却用了最弱的思考档，属于明显错配。
#   现在：三档可调（还是那句"IQ 上限在思考档"）——
#     KB_THINKING_PLAN  规划轮思考档（默认 high；设 disabled 回旧行为）
#     KB_THINKING_SELF  自检轮思考档（默认 high；自检要"逐句拆解+判断证据"最吃推理）
#     KB_THINKING_ANSWER_SELF 自检发现缺口后的**修订轮**思考档（默认 high）
#   ⚠️ 改这三行等于提升"智商"，但**会增加耗秒**（思考 token 也要算时间）——
#     主人已确认"时间不是问题、要智商"，故默认开。
THINKING_PLAN = os.getenv("KB_THINKING_PLAN", "high")
THINKING_SELF = os.getenv("KB_THINKING_SELF", "high")
THINKING_SELF_FIX = os.getenv("KB_THINKING_SELF_FIX", "high")
# ⭐ 2026-10-21（治 524）：**收口轮思考档**，默认 low。
#   收口轮 = "时间已到，把已查到的证据直接写成答案"，不需要深推理；
#   实测用 high 时单轮 34.4s，而它恰恰发生在已经超时的时刻 ⇒ 直接推过 CF 100s 墙。
#   ⚠️ 只作用于"时间到了"的收口轮（硬上限 / 预算 / 轮数用尽）；
#      「提前收口轮」（连续两轮没新证据、主动停）仍用 THINKING_DEEP，质量优先。
THINKING_CUT = os.getenv("KB_THINKING_CUT", "low")
# ⭐ 2026-10-12（评审逐行分析 P2-1）：**取证轮输出上限**（答案轮不限）。
# 动机：原先 payload 完全没有 max_tokens ⇒ 单轮输出无上限。而「时间 ≈ 输出 token ÷ 115」，
# 实测见过单轮 out=3104（那一轮光生成就 20+ 秒）。取证轮只需"选工具 + 一句理由"，
# 根本不该写 3000 字。取 2000 是**宽松**值：正常取证轮 100~300 token，
# 被截的只会是明显的"写飞"；设太小反而可能截断 tool_calls 的参数。设 0 = 关闭（回旧行为）。
# ⚠️⚠️ 2026-10-22 由 2000 提到 **3000**（**修一个正确性 bug**）：
#   原设计假设"只有最后一轮才写答案 ⇒ 那轮不设上限"，但 `_mt` 判的是
#   `step >= max_steps`（最后一轮），而**模型随时可以作答**
#   （`KB_AGENT_DEEP_REQUIRE_TOOL=0` 时中间轮是 `tool_choice=auto`）——
#   于是"提前作答"的答案**也被 2000 卡住**。
#   实测线上日志：15 次截断里 **14 次截断后运行就结束** ⇒ **答案被截半，用户拿到半截答案**。
#   实测答案轮输出分布：中位 619 / 均值 844 / **max 2608** ⇒ 取 3000 可覆盖全部观测到的答案。
#   取证轮实际输出：中位 138 / p90 571 / 仅 3% 超过 1000 ⇒ 上限的"防写飞"作用本来就只覆盖极少数轮，
#   不值得为它冒截断答案的风险。
AGENT_MID_MAX_TOKENS = int(os.getenv("KB_AGENT_MID_MAX_TOKENS", "3000") or 0)
# 背景：deep 档 55% 撞满 8 轮 —— 模型在循环里不知道自己一共几轮（只有末轮才被告知），
# 提示词说「够了就收口」但它感知不到「够了」的信号。关掉 = 回到旧行为（对比实验用）。
AGENT_BUDGET_REMIND = os.getenv("KB_AGENT_BUDGET_REMIND", "1").strip().lower() not in ("0", "false", "no", "")
# 2026-10-10 评审 ②：撞顶答案尾部标注（+ meta.rounds_capped）。关掉 = 回到无标注。
ROUNDCAP_NOTE = os.getenv("KB_ROUNDCAP_NOTE", "1").strip().lower() not in ("0", "false", "no", "")
# 2026-10-10 评审 ③'：文件级停滞判据（**默认关**，等 A/B 验证后再开）。
NEWFILE_STALL = os.getenv("KB_NEWFILE_STALL", "0").strip().lower() not in ("0", "false", "no", "")
# 2026-10-10 主人要求：答案末尾追加耗时（deep 必加；fast ≥10s 也加）。
ELAPSED_NOTE = os.getenv("KB_ELAPSED_NOTE", "1").strip().lower() not in ("0", "false", "no", "")
# 自检轮：deep 档答案产出后复核「用户问了几件事 / 每件是否都有证据」，缺口就补正。
SELFCHECK_ENABLED = os.getenv("KB_SELFCHECK_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# 2026-10-10（治 deep 空转）：把「知识库文档目录」注入 system prompt。
# 背景：线上实测 61% 的 deep 撞满 8 轮，轨迹显示模型反复换词搜「错误码汇总表 / 速查表 /
# 对照表 / 清单 / 大全」这类**库里根本不存在**的汇总型文档 —— 因为它看不到"库里有什么"。
# 关掉 = 回到旧行为（A/B 对照用：`--set KB_KB_CATALOG=0`）。
KB_CATALOG = os.getenv("KB_KB_CATALOG", "1").strip().lower() not in ("0", "false", "no", "")
# 2026-10-10（配套收敛判据）：纯检索轮 + 零新条目 + 在找「汇总型文档」⇒ 记停滞。
# 与 KB_CATALOG 配套（目录已告诉模型"没有汇总表"，它还硬搜就该被止损）。
CATALOG_STALL = os.getenv("KB_CATALOG_STALL", "1").strip().lower() not in ("0", "false", "no", "")

# ---------------- 上下文瘦身与收敛（2026-10-05）----------------
# 实测（deep 档一次问答）：累计 prompt 85k~98k，而**工具返回总量只有 6k~10k tokens** ——
# 也就是说成本大头是「每轮把历史重发一遍」，其中最占位的是早期那几条长工具结果。
# 同时 deep 档两轮实测都**撞满 8 轮上限被硬截断**，说明「轮数」本身不是解法，需要收敛判据。
HISTORY_CLIP = os.getenv("KB_HISTORY_CLIP", "0").strip().lower() not in ("0", "false", "no", "")
# ⚠️ 默认 0（关闭）：2026-10-05 用 25 题评测集实测，「折叠历史」成本 +25%、耗时 +56%、
#    cache 命中率 82%→73% —— 因为改动历史中段会让**前缀缓存从该点起失效**，
#    而命中价约为未命中的 1/10。保留开关只为将来复验（改 1 之前请先跑评测集对比）。
HISTORY_KEEP_TOOL = int(os.getenv("KB_HISTORY_KEEP_TOOL", "6"))      # 最近 N 条工具消息不折叠
HISTORY_CLIP_CHARS = int(os.getenv("KB_HISTORY_CLIP_CHARS", "500"))  # 折叠后保留的字数
# 连续 N 轮「没查到新证据」（工具返回被去重/接近空）就提前收口
STALL_ROUNDS = int(os.getenv("KB_AGENT_STALL_ROUNDS", "2"))

# ⚠️ 2026-10-22 新增：**强制分档**（A/B 实测用）。留空 = 按规则正常分级。
#   动机：实测 50% 的提问被判 deep（deep 中位 67s、fast 中位 10s），
#   但"判 deep 的这些题真的比 fast 答得好吗"从来没有实测过 ——
#   没有这个开关就只能靠改代码再改回来做对比，风险大。
#   取值：""（默认，正常分级）/ "fast" / "deep"。
FORCE_DEPTH = (os.getenv("KB_FORCE_DEPTH", "") or "").strip().lower()
if FORCE_DEPTH not in ("fast", "deep"):
    FORCE_DEPTH = ""
