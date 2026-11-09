"""轻量访问认证：**TOTP 动态令牌** → 签发签名的设备令牌。

演进过程（2026-09-22 同一天改了两轮，把「为什么」记下来，免得以后又绕回去）：
  ① 最初做的是「共享口令」→ 主人反馈「每个用户都会来问我要」，分发成本全压在他身上；
  ② 中途选过「一人一码（固定）」→ 仍要逐个分发，而且**码泄露后在被吊销前一直可用**；
  ③ **最终定：TOTP 动态令牌** —— 码 30 秒即失效、被转发出去也没用；
     每人一个独立密钥 ⇒ **可追溯**（知道是谁在用）、**可单独吊销**（离职/丢手机就停他一个）。

链路（2026-10-09 起有**两条**入口，见本文件「共享长期授权码」那一节）：
    ① TOTP：用户在验证页输 6 位动态码
    ② 共享授权码：用户在验证页输「长期授权码 + 自己的员工编号（只记录不校验）」
      → 后端校验通过（①TOTP 命中 / ②授权码匹配 + 按员工编号 找或建身份）
      → 签发**签名设备令牌**（默认 **30 天**：7 → 15（2026-10-09）→ 30（2026-10-12））
      → 之后由 Cloudflare Worker **离线验签**，全程不再打扰用户

令牌格式：`"<exp>.<fp8>.<uid>.<sig>"`
    uid 让「是谁在用」可追溯：网关验签通过后会把它放进 `X-Auth-User` 转发给后端。

⚠️ TOTP 密钥**必须明文存**（要拿它算码），所以文件权限 600、只放本机 `data/` 目录。
⚠️ `fp8` 只取前 8 位参与签名。**2026-10-09 起它已经不是"算出来的浏览器指纹"**，而是前端
   **随机生成一次、写进 Cookie、之后一直沿用**的设备标识（原因见 worker.js 顶部注释：
   指纹要"算"→ 依赖 crypto.subtle/Canvas → 在老内核/国产浏览器转码模式下抛异常 → 变空串。
   随机 ID 不依赖任何现代 API，**保证非空**）。
   ⚠️ **但仍然只取 8 位、且不能改成全量比对** —— 改长度会让所有已签发的令牌立刻验签失败，
      等于把正在使用的人全部踢下线。
⚠️ 真正的凭证是 **Cookie 里的签名令牌本身**（换浏览器 = 没 Cookie = 必然重验）；
   设备标识只是"同一台设备免重复输码"的便利标签，**不承担安全职责**。
"""
import hashlib
import hmac
import json
import os
import re
import threading
import time
from urllib.parse import quote

from . import config, totp

SIG_LEN = 32          # 签名截断长度（hex 位数）
FP_LEN = 8            # 参与签名的设备标识前缀长度（⚠️ 别改：改长度会让所有已签发令牌失效）

# 名字长度上限（2026-11-02）：只用于**显示与分组**，不参与任何鉴权判定。
NAME_MAX = 24
# 「自动生成的名字」长这样：`<邀请备注>-01`。老邀请流程每家一条，管理员看不出是谁，
# 也没法按人合并 —— 2026-11-02 起改为**用户自己填名字**（见 claim_identity / 邀请页表单）。
_ANON_NAME_RE = re.compile(r"-\d{2}$")


def norm_name(raw: str) -> str:
    """规范化姓名：去不可见字符、内部空白压成一个、截到 NAME_MAX。

    ⚠️ 只做「显示层」规范化，**不当作身份凭证** —— 同名不代表同一个人，
       真正的合并要过 `verify_code_for`（能给出该身份的 6 位动态码才算本人）。
    """
    s = "".join(ch for ch in str(raw or "") if ch.isprintable())
    s = re.sub(r"\s+", " ", s).strip()
    return s[:NAME_MAX]


def looks_anonymous(name: str) -> bool:
    """是不是老流程自动生成的名字（`灰度第一批-01` 这类）。"""
    return bool(_ANON_NAME_RE.search(str(name or "").strip()))


def find_by_name(name: str) -> list[dict]:
    """按规范化名字找人（可能 0~N 个）。返回**停机状态原样**，由调用方决定怎么提示。"""
    n = norm_name(name)
    if not n:
        return []
    return [u for u in load_users() if norm_name(u.get("name")) == n]

# 用户表/邀请表的**写锁**（2026-09-25）：
# 这两个文件是「读-改-写」模式（load_users → 改字段 → save_users），
# 并发下会互相覆盖 —— 例如两个用户同时领取邀请，各自 load 到同一份旧表，
# 后写的那个把先写的顶掉（丢用户）；`new_user` 的 uid 用 len(users)+1 生成，
# 还会让两人拿到同一个 uid。所有**写路径**都必须持这把锁。
# 纯读（verify_code、网关离线验签）不需要加锁：save_users 用原子替换，
# 读者要么看到旧文件、要么看到新文件，不会读到半截。
_LOCK = threading.Lock()


# ---------------- 用户表 ----------------

def load_users() -> list[dict]:
    """读用户表。文件不存在/损坏时返回空表（不要让认证整个炸掉）。"""
    p = config.ACCESS_CODES_FILE
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        users = doc.get("users") if isinstance(doc, dict) else doc
        return users if isinstance(users, list) else []
    except Exception:
        return []


def save_users(users: list[dict]) -> None:
    """写用户表。⚠️ 600 权限：里面是 TOTP 明文密钥，等同密码。

    ⚠️ 2026-09-25 改为**原子写**（临时文件 + `os.replace`）：
       原来直接 `p.write_text(...)` —— 若写到一半被打断（进程被杀、磁盘满），
       会留下**截断的 JSON**，而 `load_users` 遇到损坏会静默返回 `[]`，
       于是**全员无法登录**（本该 fail-closed，实际变成 fail-broken）。
       用 `os.replace` 后，读者要么看到完整旧文件、要么看到完整新文件。
    ⚠️ 调用方若要「读-改-写」，必须持有 `_LOCK`（见 new_user / add_user / touch_user）。
    """
    p = config.ACCESS_CODES_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"users": users}, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except Exception:
        pass
    os.replace(tmp, p)      # 原子替换（同一文件系统内）


def enabled() -> bool:
    """是否启用访问认证。

    ⚠️ 判据是「有没有配签名密钥」，不再看共享口令 —— 用户表为空时也**必须**启用，
       否则删光用户反而把整站敞开，这是最危险的默认值。
    """
    return bool(config.AUTH_SECRET)


def new_user(name: str, note: str = "") -> dict:
    """建一个用户（生成 TOTP 密钥）。**不落盘**。

    ⚠️ 调用方若自行调用本函数再 save_users，**必须持有 `_LOCK`** ——
       否则并发下两人会读到同一份旧表 → 算出同一个 uid → 后写的覆盖先写的。
       多数场景请直接用 `add_user()`（它把"读-生成-写"整个包进锁里）。
    """
    users = load_users()
    uid = "u" + str(len(users) + 1)
    while any(u.get("id") == uid for u in users):        # 删过人之后 id 可能撞，往后顺延
        uid = "u" + str(int(uid[1:]) + 1)
    return {
        "id": uid, "name": (name or "").strip() or uid, "note": (note or "").strip(),
        "secret": totp.new_secret(),
        "created_at": int(time.time()), "last_used_at": None, "use_count": 0, "enabled": True,
    }


def add_user(name: str, note: str = "") -> dict:
    """**原子地**新建用户并落盘（2026-09-25 新增）。

    把「读表 → 生成 uid → 写表」整个包进 `_LOCK`，消除并发撞号与丢写。
    ⚠️ uid 保持 `uN` 形式：它是 `ADMIN_UIDS` 与令牌 payload 里的既有约定，
       改成 uuid 会同时打断权限判断和历史令牌，所以只加锁、不改格式。
    """
    with _LOCK:
        u = new_user(name, note)
        users = load_users()
        users.append(u)
        save_users(users)
        return u


def verify_code(code: str) -> dict | None:
    """遍历用户表校验动态码，返回命中的用户；都不中返回 None。

    为什么遍历而不是让用户先报名字：内部工具就几十个人，遍历是微秒级；
    多一个"先填名字"的步骤，体验差、还容易让人以为自己被停用了。
    ⚠️ 代价：无法区分"本人操作"还是"别人拿了他的手机"—— 这与密码是同一类信任模型，可接受。
    """
    c = (code or "").strip()
    if not c.isdigit() or len(c) != totp.DIGITS:
        return None
    for u in load_users():
        if not u.get("enabled"):
            continue
        # ⚠️⚠️ 2026-10-09 修：`totp.verify` 对**畸形密钥**会抛异常
        #   （实测：`!!!` → "Non-base32 digit found"；`abc` → "Incorrect padding"）。
        #   而这里是**遍历全表**调它 ⇒ **任何一个用户的密钥坏了，所有人的 TOTP 登录全部 500**；
        #   更糟的是异常发生在写审计之前 —— 日志里连一条 login_fail 都没有，线上表现为
        #   「有人反复输码、服务只回 500、查不到任何痕迹」（2026-10-09 在 service.log 里
        #   看到 8 次连续 500 就是这么来的）。
        #   ⇒ 单个用户密钥坏属于**数据问题**，不该让整条登录链路 fail-broken：
        #      跳过这个人、写审计留痕、继续验后面的人（宁可少一个人能登，不能让所有人都登不上）。
        try:
            if totp.verify(str(u.get("secret") or ""), c):
                return u
        except Exception as e:                      # noqa: BLE001
            audit("totp_secret_bad", uid=u.get("id"), name=u.get("name"),
                  error=f"{type(e).__name__}: {e}")
            continue
    return None


# ---------------- 共享长期授权码 + 员工编号（2026-10-09 主人定：放开内测） ----------------
# 背景：原来的入口是 TOTP 动态码 —— 安全，但**分发成本全压在主人身上**：
#   每人得先扫码绑密钥、手机上还得装 Authenticator。内测扩大后这层就不合适了。
# 主人 2026-10-09 定：
#   ① **一条共享的长期授权码**（发到群里，谁拿到谁能进），码不变、不自动过期，
#      要换码/停用就改 `KB_ACCESS_CODE`（改完重启服务即可，无需改代码）；
#   ② 用户进来时**必须填自己的「员工编号」**，我们**只记录、不校验**（不查工号库），
#      拿它当身份标识 —— 于是"匿名共享码"也能知道是**谁**在用、也能**单独停用某个人**；
#   ③ 老的 TOTP 路径**保留**（6 位纯数字仍走 TOTP），已绑密钥的人不受影响。
#
# ⚠️ 安全边界（必须清楚，别自欺）：
#   · 共享码是**长期口令**，泄露 = 谁都能进 → 所以①码要够长（>=16 位随机串，
#     不是 6 位数字，既抗撞也避免和 TOTP 的 6 位数字格式混淆）；②真要收紧就换码。
#   · 员工编号「不校验」⇒ **可以被冒填**。它承担的是「可追溯」，**不是**「防冒充」。
#     要防冒充只能回到 TOTP / 绑定密钥那条路。

STAFF_ID_MAX = 64          # 员工编号 长度上限（只做防脏，不做真伪校验）


def _eq(a: str, b: str) -> bool:
    """常量时间比较。
    ⚠️ 用 bytes 比而不是 str：`hmac.compare_digest` 遇到非 ASCII 的 str 会直接抛
       TypeError —— 授权码里出现一个中文字符就会把登录打成 500。"""
    return hmac.compare_digest(str(a).encode("utf-8"), str(b).encode("utf-8"))


def access_codes() -> list[str]:
    """当前的共享授权码列表（`KB_ACCESS_CODE` 支持逗号分隔多个，便于**平滑换码**：
    先加新码 → 等大家都换过来 → 再去掉旧码，中间不需要所有人同时改。"""
    return [c.strip() for c in (config.ACCESS_CODE or "").split(",") if c.strip()]


def access_code_enabled() -> bool:
    """是否开了「共享授权码」这条入口。留空即关闭（回到只有 TOTP 的状态）。"""
    return bool(access_codes())


def verify_access_code(code: str) -> bool:
    """校验共享授权码。大小写不敏感、忽略首尾空白（发到群里常被复制带上空格）。"""
    c = (code or "").strip()
    if not c:
        return False
    low = c.lower()
    return any(_eq(x.lower(), low) for x in access_codes())


def norm_staff_id(raw: str) -> str:
    """规范化用户填的「员工编号」。

    ⚠️ 主人明确：**只记录、不校验** —— 不查工号库、不验真伪、不做格式强约束。
       这里只做「不让脏东西进 JSON / 审计日志 / 管理页」的最小清洗：
         · 去掉所有空白与控制字符（工号/手机号本身不含空白，去掉比保留安全）；
         · 截断到 STAFF_ID_MAX，避免有人塞一篇文章进来。
    ⚠️ 故意**不**限制成「必须数字/必须 N 位」—— 主人说"只记录不校验"，
       我们就不该自作主张加校验规则（加了就可能把某些真实工号挡在外面）。
    """
    s = "".join(ch for ch in str(raw or "") if ch.isprintable() and not ch.isspace())
    return s.strip()[:STAFF_ID_MAX]


def find_or_create_by_staff(staff_id: str, name: str = "") -> dict:
    """按「员工编号」找一个身份；没有就**当场建一个**。返回 {ok, user?, error?}。

    设计取舍：
      · 为什么"见到新 ID 就建档"而不是"先审批" —— 内测就是要自助。跑完内测
        用 `enabled=False` / `revoked_uids.json` 收拾即可，两个机制都已具备。
      · 身份复用：同一个人换设备/换浏览器，只要还填同一个员工编号，
        就**还是同一条记录**（use_count 累加，不会像匿名邀请那样每人堆一堆垃圾账号）。
      · ⚠️ 整个「查 → 建 → 写」必须包进 `_LOCK`：两人同时首次进入时，
        否则会各自 load 到同一份旧表 → 算出同一个 uid / 后写的顶掉先写的。
      · 2026-11-02：`name` 是用户在登录页**自己填的名字**（可空）。
        只在「新建」时使用；已有记录**不改名**（避免有人用共享码把别人的名字改掉）。
    """
    hid = norm_staff_id(staff_id)
    if not hid:
        return {"ok": False, "error": "请填写你的 LDAP 账号"}
    with _LOCK:
        users = load_users()
        for u in users:
            if norm_staff_id(u.get("staff_id") or "") == hid:
                if not u.get("enabled"):
                    return {"ok": False, "error": "这个身份已被管理员停用，请联系管理员"}
                return {"ok": True, "user": u, "created": False}
        u = new_user(norm_name(name) or hid, note="共享授权码自助加入")
        u["staff_id"] = hid
        u["auth_via"] = "access_code"
        u["name_source"] = "user" if norm_name(name) else "staff"
        users.append(u)
        save_users(users)
        return {"ok": True, "user": u, "created": True}



def set_staff_id(uid: str, staff_id: str) -> None:
    """给一条已有身份补记员工编号（TOTP 老用户顺手填了的话）。

    ⚠️ 只补**空**的：不覆盖别人已经填过的值 —— 否则拿到共享码的人只要在 TOTP 路径
       乱填一个员工编号，就能把某个已有身份的归属改掉。
    """
    hid = norm_staff_id(staff_id)
    if not hid:
        return
    with _LOCK:
        users = load_users()
        for u in users:
            if u.get("id") == uid and not norm_staff_id(u.get("staff_id") or ""):
                u["staff_id"] = hid
                save_users(users)
                return


def set_staff_id_self(uid: str, staff_id: str) -> bool:
    """本人自助补/改自己的 LDAP 账号（2026-11-04）。

    与 `set_staff_id()` 的区别：**允许覆盖自己已有的值**（填错了要能改）。
    ⚠️ 仍然只改「自己这一条」—— uid 由设备令牌解出，不接受外部传入，
       否则又是一条越权改别人归属的路。
    """
    hid = norm_staff_id(staff_id)
    if not hid:
        return False
    with _LOCK:
        users = load_users()
        for u in users:
            if u.get("id") == uid:
                u["staff_id"] = hid
                save_users(users)
                return True
    return False


def verify_code_for(uid: str, code: str) -> bool:
    """校验「这个 6 位码是不是**某个指定身份**的」。合并同名身份前用它证明是本尊。

    ⚠️ 不能用 `verify_code()`（它是遍历全表）—— 那会把「碰巧撞上别人的码」
       也算通过，等于谁都能认领任意同名身份。
    ⚠️ 密钥损坏时返回 False（与 verify_code 的口径一致：数据问题不该 500）。
    """
    c = (code or "").strip()
    if not c.isdigit() or len(c) != totp.DIGITS:
        return False
    u = next((x for x in load_users() if x.get("id") == uid and x.get("enabled")), None)
    if not u:
        return False
    try:
        return totp.verify(str(u.get("secret") or ""), c)
    except Exception as e:                          # noqa: BLE001
        audit("totp_secret_bad", uid=uid, name=u.get("name"), error=f"{type(e).__name__}: {e}")
        return False


def rename_user(uid: str, name: str) -> dict | None:
    """管理员给身份改名（用于把老邀请的匿名名改成人名，或纠正写错的名字）。

    改名**不合并身份** —— 只是把显示名改掉；多条记录改成同一个名字后，
    管理面板会把它们归到同一组并按设备数统计（见 /admin/status 的返回）。
    """
    n = norm_name(name)
    if not n:
        return None
    with _LOCK:
        users = load_users()
        for u in users:
            if u.get("id") == uid:
                u["name"] = n
                u["name_source"] = "admin"
                save_users(users)
                return u
    return None





def touch_user(uid: str, fp: str = "") -> None:
    """记录一次成功使用（谁在用、多久没用过、**在几台设备上用过**）。

    ⚠️ 2026-09-25：读-改-写整体加锁 —— 原来多个用户同时登录时，
       各自的「load → 改自己那行 → save」会互相覆盖，后写的顶掉先写的，
       表现为 use_count 莫名偏低、last_used_at 丢更新。

    ⚠️ 2026-11-02 新增 `devices`：按 `fp8`（令牌里那个设备标识前缀）去重记录。
       这样同一个身份在手机 + 电脑上各登一次 = 2 台设备，管理员面板可以直接显示。
       · 只存 `fp8` 与时间，**不存 IP / UA**（够用即可，少收集一点）；
       · 最多保留 `_MAX_DEVICES` 台，按最后使用时间淘汰最旧的 —— 防止有人反复清
         Cookie 刷出无限条目，把 users 文件撑大。
    """
    now = int(time.time())
    fp8 = (fp or "").strip()[:FP_LEN]
    with _LOCK:
        users = load_users()
        for u in users:
            if u.get("id") != uid:
                continue
            u["last_used_at"] = now
            u["use_count"] = int(u.get("use_count") or 0) + 1
            if fp8:
                devs = u.get("devices")
                if not isinstance(devs, dict):
                    devs = {}
                d = devs.get(fp8)
                if not isinstance(d, dict):
                    d = {"first_at": now}
                d["last_at"] = now
                d["n"] = int(d.get("n") or 0) + 1
                devs[fp8] = d
                if len(devs) > _MAX_DEVICES:
                    # 淘汰最久没用的（保留最近 _MAX_DEVICES 台）
                    keep = sorted(devs.items(),
                                  key=lambda kv: int((kv[1] or {}).get("last_at") or 0),
                                  reverse=True)[:_MAX_DEVICES]
                    devs = dict(keep)
                u["devices"] = devs
            save_users(users)
            return


_MAX_DEVICES = 20     # 单个身份最多记多少台设备（超出淘汰最久未用的）


def device_summary(u: dict) -> dict:
    """给管理面板用的设备统计：台数 + 最近一次。**不返回 fp 本身**（没必要外泄）。"""
    devs = u.get("devices")
    if not isinstance(devs, dict):
        devs = {}
    last = max([int((v or {}).get("last_at") or 0) for v in devs.values()] or [0])
    return {"count": len(devs), "last_at": last,
            "ids": sorted(devs.keys())}      # ids 仅用于跨身份去重（同组求和会重复计数）


def audit(event: str, **kw) -> None:
    """审计流水（JSONL，一行一条，可直接 tail 看）。"""
    try:
        p = config.AUTH_AUDIT_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": int(time.time()), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass          # 审计失败绝不能影响登录本身


# ---------------- 设备令牌 ----------------

def _sign(exp: int, fp8: str, uid: str) -> str:
    msg = f"{exp}.{fp8}.{uid}".encode("utf-8")
    return hmac.new(config.AUTH_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()[:SIG_LEN]


# ---------------- 邀请链接（2026-09-22 主人场景）----------------
# 场景：一条链接发到灰度群里 → 群成员各自点开、自助领取 → 24 小时后失效 → 之后再生成一条发给下一批。
# 与「专属绑定链接」的区别：**邀请链接不含 uid**，谁点谁领、彼此独立（各生成一份 TOTP 密钥）。
#
# ⚠️ 为什么必须"每人一份密钥"、绝不能全群共用一个：TOTP 是时间同步的，
#    同一个密钥在同一时刻算出**同一个码** —— 等于回到"共享口令"，还多一层假安全感。
# ⚠️ 匿名是主人明确选的（不想让大家填名字），但那样"出了事不知道找谁"。
#    折中：领取时自动起一个**可区分但匿名**的名字（如「灰度第一批-03」）——
#    你仍不知道是谁，但能看清"有哪几个身份在用"，也能单独停用某一个。

def _invite_sig(exp: int, iid: str = "") -> str:
    """邀请令牌签名。

    ⚠️ 2026-10-23 加 `iid`（**一次性邀请**配套改动）：
      · 不带 iid = **旧格式**，签名内容 `join.{exp}` —— 已在外面发出去的链接必须continue 验得过；
      · 带 iid  = **新格式**，签名内容 `join.{iid}.{exp}` —— 令牌里带邀请 id。

    为什么要带 id：旧格式的令牌里**只有到期时间**，`verify_invite_token` 只能按 `exp`
    去表里找记录 —— 同一秒创建、TTL 又相同的两条邀请会**撞在同一时刻**，
    于是"找到的是哪一条"取决于列表顺序（既有隐患）。
    一次性邀请必须精确对应到**唯一一条**记录，否则「已被领取」可能判到别人头上。
    """
    msg = f"join.{iid}.{exp}" if iid else f"join.{exp}"
    return hmac.new(config.AUTH_SECRET.encode("utf-8"),
                    msg.encode("utf-8"), hashlib.sha256).hexdigest()[:SIG_LEN]


def invite_kind(it: dict) -> str:
    """这条邀请是一次性还是多人共用。

    ⚠️ 缺省 `multi` 是**故意的**：改动前创建的邀请记录里没有 `kind` 字段，
    它们全都是"多人共用、到期失效"的老行为 —— 默认值必须让老记录行为不变。
    """
    k = str((it or {}).get("kind") or "multi").strip().lower()
    return "single" if k == "single" else "multi"


def claim_sig(uid: str) -> str:
    """给「已领取」标记签个名，放进 Cookie。

    ⚠️ 为什么需要它：邀请页刷新时不能又建一个新身份（否则库里堆一串垃圾用户）。
       但 `/auth/join/<token>` 是**免鉴权**的，这个 Cookie 就成了唯一依据 ——
       所以必须**签名**，否则伪造一个 uid 就能认领别人的身份。
    """
    return hmac.new(config.AUTH_SECRET.encode("utf-8"),
                    f"claim.{uid}".encode("utf-8"), hashlib.sha256).hexdigest()[:SIG_LEN]


def read_claim_cookie(value: str) -> str:
    """从 Cookie 里解出已领取的 uid；签名不对返回空串。"""
    parts = (value or "").split(".")
    if len(parts) != 2:
        return ""
    uid, sig = parts
    if not config.AUTH_SECRET or not hmac.compare_digest(sig, claim_sig(uid)):
        return ""
    return uid


# ---------------- 令牌吊销名单（2026-09-25） ----------------
# 动机：设备令牌是**离线验签**的（网关只验签名、不回源查库），所以"用户表里删掉这个人"
#      并不能让他的令牌失效 —— 签名照样验得过，7 天内照用。要真正落实「停用即失效、
#      离职即停」，必须有一份**显式名单**，由后端在每个带身份的请求上比对。

def load_revoked() -> list[dict]:
    p = config.REVOKED_FILE
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        arr = doc.get("revoked") if isinstance(doc, dict) else doc
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def save_revoked(items: list[dict]) -> None:
    """写吊销名单（原子替换，理由同 save_users）。"""
    p = config.REVOKED_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"revoked": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except Exception:
        pass
    os.replace(tmp, p)


def revoke(uid: str, reason: str = "") -> None:
    """把 uid 加入吊销名单（停用 / 删除用户时调用）。重复调用只更新原因与时间。"""
    uid = (uid or "").strip()
    if not uid:
        return
    with _LOCK:
        items = load_revoked()
        for it in items:
            if it.get("uid") == uid:
                it["reason"] = reason or it.get("reason") or ""
                it["at"] = int(time.time())
                break
        else:
            items.append({"uid": uid, "reason": reason, "at": int(time.time())})
        save_revoked(items)


def unrevoke(uid: str) -> None:
    """把 uid 移出吊销名单（恢复某个账号时用）。"""
    uid = (uid or "").strip()
    if not uid:
        return
    with _LOCK:
        save_revoked([it for it in load_revoked() if it.get("uid") != uid])


def is_revoked(uid: str) -> bool:
    """该 uid 现在是否应当被拒绝。判据（任一命中即拒）：
      ① 显式吊销名单命中；
      ② 用户表里有这个人但 `enabled=False`（停用即失效）；
      ③ 用户表正常、但已查无此人（已被删除）。

    ⚠️ 边界：用户表文件缺失时 `load_users()` 返回 `[]` —— 此时**不做第 ③ 条判断**，
       否则一次文件异常会把所有人挡在门外。那种属于配置问题，由启动日志与 /healthz 暴露，
       不该在认证路径上放大影响面。
    """
    uid = (uid or "").strip()
    if not uid:
        return False                       # 无身份头 = 本机直连/免身份路径，交原有校验处理
    for it in load_revoked():
        if it.get("uid") == uid:
            return True
    if not config.ACCESS_CODES_FILE.is_file():
        return False                       # 用户表缺失，不据此判吊销（见上）
    for u in load_users():
        if u.get("id") == uid:
            return not u.get("enabled")
    return True                            # 用户表正常但查无此人 → 已删除 → 拒


def load_invites() -> list[dict]:
    p = config.INVITES_FILE
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        arr = doc.get("invites") if isinstance(doc, dict) else doc
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def save_invites(invites: list[dict]) -> None:
    """写邀请表（**原子替换**，理由同 save_users）。
    ⚠️ 调用方若「读-改-写」（如 claim_identity 记录已领名单），必须持有 `_LOCK`。"""
    p = config.INVITES_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"invites": invites}, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except Exception:
        pass
    os.replace(tmp, p)


def new_invite(note: str = "", ttl_hours: int | None = None, kind: str = "multi") -> dict:
    """新建一条邀请（写进邀请表并返回）。

    `kind`：
      · `"multi"`（默认）—— **多人共用**：发到群里，谁点谁领，到期失效；
      · `"single"`         —— **一次性**：**一个人领取后立即失效**，别人再点会被拒。

    ⚠️ 一次性邀请**仍然带到期时间**（不是"永不过期，直到被领"）：
       一条发出去没人用的邀请，如果只等"被领"才失效，就会**永久留在表里可被使用** ——
       等于埋了一个没人看着的入口。时间上限 + 一次使用，两个约束都要有。
    ⚠️ 令牌格式：新格式是 `{id}.{exp}.{sig}`（**三段**，与本文件 `issue_bind_token`
       的 `{uid}.{exp}.{sig}` 同风格）；旧格式 `{exp}.{sig}`（两段）继续支持，见 `verify_invite_token`。
    """
    if not config.AUTH_SECRET:
        return {"ok": False, "error": "未配置 KB_AUTH_SECRET"}
    kind = str(kind or "multi").strip().lower()
    if kind not in ("multi", "single"):
        kind = "multi"
    hours = ttl_hours if ttl_hours else config.AUTH_BIND_TTL_HOURS
    exp = int(time.time()) + max(1, hours) * 3600
    invites = load_invites()
    iid = "inv" + str(len(invites) + 1)
    while any(x.get("id") == iid for x in invites):
        iid = "inv" + str(int(iid[3:]) + 1)
    rec = {"id": iid, "note": (note or "").strip() or iid, "exp": exp,
           "created_at": int(time.time()), "enabled": True, "claimed": [], "kind": kind}
    invites.append(rec)
    save_invites(invites)
    return {"ok": True, "id": iid, "token": f"{iid}.{exp}.{_invite_sig(exp, iid)}",
            "expires_at": exp, "ttl_hours": hours, "note": rec["note"], "kind": kind}


def verify_invite_token(token: str, known_uid: str = "") -> tuple[bool, str, dict | None]:
    """校验邀请令牌 → (是否有效, 原因, 邀请记录)。

    `known_uid`：请求方已能确认的身份（**必须**是 `kb_claim` 签名解出来的），
    ⚠️ 2026-10-28 新增 —— 一次性邀请要放行「领取者本人重进」：否则他刷新一次、
       或换标签页再看一眼自己的二维码，就会撞上自己造成的"已被领取"。
    """
    if not config.AUTH_SECRET:
        return False, "未配置 AUTH_SECRET", None
    parts = (token or "").strip().split(".")
    iid = ""
    if len(parts) == 3:                       # 新格式 {id}.{exp}.{sig}
        iid, exp_s, sig = parts
    elif len(parts) == 2:                     # 旧格式 {exp}.{sig}
        exp_s, sig = parts
    else:
        return False, "邀请链接格式不对", None
    try:
        exp = int(exp_s)
    except ValueError:
        return False, "邀请链接格式不对", None
    if not hmac.compare_digest(sig, _invite_sig(exp, iid)):
        return False, "邀请链接无效", None
    if exp < int(time.time()):
        return False, f"邀请链接已过期（有效期 {config.AUTH_BIND_TTL_HOURS} 小时），请找管理员要新的", None

    invites = load_invites()
    if iid:
        it = next((x for x in invites if x.get("id") == iid), None)
        if it is None:
            return False, "邀请链接已失效（记录已不存在）", None
        if int(it.get("exp") or 0) != exp:
            # 令牌里的到期时间与记录不符 ⇒ 要么记录被改过，要么令牌是拼的
            return False, "邀请链接已失效（到期时间已变更）", None
    else:
        it = next((x for x in invites if x.get("exp") == exp), None)
        if it is None:
            return False, "邀请链接已失效（可能已被作废）", None
    if not it.get("enabled"):
        return False, "这条邀请已被管理员作废", None
    # ⭐ 一次性邀请：**已被领取即失效**（这是与多人共用的唯一区别）
    # ⚠️ 2026-10-28：但**领取者本人重进必须放行**。
    #   真实事故：管理员生成一条一次性邀请 → 自己点开看了一眼 → 名额被吃掉 →
    #   发给对方时只剩"已被领取"。本人判定只能靠签名 Cookie（known_uid），伪造不了。
    if invite_kind(it) == "single" and it.get("claimed"):
        if known_uid and known_uid in (it.get("claimed") or []):
            return True, "ok", it
        return False, ("这条邀请是<b>一次性</b>的，已经被领取过了 —— "
                       "请找管理员要一条新的（每条只能给一个人）"), None
    return True, "ok", it


def revoke_invite(invite_id: str) -> bool:
    """作废一条邀请（管理员用）。

    做法：把 `enabled` 置 False —— `verify_invite_token` 会据此返回"已被管理员作废"，
    **已经领过的人不受影响**（他们的身份与令牌早已独立生成），只影响还没领的链接。
    ⚠️ 整段加锁：与 `claim_identity` 一样是「读-改-写」，两人同时操作会互相覆盖。
    """
    invite_id = (invite_id or "").strip()
    if not invite_id:
        return False
    with _LOCK:
        invites = load_invites()
        hit = False
        for it in invites:
            if it.get("id") == invite_id:
                it["enabled"] = False
                it["revoked_at"] = int(time.time())
                hit = True
        if hit:
            save_invites(invites)
        return hit


def reset_invite_claims(invite_id: str) -> dict:
    """把一条邀请的**已领名单清空**，让它能再服务一个人（「重新启用」）。

    场景（2026-10-23 主人报的真实问题）：一次性邀请被领走后链接就 410 了，
    而领取的幂等**只靠浏览器 Cookie**（`kb_claim`）——
    换个浏览器 / 无痕 / 从微信里打开，就会被当成新的人、把邀请消耗掉。
    实测「陈总专用」连建两条、两条都被同一个人领走，管理员再点就是 410。
    ⇒ 给管理员一个**一键救回**的动作，比"作废了重新生成"顺手得多。

    ⚠️ **已经领过的人不受影响** —— 他们的身份与 TOTP 密钥在领取时就独立生成了
       （与 `revoke_invite` 同一个道理）。清空名单只是"允许再来一个人"。
    ⚠️ 整段加锁：与 `claim_identity` 一样是「读-改-写」。
    """
    invite_id = (invite_id or "").strip()
    if not invite_id:
        return {"ok": False, "error": "缺少 invite_id"}
    with _LOCK:
        invites = load_invites()
        for it in invites:
            if it.get("id") == invite_id:
                cleared = len(it.get("claimed") or [])
                it["claimed"] = []
                it["reset_at"] = int(time.time())
                it["reset_count"] = int(it.get("reset_count") or 0) + 1
                save_invites(invites)
                return {"ok": True, "invite_id": invite_id, "cleared": cleared,
                        "reset_count": it["reset_count"]}
        return {"ok": False, "error": "没找到这条邀请"}


def claim_identity(invite: dict, name: str = "") -> dict | None:
    """为一个领取者创建**独立身份**（名字优先用**用户自己填的**）。

    ⚠️ 2026-11-02：原来名字是 `<邀请备注>-01`（自动序号）—— 管理员在列表里根本看不出
       这是谁，也没法按人合并。现在由邀请页表单收集姓名，落库即 `name=用户填写的名字`。
       `name` 为空时**退回老行为**（`<备注>-NN`），保证老链接/老调用方不炸。
    ⚠️ 这里**只负责建身份**，不负责合并 —— 同名合并必须先在邀请页验过 6 位动态码
       （见 `verify_code_for`），否则任何人填「张三」就能拿到张三的密钥 = 身份冒用。

    ⚠️ 2026-09-25：整个「读用户表 → 生成身份 → 写回 → 记邀请已领名单」包进 `_LOCK`。
       原来无锁，两人同时点同一条邀请链接时会：① seq 算出同一个序号（名字重号）；
       ② 后写的覆盖先写的（**丢掉一个人**）；③ 邀请的 claimed 名单也丢一条。

    ⚠️ 2026-10-23（一次性邀请）：**返回 `None` 表示"抢不到"**（已被领/已作废/记录没了）。
       `verify_invite_token` 与这里**必须各判一次**：
         · verify 那次是给用户看原因（页面层）；
         · 这一次是**在锁内**做的，用来挡"两个人同时点同一条一次性链接"的竞态 ——
           两个请求可能都在 verify 时看到 claimed 为空，只有一个能领到。
    """
    with _LOCK:
        # ⭐ 锁内拿最新状态再判一次（不能用调用方传进来的 invite 快照）
        cur = next((x for x in load_invites() if x.get("id") == invite.get("id")), None)
        if cur is None:
            return None
        if not cur.get("enabled"):
            return None
        if invite_kind(cur) == "single" and cur.get("claimed"):
            return None
        users = load_users()
        seq = len([u for u in users if u.get("from_invite") == invite.get("id")]) + 1
        uname = norm_name(name)
        auto_named = not uname
        u = new_user(uname or f"{invite.get('note') or '受邀'}-{seq:02d}",
                     note=f"来自邀请 {invite.get('id')}")
        u["from_invite"] = invite.get("id")
        # 记录名字来源：老数据是 auto，管理员面板据此提示「这个名字要补」
        u["name_source"] = "auto" if auto_named else "user"
        users.append(u)
        save_users(users)
        # 记进邀请记录的已领名单（用于"这条邀请被领了多少份"）
        invites = load_invites()
        for it in invites:
            if it.get("id") == invite.get("id"):
                it.setdefault("claimed", []).append(u["id"])
        save_invites(invites)
        return u


def mark_claim(invite_id: str, uid: str) -> bool:
    """把 uid 记进邀请的已领名单（**合并路径**用：没建新身份，但这条邀请确实被用了）。

    与 `claim_identity` 内部那段逻辑等价，单独抽出来是因为合并发生在 `_LOCK` 之外 ——
    ⚠️ 不能把 `mark_claim` 再套进 `claim_identity` 的锁里：`_LOCK` 是不可重入的
       `threading.Lock`，嵌套会**自锁死**（2026-10-29 在 cases.delete_case 上踩过同一个坑）。
    """
    invite_id = (invite_id or "").strip()
    if not invite_id:
        return False
    with _LOCK:
        invites = load_invites()
        for it in invites:
            if it.get("id") != invite_id:
                continue
            if not it.get("enabled"):
                return False
            if invite_kind(it) == "single" and it.get("claimed"):
                return False          # 一次性名额已被别人抢走
            it.setdefault("claimed", []).append(uid)
            save_invites(invites)
            return True
    return False


def issue_bind_token(uid: str, ttl_hours: int | None = None) -> dict:
    """生成**限时绑定令牌**（默认 24 小时）—— 给本人扫码绑定的那条链接用。

    ⚠️ 它和设备令牌是**两码事**，别混：
      · 绑定令牌：只用于「把密钥安全递给本人」这一次动作，24 小时就废，**换不到任何数据**；
      · 设备令牌：验证通过后进站用的，7 天有效。
    ⚠️ 为什么只让**链接**过期、不让密钥过期：密钥一换，**已经绑定好的人也会一起失效**，
       得重新绑一遍 —— 那才是真的折腾人。限住分发窗口就够了。
    """
    if not config.AUTH_SECRET:
        return {"ok": False, "error": "未配置 KB_AUTH_SECRET"}
    hours = ttl_hours if ttl_hours else config.AUTH_BIND_TTL_HOURS
    exp = int(time.time()) + max(1, hours) * 3600
    sig = hmac.new(config.AUTH_SECRET.encode("utf-8"),
                   f"bind.{uid}.{exp}".encode("utf-8"), hashlib.sha256).hexdigest()[:SIG_LEN]
    return {"ok": True, "token": f"{uid}.{exp}.{sig}", "expires_at": exp, "ttl_hours": hours}


def verify_bind_token(token: str) -> tuple[str, str]:
    """校验绑定令牌 → (uid, 原因)。uid 为空串表示无效。"""
    if not config.AUTH_SECRET:
        return "", "未配置 AUTH_SECRET"
    parts = (token or "").strip().split(".")
    if len(parts) != 3:
        return "", "绑定链接格式不对"
    uid, exp_s, sig = parts
    try:
        exp = int(exp_s)
    except ValueError:
        return "", "绑定链接格式不对"
    if exp < int(time.time()):
        return "", f"绑定链接已过期（有效期 {config.AUTH_BIND_TTL_HOURS} 小时），请找管理员重新生成"
    expect = hmac.new(config.AUTH_SECRET.encode("utf-8"),
                      f"bind.{uid}.{exp}".encode("utf-8"), hashlib.sha256).hexdigest()[:SIG_LEN]
    if not hmac.compare_digest(sig, expect):
        return "", "绑定链接无效"
    if not any(u.get("id") == uid and u.get("enabled") for u in load_users()):
        return "", "该用户不存在或已停用"
    return uid, "ok"


def issue(fp: str, uid: str, ttl_days: int | None = None) -> dict:
    """签发设备令牌。返回 {ok, token, max_age, fp8, uid, expires_at, ttl_days}。"""
    if not config.AUTH_SECRET:
        return {"ok": False, "error": "未配置 KB_AUTH_SECRET，无法签发令牌"}
    fp8 = (fp or "").strip()[:FP_LEN]
    if not fp8:
        # 正常流程到不了这里：前端 worker.js 的 getDeviceId() 有三级降级
        # （randomUUID → getRandomValues → Math.random），**保证一定带得上设备标识**。
        # 能走到这里说明请求不是从正常页面发出的（页面脚本被拦/被改写，或有人直接调接口）。
        # ⚠️ 文案必须给**可操作动作**：旧文案「缺少浏览器指纹」用户完全看不懂，
        #    只会反复重输动态码（2026-10-09 就是这么踩的，见 worker.js 顶部注释）。
        return {"ok": False,
                "error": "未能读取本机设备标识：请用最新版 Chrome / Edge / Safari 打开本站"
                         "（避免用国产浏览器的“极速/兼容”模式），或清除本站 Cookie 后重试"}
    days = ttl_days if ttl_days else config.AUTH_TTL_DAYS
    exp = int(time.time()) + max(1, days) * 86400
    token = f"{exp}.{fp8}.{uid}.{_sign(exp, fp8, uid)}"
    return {"ok": True, "token": token, "max_age": max(1, days) * 86400, "fp8": fp8,
            "uid": uid, "expires_at": exp, "ttl_days": days}


def verify(token: str, fp: str = "") -> tuple[bool, str, str]:
    """校验令牌 → (是否有效, 原因, uid)。后端自用；Worker 有等价实现（Web Crypto）。"""
    if not enabled():
        return True, "认证未启用", ""
    parts = (token or "").strip().split(".")
    if len(parts) != 4:
        return False, "令牌格式不对", ""
    exp_s, fp8, uid, sig = parts
    try:
        exp = int(exp_s)
    except ValueError:
        return False, "令牌格式不对", ""
    if exp < int(time.time()):
        return False, "验证已过期，请重新验证", ""
    if not hmac.compare_digest(sig, _sign(exp, fp8, uid)):
        return False, "令牌无效（签名不匹配）", ""
    if fp and fp[:FP_LEN] != fp8:
        return False, "换了设备或浏览器，请重新验证", ""
    return True, "ok", uid


def set_cookie_headers(token: str, fp: str, max_age: int) -> list[str]:
    """两个 Set-Cookie：令牌（HttpOnly）+ 指纹（JS 可读写）。

    ⚠️ 令牌必须 HttpOnly：前端 JS 读不到，XSS 也偷不走。
    ⚠️ 指纹单独放一个**非** HttpOnly 的 Cookie：Worker 验签时要拿它跟令牌里的 fp8 比对。
    """
    ttl = f"; Max-Age={max_age}"
    common = f"; Path=/; Secure; SameSite=Lax{ttl}"
    return [
        f"{config.AUTH_COOKIE}={token}{common}; HttpOnly",
        f"{config.AUTH_FP_COOKIE}={quote(fp or '')}{common}",
    ]


def clear_cookie_headers() -> list[str]:
    """清掉两个 Cookie（「退出验证」用）。"""
    dead = "; Path=/; Secure; SameSite=Lax; Max-Age=0"
    return [f"{config.AUTH_COOKIE}=; HttpOnly{dead}", f"{config.AUTH_FP_COOKIE}=;{dead}"]
