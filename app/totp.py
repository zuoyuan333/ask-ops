"""TOTP 动态令牌（RFC 6238）—— 只用标准库实现，不引第三方依赖。

为什么自己写：hmac / hashlib / base64 / struct 四件套就够，不必为此装包。
少一个依赖 = 少一个供应链风险面，也让"本机部署"更简单。

⚠️ 参数**按主流 Authenticator 的默认值**：SHA-1 / 30 秒 / 6 位数字。
   换算法或步长会让市面上大部分 App 扫不出来（Google Authenticator、微软、1Password 都按这套默认走），
   **别自作主张改**。

⚠️ 与「固定码」相比，TOTP 的价值是**不怕泄露**：码 30 秒就失效，被转发出去也没用。
   代价是用户要装一个 Authenticator App。
"""
import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote

STEP = 30          # 步长（秒）
DIGITS = 6         # 位数


def new_secret(nbytes: int = 20) -> str:
    """生成新密钥（Base32、无填充）。20 字节 = 160 bit，与主流 App 的默认长度一致。"""
    return base64.b32encode(secrets.token_bytes(nbytes)).decode().rstrip("=")


def _decode(secret_b32: str) -> bytes:
    s = (secret_b32 or "").strip().upper().replace(" ", "").replace("-", "")
    s += "=" * ((8 - len(s) % 8) % 8)          # Base32 要补齐到 8 的倍数
    return base64.b32decode(s, casefold=True)


def code_at(secret_b32: str, ts: float | None = None) -> str:
    """算出某个时刻的 6 位码（主要给测试与排障用）。"""
    counter = int((ts if ts is not None else time.time()) // STEP)
    digest = hmac.new(_decode(secret_b32), struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F                                   # 动态截断（RFC 6238 规定）
    val = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** DIGITS)
    return str(val).zfill(DIGITS)


def verify(secret_b32: str, code: str, window: int = 1) -> bool:
    """校验动态码。

    ⚠️ 默认**前后各容 1 个周期（±30 秒）**：手机时钟漂移非常常见，
       容差给 0 会让"明明码是对的"天天失败，用户会骂到你这儿来。
    ⚠️ 比较用 compare_digest（常数时间），避免时序侧信道。
    """
    c = (code or "").strip().replace(" ", "").replace("-", "")
    if not c.isdigit() or len(c) != DIGITS:
        return False
    now = time.time()
    for w in range(-window, window + 1):
        if hmac.compare_digest(code_at(secret_b32, now + w * STEP), c):
            return True
    return False


def otpauth_url(secret_b32: str, name: str, issuer: str = "POS 报障助手") -> str:
    """给 App 扫的标准 URI。前端把它渲染成二维码，用户扫一下即完成绑定。"""
    label = quote(f"{issuer}:{name}")
    return (f"otpauth://totp/{label}"
            f"?secret={secret_b32}&issuer={quote(issuer)}"
            f"&algorithm=SHA1&digits={DIGITS}&period={STEP}")
