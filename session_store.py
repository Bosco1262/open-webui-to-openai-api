"""
Credential management: load/save session.json, plus optional browser-based login capture.

session.json stores exactly the request headers needed to access Open WebUI after a
browser login; this project never parses or uploads these credentials in any way.

凭证管理：加载/保存 session.json，以及可选的浏览器登录抓取。

session.json 里存的就是浏览器登录后访问 Open WebUI 所需的请求头，
本项目不会以任何方式解析或上传这些凭证。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

import lang
from config import DEFAULT_USER_AGENT, Settings

# Playwright is an optional dependency
# Playwright 是可选依赖
try:
    from playwright.async_api import async_playwright
except ImportError:
    # pragma: no cover - depends on the runtime environment
    # 取决于运行环境
    async_playwright = None  # type: ignore[assignment]

logger = logging.getLogger("webui-proxy.session")

# Upstream endpoints that the frontend only calls after a successful login,
# used to judge that "the user is really logged in"
#
# 只有登录成功后前端才会去调用的上游接口，用于判定"确实登录了"
AUTHED_PATH_HINTS = (
    "/api/models",
    "/api/chat/completions",
    "/api/chats",
    "/api/v1/",
    "/api/users",
    "/api/folders",
    "/api/knowledge",
)


class SessionError(RuntimeError):
    """
    Base class for credential-related errors.

    凭证相关错误的基类。
    """


class SessionMissing(SessionError):
    """
    session.json does not exist.

    session.json 不存在。
    """


class SessionInvalid(SessionError):
    """
    session.json exists but its content is unusable.

    session.json 存在但内容不可用。
    """


@dataclass
class Session:
    authorization: str = ""
    cookie: str = ""
    user_agent: str = ""
    captured_at: float = 0.0
    base_url: str = ""

    def is_usable(self) -> bool:
        return bool(self.authorization.strip() or self.cookie.strip())

    def to_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent or DEFAULT_USER_AGENT,
        }
        if self.authorization:
            headers["Authorization"] = self.authorization
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    def to_dict(self) -> Dict[str, Any]:
        """
        Write snake_case keys only.

        Early single-file versions wrote Authorization / Cookie / User-Agent; we no
        longer write both key styles (from_dict accepts both, so readers need not care).

        只写 snake_case 一种键名。

        早期单文件版本写的是 Authorization / Cookie / User-Agent，这里不再重复
        写两份（from_dict 对两种键名都兼容，读取侧无需关心）。
        """
        return asdict(self)

    def age_days(self) -> Optional[float]:
        if not self.captured_at:
            return None
        return (time.time() - self.captured_at) / 86400.0

    def describe(self) -> str:
        """
        Return a redacted credential summary, safe to write into logs.

        返回脱敏后的凭证摘要，可安全写进日志。
        """
        parts = []
        if self.authorization:
            parts.append(f"token={self.authorization[:16]}…(len={len(self.authorization)})")
        if self.cookie:
            parts.append(f"cookie(len={len(self.cookie)})")
        age = self.age_days()
        if age is not None:
            parts.append(f"age={age:.1f}d")
        return ", ".join(parts) or lang.t("describe_empty")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Session":
        get = _ci_getter(raw)
        captured_at = raw.get("captured_at") or raw.get("capturedAt") or 0.0
        try:
            captured_at = float(captured_at)
        except (TypeError, ValueError):
            captured_at = 0.0
        return cls(
            authorization=get("authorization") or "",
            cookie=get("cookie") or "",
            user_agent=get("user_agent") or get("user-agent") or "",
            captured_at=captured_at,
            base_url=str(raw.get("base_url") or ""),
        )


def _ci_getter(raw: Dict[str, Any]):
    """
    Build a case-insensitive getter.

    构造一个大小写不敏感的取值函数。
    """
    lowered = {str(key).lower(): value for key, value in raw.items()}

    def get(key: str) -> Any:
        return lowered.get(key.lower())

    return get


# --------------------------------------------------------------------------- #
# Read / write
# 读写
# --------------------------------------------------------------------------- #
def session_exists(settings: Settings) -> bool:
    return settings.session_file.exists()


def load_session(settings: Settings) -> Session:
    path: Path = settings.session_file
    if not path.exists():
        raise SessionMissing(
            lang.t("session_missing_file", path=path)
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionInvalid(lang.t("session_unparseable", path=path, exc=exc)) from exc
    if not isinstance(raw, dict):
        raise SessionInvalid(lang.t("session_wrong_shape", path=path))

    session = Session.from_dict(raw)
    if not session.is_usable():
        raise SessionInvalid(
            lang.t("session_no_creds", path=path)
        )
    return session


def save_session(settings: Settings, session: Session) -> None:
    path: Path = settings.session_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    # Tighten permissions on POSIX so other users on the same machine cannot read the credentials
    # POSIX 下收敛权限，避免凭证被同机其他用户读取
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            # pragma: no cover - permission-system differences
            # 权限系统差异
            pass


# --------------------------------------------------------------------------- #
# Browser login capture
# 浏览器登录抓取
# --------------------------------------------------------------------------- #
def _normalize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {str(name).lower(): str(value) for name, value in (headers or {}).items()}


def _build_cookie_header(cookies: Any) -> str:
    """
    Render a Playwright cookie jar (or a ready-made header string) as a Cookie header.

    把 Playwright 的 Cookie Jar（或现成的头字符串）渲染成 Cookie 请求头。
    """
    if isinstance(cookies, str):
        return cookies
    pairs = []
    for cookie in cookies or []:
        name = cookie.get("name")
        value = cookie.get("value")
        if name is not None and value is not None:
            pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def is_login_signal(url: str, headers: Dict[str, str], api_prefix: str) -> bool:
    """
    Judge whether a request indicates that "the user has logged in".

    Kept as a pure function so it can be tested without Playwright.

    - Strong signal: a request to the upstream /api/ carrying a non-empty Bearer token;
    - Weak signal: cookie only -- anonymous visits carry cookies too (theme, CSRF, etc.),
      so additionally require the request to hit an endpoint that the frontend only
      calls when logged in.

    判断一个请求是否说明"用户已经登录了"。

    拆成纯函数是为了可以脱离 Playwright 单独测试。

    - 强信号：发往上游 /api/ 的请求带了非空 Bearer Token；
    - 弱信号：只有 Cookie —— 匿名访问同样会带 Cookie（主题、CSRF 等），
      因此额外要求命中的是必须登录后前端才会调用的接口。
    """
    if not url.startswith(api_prefix):
        return False

    lowered = {
        str(name).lower(): str(value) for name, value in (headers or {}).items()
    }
    authorization = lowered.get("authorization", "").strip()
    cookie = lowered.get("cookie", "").strip()
    if not authorization and not cookie:
        return False

    if authorization.lower().startswith("bearer ") and len(authorization) > len("bearer "):
        return True
    return bool(cookie) and any(hint in url for hint in AUTHED_PATH_HINTS)


async def _credentials_are_valid(settings: Settings, session: Session) -> bool:
    """
    Make one real request to the upstream to check whether the captured
    credentials are currently valid.

    The capture logic can only see "the request carried credentials", but what it
    carries is not necessarily valid:
    - Early in page load, the frontend sends probe requests with an old token from
      localStorage (which may have expired);
    - When a captive portal (campus/corporate network) has not completed
      authentication, every request is redirected by the gateway to the auth page --
      what gets captured is just the "pre-redirect" old request headers.

    Therefore credentials only count as valid if they pass one real upstream
    authentication: 401/403 (invalid/expired), 3xx (redirected by a portal), and
    network errors are all judged invalid; 404 means trying the next candidate prefix.

    对上游做一次真实请求，校验抓到的凭证当前是否有效。

    抓取逻辑只能看到"请求带了凭证"，但带的不一定是有效凭证：
    - 页面加载早期，前端会用 localStorage 里的旧 Token 发探测请求（Token 可能已过期）；
    - 校园网 / 公司网等强制门户（captive portal）未完成认证时，任何请求都会被
      网关重定向到认证页——此时抓到的只是"重定向前"的旧请求头。

    因此凭证必须通过上游一次真实鉴权才算有效：401/403（无效/过期）、
    3xx（被门户重定向）、网络错误一律判无效；404 则换下一个候选前缀再试。
    """
    if not session.is_usable():
        return False
    headers = session.to_headers()
    async with httpx.AsyncClient(
        verify=settings.upstream_verify_ssl,
        trust_env=settings.upstream_trust_env,
        timeout=15.0,
        # A portal redirect yields 3xx, which is exactly what we judge invalid
        # 被门户重定向时拿到 3xx，正好判无效
        follow_redirects=False,
    ) as client:
        for prefix in settings.prefix_candidates():
            url = f"{settings.open_webui_base_url}{prefix}/models"
            try:
                resp = await client.get(url, headers=headers)
            except httpx.RequestError:
                return False
            if resp.status_code == 404:
                continue
            return 200 <= resp.status_code < 300
    return False


async def perform_browser_login(
    settings: Settings,
    *,
    headless: Optional[bool] = None,
    timeout: Optional[int] = None,
) -> Session:
    """
    Open a browser for the user to log in manually, and capture the post-login
    request headers.

    The standard for "login succeeded" is far stricter than "a cookie exists":
    it must be a request to the upstream /api/ carrying a Bearer token or session
    cookie, AND those credentials must pass one real upstream authentication
    (see _credentials_are_valid). The anonymous cookies produced by the first page
    load, and the expired old token probe requests sent early in page load, are
    never saved as a valid login state.

    打开浏览器让用户手动登录，抓取登录后的请求头。

    判定"登录成功"的标准比"存在 Cookie"严格得多：
    必须是发往上游 /api/ 的请求，且携带 Bearer Token 或会话 Cookie，
    并且这些凭证能通过上游一次真实鉴权（见 _credentials_are_valid）。
    首页首次加载产生的匿名 Cookie、以及页面早期发出的过期旧 Token
    探测请求，都不会被当成有效登录态保存。
    """
    if async_playwright is None:
        raise SessionError(lang.t("playwright_missing"))

    headless = settings.login_headless if headless is None else headless
    timeout = settings.login_timeout if timeout is None else timeout
    api_prefix = f"{settings.open_webui_base_url}/api"

    print("=" * 62, flush=True)
    print(lang.t("login_banner_open", url=settings.open_webui_base_url), flush=True)
    print(lang.t("login_banner_auto"), flush=True)
    print(lang.t("login_banner_portal"), flush=True)
    print(lang.t("login_banner_timeout", timeout=timeout), flush=True)
    print("=" * 62, flush=True)

    captured = Session(base_url=settings.open_webui_base_url)
    event = asyncio.Event()

    async def on_request(request) -> None:
        try:
            url = str(request.url)
            headers = _normalize_headers(await request.all_headers())
            if not is_login_signal(url, headers, api_prefix):
                return

            authorization = headers.get("authorization", "").strip()
            cookie = headers.get("cookie", "").strip()
            if authorization:
                captured.authorization = authorization
            if cookie:
                captured.cookie = cookie
            captured.user_agent = headers.get("user-agent", "") or DEFAULT_USER_AGENT
            captured.captured_at = time.time()
            event.set()
        except Exception as exc:
            # pragma: no cover - page event callbacks must not break the flow
            # 页面事件回调不应打断整体流程
            logger.debug(lang.t("capture_error_debug", exc=exc))

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context(ignore_https_errors=not settings.upstream_verify_ssl)
        page = await context.new_page()
        page.on("request", on_request)
        try:
            await page.goto(settings.open_webui_base_url, wait_until="domcontentloaded")
        except Exception as exc:
            logger.warning(lang.t("goto_failed", exc=exc))

        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                await asyncio.wait_for(event.wait(), timeout=remaining)
                event.clear()
                # Quiet observation period: login is considered finished only if no
                # newer credentials arrive during this window
                #
                # 静默观察期：期间没有更新的凭证才认为登录流程结束
                try:
                    await asyncio.wait_for(event.wait(), timeout=settings.login_quiet_period)
                except asyncio.TimeoutError:
                    pass
                # Crucial step: captured credentials must pass a real upstream
                # authentication to count as a successful login. Old-token probe
                # requests sent early in page load, and old request headers captured
                # while a campus portal is unauthenticated, are both stopped here and
                # we keep waiting for the user to complete the real login.
                #
                # 关键一步：抓到的凭证必须能通过上游真实鉴权才算登录成功。
                # 页面早期的旧 Token 探测请求、校园网等门户未认证时的旧请求头
                # 都会被这里拦下，继续等待用户完成真正的登录。
                if await _credentials_are_valid(settings, captured):
                    break
                logger.info(lang.t("validate_failed"))
            await _enrich_from_browser(page, context, settings, captured)
        except asyncio.TimeoutError:
            await browser.close()
            raise SessionError(lang.t("login_timeout", timeout=timeout))
        except SessionError:
            await browser.close()
            raise
        except Exception as exc:
            # e.g. the user closed the browser directly: continue as long as credentials were captured
            # 用户直接关掉浏览器等情况：只要抓到了凭证就继续
            if not captured.is_usable():
                await browser.close()
                raise SessionError(lang.t("login_interrupted", exc=exc)) from exc
            logger.warning(lang.t("login_browser_exit", exc=exc))
        finally:
            try:
                if not browser.is_closed():
                    await browser.close()
            except Exception:  # pragma: no cover
                pass

    if not captured.is_usable():
        raise SessionError(lang.t("login_no_creds"))

    save_session(settings, captured)
    logger.info(lang.t("creds_saved", path=settings.session_file, desc=captured.describe()))
    return captured


async def _enrich_from_browser(page, context, settings: Settings, session: Session) -> None:
    """
    Additionally capture the token in localStorage and the full cookie jar.

    The cookie in request headers may be incomplete; Open WebUI also stores its JWT
    in the `token` key of localStorage, so reading it directly yields the most
    complete identity information.

    补充抓取 localStorage 里的 token 与完整 Cookie Jar。

    请求头里的 Cookie 可能不完整；Open WebUI 也把 JWT 存在 localStorage 的
    `token` 键里，直接读取能得到最完整的身份信息。
    """
    try:
        token = await page.evaluate("() => { try { return localStorage.getItem('token') || ''; } catch (e) { return ''; } }")
    except Exception as exc:
        logger.debug(lang.t("localStorage_failed", exc=exc))
        token = ""
    if token and token != "undefined":
        session.authorization = f"Bearer {token}"

    try:
        # Only take cookies of the Open WebUI domain: the browser may also carry
        # cookies of other sites such as a campus portal, and mixing them into the
        # upstream request headers does no good at all.
        #
        # 只取 Open WebUI 域的 Cookie：浏览器里可能还带着校园网门户等其他
        # 站点的 Cookie，混进上游请求头没有任何好处。
        cookies = await context.cookies(settings.open_webui_base_url)
        cookie_header = _build_cookie_header(cookies)
        if cookie_header:
            session.cookie = cookie_header
    except Exception as exc:
        logger.debug(lang.t("cookies_failed", exc=exc))

    if not session.user_agent:
        try:
            session.user_agent = await page.evaluate("() => navigator.userAgent") or DEFAULT_USER_AGENT
        except Exception:
            session.user_agent = DEFAULT_USER_AGENT
    session.captured_at = time.time()
