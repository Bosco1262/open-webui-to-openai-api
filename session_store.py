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
from typing import Any, Dict, Optional, Tuple

import httpx

import lang
from atomic_json import atomic_write_json
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


# --------------------------------------------------------------------------- #
# Upstream response shape
# 上游响应形态
# --------------------------------------------------------------------------- #
def looks_like_model_list(response: httpx.Response) -> bool:
    """
    Whether a 2xx /models answer really is the model list.

    Open WebUI's SPA answers unknown paths with HTTP 200 and an HTML page, so a 2xx on
    its own proves neither that the route exists nor that the credentials are valid.
    Both callers -- the prefix probe (upstream.probe_prefix) and the browser-login
    validation (see _credentials_are_valid) -- must therefore look at the body, not
    just the status code.


    2xx 的 /models 回答是否真的是模型列表。

    Open WebUI 的 SPA 会用 HTTP 200 + 一页 HTML 回答未知路径，因此单凭 2xx 既不能证明
    路由存在，也不能证明凭证有效。两个调用方——前缀探测（upstream.probe_prefix）与
    浏览器登录校验（见 _credentials_are_valid）——都必须看响应体，而不是只看状态码。
    """
    if "html" in response.headers.get("content-type", "").lower():
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, (dict, list))


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
        """
        Build the header set forwarded upstream: the JSON Accept/Content-Type pair, the
        user agent, and the captured Authorization / Cookie when present.

        构造转发给上游的请求头：JSON 的 Accept/Content-Type 组合、User-Agent，以及
        抓到的 Authorization / Cookie（仅在有值时携带）。
        """
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

        The token prefix is shortened to 8 characters: for a JWT that is already almost
        no information, and for a hand-made token it limits how much of a secret ends up
        in a log line (R12).


        返回脱敏后的凭证摘要，可安全写进日志。

        Token 前缀缩短到 8 个字符：对 JWT 而言本就几乎没有信息量，对手工签发的
        token 则限制了泄露进日志的密钥长度（R12）。
        """
        parts = []
        if self.authorization:
            parts.append(f"token={self.authorization[:8]}…(len={len(self.authorization)})")
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
        # Coerce every header field to str: a hand-edited file may carry numbers or
        # other non-strings, and is_usable()/to_headers() would otherwise raise
        # AttributeError/TypeError and turn a bad file into persistent HTTP 500s.
        #
        # 所有头字段强制转 str：手工编辑的文件可能带数字等非字符串值，
        # 否则 is_usable()/to_headers() 会抛 AttributeError/TypeError，
        # 把一个坏文件变成持续的 HTTP 500。
        return cls(
            authorization=str(get("authorization") or ""),
            cookie=str(get("cookie") or ""),
            user_agent=str(get("user_agent") or get("user-agent") or ""),
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


# Parsed session.json, keyed by path -> (mtime_ns, size, Session). load_session() runs on
# every request (via app._session_or_error), so re-reading and re-parsing the file each
# time is pure waste; the stat check keeps an externally replaced file -- a re-login from
# another process, a hand-edited session.json -- visible. Failures are never cached.
#
# 已解析的 session.json，键为路径 -> (mtime_ns, size, Session)。load_session() 每个请求都会
# 被调用（经 app._session_or_error），每次重新读盘解析纯属浪费；stat 检查保证被外部替换的
# 文件（另一进程重新登录、手工编辑 session.json）依然可见。失败结果一律不缓存。
_session_cache: Dict[str, Tuple[int, int, Session]] = {}


def _invalidate_session_cache(settings: Settings) -> None:
    """Drop the cached parse of the credential file. / 丢弃凭证文件的解析缓存。"""
    _session_cache.pop(str(settings.session_file), None)


def _normalize_base_url(value: str) -> str:
    """
    Compare upstream addresses the way they are written in practice: a trailing slash
    or a different case of the scheme/host must not count as a different site.

    按实际书写习惯比较上游地址：结尾多余的斜杠、scheme/host 的大小写差异都不算换了站点。
    """
    return value.strip().rstrip("/").lower()


def _assert_base_url_matches(settings: Settings, session: Session) -> None:
    """
    Refuse credentials captured for a different upstream (D7).

    `base_url` was recorded from the very beginning but never read, so pointing
    OPEN_WEBUI_BASE_URL at another instance silently reused credentials that cannot
    work there -- the failure then surfaced much later, as an upstream 401 attributed
    to an expired token. Browser credentials are bound to the site that issued them,
    so a mismatch is a configuration error with exactly one fix.

    An empty `base_url` (files written by older versions, or a hand-edited file) is
    allowed: there is nothing to compare against.


    拒绝为另一个上游抓取的凭证（D7）。

    `base_url` 从一开始就记录在文件里，却从未被读取，于是把 OPEN_WEBUI_BASE_URL 指向
    另一个实例时会静默复用在那里根本不可能生效的凭证——问题随后才以"上游 401、疑似
    token 过期"的形式暴露出来。浏览器凭证与签发它的站点绑定，因此不一致属于配置错误，
    只有一种修法。

    空的 `base_url`（旧版本写出的文件、或手工编辑过的文件）放行：没有可比对的对象。
    """
    stored = _normalize_base_url(session.base_url)
    if not stored:
        return
    configured = _normalize_base_url(settings.open_webui_base_url)
    if stored != configured:
        raise SessionInvalid(
            lang.t(
                "session_base_url_mismatch",
                path=settings.session_file,
                stored=session.base_url.strip(),
                configured=settings.open_webui_base_url,
            )
        )


def load_session(settings: Settings) -> Session:
    """
    Load the credential file, validating it on the way in: a missing file raises
    SessionMissing; unreadable, unparseable, unusable or wrong-upstream files raise
    SessionInvalid. The parsed Session is cached by (mtime, size), so repeat calls
    cost one stat.

    读取凭证文件并在读取途中完成校验：文件缺失抛 SessionMissing；不可读、不可解析、
    不可用或属于另一个上游的文件抛 SessionInvalid。解析结果按 (mtime, size) 缓存，
    重复调用只多付一次 stat。
    """
    path: Path = settings.session_file
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        _invalidate_session_cache(settings)
        raise SessionMissing(
            lang.t("session_missing_file", path=path)
        ) from exc
    except OSError as exc:
        # Permission denied, path is a directory, ...: reporting "not found" here
        # would send the operator down a re-login path that can never fix it.
        #
        # 权限拒绝、路径是目录等：报成"文件不存在"会把排障引向
        # 永远解决不了问题的重新登录路径。
        _invalidate_session_cache(settings)
        raise SessionInvalid(
            lang.t("session_unreadable", path=path, exc=exc)
        ) from exc

    key = str(path)
    cached = _session_cache.get(key)
    if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _invalidate_session_cache(settings)
        raise SessionInvalid(lang.t("session_unparseable", path=path, exc=exc)) from exc
    if not isinstance(raw, dict):
        _invalidate_session_cache(settings)
        raise SessionInvalid(lang.t("session_wrong_shape", path=path))

    try:
        session = Session.from_dict(raw)
    except Exception as exc:
        # from_dict is defensive, but a pathological file (e.g. a JSON list with a
        # dict-shaped tail) must still surface as SessionInvalid, never as 500.
        #
        # from_dict 已设防，但病态文件仍须归为 SessionInvalid，而不是 500。
        _invalidate_session_cache(settings)
        raise SessionInvalid(
            lang.t("session_wrong_shape", path=path)
        ) from exc
    if not session.is_usable():
        _invalidate_session_cache(settings)
        raise SessionInvalid(
            lang.t("session_no_creds", path=path)
        )
    _assert_base_url_matches(settings, session)
    _session_cache[key] = (stat.st_mtime_ns, stat.st_size, session)
    return session


def save_session(settings: Settings, session: Session) -> None:
    path: Path = settings.session_file
    # Atomic replace with flush + fsync before the rename (R6/D10, shared with the probe
    # cache): an interrupted write must never leave a half-written credential file
    # behind, and the rename must not become durable before the bytes it points at.
    #
    # 原子替换，且在改名之前 flush + fsync（R6/D10，与探测缓存共用）：写盘中断绝不能
    # 留下半截凭证文件，改名也不得早于它指向的字节而先持久化。
    atomic_write_json(path, session.to_dict())
    # Do not rely on the stat check alone: the caller may immediately read the file back
    # within the filesystem's mtime granularity.
    #
    # 不只依赖 stat 检查：调用方可能在文件系统的 mtime 粒度内立刻回读该文件。
    _invalidate_session_cache(settings)
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
    # Exact prefix, or prefix + "/": a bare startswith would also match sibling
    # namespaces such as {base}/api-evil or {base}/apiary.
    #
    # 精确等于前缀，或前缀 + "/"：裸的 startswith 会把 {base}/api-evil、
    # {base}/apiary 这类同级命名空间也一并命中。
    if not (url == api_prefix or url.startswith(api_prefix + "/")):
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
    network errors are all judged invalid; 404 means trying the next candidate prefix;
    and a 2xx whose body is not the model list (the SPA's 200 + HTML page for an
    unknown path) proves nothing either, so that also moves on to the next candidate.


    对上游做一次真实请求，校验抓到的凭证当前是否有效。

    抓取逻辑只能看到"请求带了凭证"，但带的不一定是有效凭证：
    - 页面加载早期，前端会用 localStorage 里的旧 Token 发探测请求（Token 可能已过期）；
    - 校园网 / 公司网等强制门户（captive portal）未完成认证时，任何请求都会被
      网关重定向到认证页——此时抓到的只是"重定向前"的旧请求头。

    因此凭证必须通过上游一次真实鉴权才算有效：401/403（无效/过期）、
    3xx（被门户重定向）、网络错误一律判无效；404 换下一个候选前缀再试；
    响应体不是模型列表的 2xx（未知路径被 SPA 用 200 + HTML 回答）同样证明不了什么，
    也换下一个候选。
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
            if not (200 <= resp.status_code < 300):
                return False
            if not looks_like_model_list(resp):
                continue
            return True
    return False


async def _launch_browser(playwright: Any, *, headless: bool) -> Any:
    """
    Launch Chromium, turning every failure into a SessionError that carries the fix.

    A browser that was never installed (`playwright install chromium` not run) or a
    server without a display used to escape as a raw traceback: the caller only handles
    SessionError, so the operator got a page of Playwright stack instead of the two
    lines that actually solve it (D9).


    启动 Chromium，并把所有失败翻译成附带解决办法的 SessionError。

    浏览器未安装（没执行过 `playwright install chromium`）或在无显示器的服务器上运行时，
    异常原先会以裸 traceback 逃出：调用方只处理 SessionError，于是运维看到的是一页
    Playwright 堆栈，而不是真正能解决问题的两行提示（D9）。
    """
    try:
        return await playwright.chromium.launch(headless=headless)
    except Exception as exc:  # noqa: BLE001 - every launch failure needs the same hints
        raise SessionError(lang.t("browser_launch_failed", exc=exc)) from exc


async def _open_login_page(browser: Any, settings: Settings) -> Tuple[Any, Any]:
    """
    Create the browser context and the page the login is watched in (D9).

    创建登录观察所用的浏览器上下文与页面（D9）。
    """
    try:
        context = await browser.new_context(
            ignore_https_errors=not settings.upstream_verify_ssl
        )
        return await context.new_page(), context
    except Exception as exc:  # noqa: BLE001 - same treatment as a launch failure
        raise SessionError(lang.t("browser_context_failed", exc=exc)) from exc


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
        """
        Playwright request hook: capture the credential headers of requests that carry
        real identity information (is_login_signal filters out the anonymous first-load
        cookies). A failure here is logged and never breaks the login flow.

        Playwright 请求钩子：抓取携带真实身份信息的请求头（首屏匿名 Cookie 会被
        is_login_signal 挡掉）。此处异常只记日志，绝不打断登录流程。
        """
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
        browser = await _launch_browser(playwright, headless=headless)
        try:
            page, context = await _open_login_page(browser, settings)
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
                # The browser is closed by the finally below, which also covers a
                # failure while opening the context/page.
                #
                # 浏览器由下方 finally 统一关闭，它同时覆盖"打开上下文/页面时失败"的情况。
                raise SessionError(lang.t("login_timeout", timeout=timeout))
            except SessionError:
                raise
            except Exception as exc:
                # e.g. the user closed the browser directly: continue as long as credentials were captured
                # 用户直接关掉浏览器等情况：只要抓到了凭证就继续
                if not captured.is_usable():
                    raise SessionError(lang.t("login_interrupted", exc=exc)) from exc
                logger.warning(lang.t("login_browser_exit", exc=exc))
        finally:
            try:
                if not browser.is_closed():
                    await browser.close()
            except Exception:  # pragma: no cover - closing is best effort
                pass

    if not captured.is_usable():
        raise SessionError(lang.t("login_no_creds"))

    save_session(settings, captured)
    logger.info(lang.t("creds_saved", path=settings.session_file, desc=captured.describe()))
    # U-0: this file is not a sample -- it is a working upstream session, with the same
    # power as the operator's browser login. Say so at the one moment it is written.
    #
    # U-0：这个文件不是示例，而是一个能直接用的上游会话，权限等同于运维的浏览器登录态。
    # 在写出它的那一刻就把这一点讲清楚。
    logger.warning(lang.t("creds_live_warning", path=settings.session_file))
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
