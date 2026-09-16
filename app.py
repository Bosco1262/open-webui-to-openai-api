"""
open-webui-to-openai-api -- Reverse-proxy an Open WebUI instance into an OpenAI-compatible API.

Entry points:
    python app.py              # Start the service (logs in via browser first if needed)
    python app.py --login      # Force re-login and refresh credentials
    python app.py --check      # Only verify whether the current credentials still work

open-webui-to-openai-api -- 把 Open WebUI 反代为 OpenAI 兼容接口。

入口：
    python app.py              # 启动服务（必要时先做一次浏览器登录）
    python app.py --login      # 强制重新登录并刷新凭证
    python app.py --check      # 只校验当前凭证是否可用
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Any, Dict, List, Optional, Tuple

import lang
import httpx
import uvicorn
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

try:
    from config import Settings, settings
except RuntimeError as exc:
    # Invalid configuration (most commonly a missing http:// in OPEN_WEBUI_BASE_URL) is
    # raised as RuntimeError at import time. Intercept it here and turn it into a
    # one-line friendly message before exiting, instead of dumping a full traceback.
    # ImportErrors for missing dependencies are NOT caught here and propagate as usual.
    #
    # 配置非法（最常见：OPEN_WEBUI_BASE_URL 漏了 http://）会在 import 期抛
    # RuntimeError。在这里拦下来翻译成一行友好提示直接退出，而不是让用户
    # 面对一整页 traceback。缺依赖之类的 ImportError 不在此列，照常抛出。
    print(lang.t("startup_failed", exc=exc), file=sys.stderr)
    sys.exit(2)
from session_store import (
    SessionError,
    SessionInvalid,
    SessionMissing,
    load_session,
    perform_browser_login,
    session_exists,
)
from upstream import (
    AUTH_FAILURE_CODES,
    UpstreamClient,
    UpstreamRequestInvalid,
    UpstreamUnavailable,
)
from model_probe import looks_like_effort_error  # noqa: F401 - used by the chat route
# Request correlation (U-7): one id per request, carried by every log line it produces
# and echoed back in the X-Request-ID response header.
#
# 请求关联（U-7）：每个请求一个 id，随它产生的每一行日志携带，并在 X-Request-ID
# 响应头里回显。
from request_context import (
    RequestIdFormatter,
    bind_request_id,
    new_request_id,
    request_id_suffix,
)
# Model-list normalization (pure logic, no HTTP/cache/config) lives in models.py (R5);
# the names are re-exported so routes and tests keep addressing them through app.
#
# 模型列表规范化（纯逻辑，不碰 HTTP/缓存/配置）在 models.py（R5）；这里重新导出这些
# 名字，使路由与测试继续通过 app 模块访问它们。
from models import (  # noqa: F401 - re-exports for routes and tests
    _model_fingerprint,
    _model_summaries,
    _parse_timestamp,
    _QUANT_PATTERN,
    _raw_model_capabilities,
    _shared_default_capabilities,
    normalize_model,
)
# Runtime singletons (upstream client, probe cache) and the whole probe orchestration
# live in probe_runner; the names are re-exported here so routes and tests can keep
# addressing them through the app module.
#
# 运行时单例（上游客户端、探测缓存）与整套探测编排都在 probe_runner；这里
# 重新导出这些名字，使路由与测试继续通过 app 模块访问它们。
import probe_runner  # noqa: E402 - main() mirrors rebuilt singletons into it
from probe_runner import (  # noqa: F401 - re-exports for routes and tests
    VALIDATION_FAILURE_CODES,
    _background_tasks,
    _ensure_instance_meta,
    _fetch_raw_models,
    _get_raw_models_cached,
    _healing,
    _http_error,
    _instance_meta,
    _probe_model,
    _PROBE_ANNOUNCE_TIMEOUT,
    _refresh_model_probe,
    _refresh_state,
    _RefreshState,
    _shutdown_background_tasks,
    _spawn_model_probe_refresh,
    _startup_check,
    _trigger_probe_heal,
    HttpError,
    extract_model_list,
    model_probe,
    probe_cache_status,
    probe_health,
    upstream,
)

logger = logging.getLogger("webui-proxy")

VERSION = "1.1.0"


def configure_logging(current: Settings) -> None:
    level_name = current.log_level.upper()
    # uvicorn understands TRACE, but stdlib logging has no such level; a bare getattr
    # would silently fall back to INFO (fewer logs than the user expects), so map it
    # explicitly to the closest level, DEBUG.
    #
    # uvicorn 认得 TRACE，但 stdlib logging 没有这个级别，直接 getattr 会静默
    # 退回 INFO（日志比用户预期的少），这里显式映射到最接近的 DEBUG。
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.DEBUG if level_name == "TRACE" else logging.INFO
    # The handler carries the per-request correlation id (U-7). RequestIdFormatter
    # supplies "-" when a record was emitted outside a request (startup, CLI), so the
    # format needs no per-call-site cooperation.
    #
    # 处理器携带逐请求的关联 id（U-7）。RequestIdFormatter 在请求之外（启动、CLI）
    # 产生的记录上填空 "-"，因此日志格式不需要任何调用点配合。
    handler = logging.StreamHandler()
    handler.setFormatter(
        RequestIdFormatter("%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s")
    )
    # force=True: this runs at import time and again from the lifespan, and duplicate
    # handlers would double every log line.
    #
    # force=True：本函数在 import 期与 lifespan 中各跑一次，重复的处理器会让每行日志翻倍。
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _enforce_listen_safety(current: Settings) -> None:
    """
    D13: an unauthenticated proxy listening on a non-loopback interface is an
    open door to the upstream session. Refuse to start unless the operator
    explicitly acknowledges with ALLOW_INSECURE=true.
    """
    if current.authentication_enabled() or current.allow_insecure:
        return
    host = (current.proxy_host or "").strip().lower()
    loopback = host in ("127.0.0.1", "localhost", "::1", "[::1]", "")
    if not loopback:
        logger.error(lang.t("insecure_listen_refused", host=current.proxy_host))
        raise SystemExit(2)


def _log_exposure_summary(current: Settings) -> None:
    """
    Report the settings that decide how far the deployment is exposed: the passthrough
    allowlist, how many proxy keys exist (U-1), and whether the upstream link is
    cleartext (U-8). All of them are decisions the operator should see confirmed in the
    startup log, not discover from behavior.

    报告决定暴露面的几项设置：透传白名单、Key 数量（U-1），以及上游链路是否明文（U-8）。
    它们都是运维应当在启动日志里看到确认、而不是从行为反推的决定。
    """
    if current.upstream_is_plain_http_nonloopback():
        # U-8: a notice, not a refusal -- "http://192.168.x.x:3000" is one of this
        # project's normal deployments. Loopback stays silent.
        #
        # U-8：提示而非拒绝——"http://192.168.x.x:3000" 是本项目的常规部署形态之一。
        # 回环地址不提示。
        logger.warning(
            lang.t("upstream_plain_http_warning", url=current.open_webui_base_url)
        )
    if current.passthrough_allow_all:
        logger.warning(lang.t("passthrough_unrestricted"))
    elif current.passthrough_allow:
        logger.info(
            lang.t("passthrough_allowlist", paths=", ".join(current.passthrough_allow))
        )
    else:
        logger.info(lang.t("passthrough_disabled"))
    if current.proxy_api_keys:
        logger.info(
            lang.t(
                "proxy_keys_summary",
                count=len(current.proxy_api_keys),
                names=", ".join(sorted(current.proxy_api_keys)),
            )
        )


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    configure_logging(settings)
    _enforce_listen_safety(settings)
    logger.info("=" * 60)
    logger.info(lang.t("banner_start", version=VERSION))
    logger.info(lang.t("banner_upstream", url=settings.open_webui_base_url))
    if settings.proxy_host in ("0.0.0.0", "::"):
        # 0.0.0.0 only means "listen on all interfaces"; it is NOT a valid API address to
        # put into a client -- Chromium/Electron refuses to connect to 0.0.0.0 and fails
        # directly with net::ERR_ADDRESS_INVALID.
        #
        # 0.0.0.0 只是"监听所有网卡"，不是能填进客户端的 API 地址——
        # Chromium/Electron 连 0.0.0.0 会直接报 net::ERR_ADDRESS_INVALID。
        logger.info(lang.t("banner_listen_all", port=settings.proxy_port))
        logger.info(lang.t("banner_local", port=settings.proxy_port))
        logger.info(lang.t("banner_lan", port=settings.proxy_port))
    else:
        logger.info(lang.t("banner_host", host=settings.proxy_host, port=settings.proxy_port))
    logger.info(lang.t("banner_session", path=settings.session_file))
    logger.info(lang.t("banner_style", style=settings.upstream_api_style))
    logger.info("=" * 60)
    _log_exposure_summary(settings)
    await _startup_check()
    # Background per-model probe: covers the "first login / empty cache" case as well
    # as plain startups; never blocks the service from serving.
    #
    # 后台逐模型探测：既覆盖"首次登录 / 缓存为空"，也覆盖普通启动；
    # 绝不阻塞服务对外提供服务。
    _spawn_model_probe_refresh()
    try:
        yield
    finally:
        await _shutdown_background_tasks()
        await upstream.aclose()


app = FastAPI(
    title="Open WebUI to OpenAI API",
    version=VERSION,
    description=lang.t("app_description"),
    lifespan=lifespan,
)

# --------------------------------------------------------------------------- #
# CORS (optional, off by default)
# CORS（可选，默认关闭）
# --------------------------------------------------------------------------- #
if settings.cors_origins:
    # Only enabled when PROXY_CORS_ORIGINS is explicitly configured; off by default to
    # avoid widening the exposure surface. Browser preflight (OPTIONS) is answered by
    # the middleware before routing, so the /v1/{path} route not listing OPTIONS does
    # no harm. allow_credentials stays False: this proxy authenticates via the
    # Authorization header and does not rely on cookies, and False is what permits "*".
    #
    # 只有显式配置 PROXY_CORS_ORIGINS 才启用；默认不开，避免扩大暴露面。
    # 浏览器预检（OPTIONS）由中间件在路由之前直接应答，因此 /v1/{path}
    # 路由没有列 OPTIONS 方法也不影响。allow_credentials 保持 False：
    # 本代理走 Authorization 头鉴权，不依赖 Cookie，False 才允许通配 "*"。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )


# --------------------------------------------------------------------------- #
# Request context: correlation id + response hardening (U-7, U-9)
# 请求上下文：关联 id + 响应加固（U-7、U-9）
# --------------------------------------------------------------------------- #
# Static hardening applied to every response. None of these need per-route decisions:
# the proxy never serves HTML, and everything it does serve (the upstream address, the
# probed prefix, the model list) is per-deployment information that must not sit in a
# shared or browser cache.
#
# 对所有响应统一施加的静态加固。它们都不需要逐路由判断：本代理不吐 HTML，而它吐出的
# 一切（上游地址、探测到的前缀、模型列表）都是部署级信息，不应留在共享缓存或浏览器缓存里。
_SECURITY_RESPONSE_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"cache-control", b"no-store, private"),
)


def _request_header(scope: Any, name: bytes) -> str:
    """
    Read one request header out of an ASGI scope (header names are case-insensitive
    and conventionally lowercase, but are not required to be).

    从 ASGI scope 中读取一个请求头（头名不区分大小写，习惯上小写，但并不强制）。
    """
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1", errors="replace")
    return ""


class RequestContextMiddleware:
    """
    Attach a correlation id to every request and harden every response (U-7, U-9).

    Written as a pure ASGI middleware rather than a BaseHTTPMiddleware one on purpose:
    this service streams SSE, and the BaseHTTPMiddleware wrapper would put an extra
    buffering channel in front of every streamed response.

    The id is the client's own X-Request-ID when it supplied a usable one (so a client
    can match its retry against this service's log), a fresh UUID otherwise. It is
    bound for the request's context -- every log line the request produces carries it --
    and echoed back in the response header, so a client reporting "the upstream errored"
    can be answered by grepping one id.

    为每个请求附上关联 id，并加固每个响应（U-7、U-9）。

    刻意写成纯 ASGI 中间件而不是 BaseHTTPMiddleware：本服务要流式输出 SSE，而
    BaseHTTPMiddleware 包装层会给每个流式响应前面多插一条缓冲通道。

    id 在客户端提供了可用值时用它的 X-Request-ID（便于客户端把自己的重试与
    本服务日志对上），否则生成新的 UUID。它绑定在该请求的上下文里——该请求产生的
    每一行日志都携带它——并在响应头里回显，因此"上游报错了"的反馈只需要 grep 一个 id。
    """

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = new_request_id(_request_header(scope, b"x-request-id"))
        # No reset on the way out: each request is handled in its own task, so the value
        # cannot leak into another request's context.
        #
        # 退出时不做重置：每个请求在自己的任务里处理，该值不会泄漏到别的请求上下文。
        bind_request_id(request_id)

        async def send_with_context(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                _harden_response_headers(
                    message.setdefault("headers", []), request_id
                )
            await send(message)

        await self.app(scope, receive, send_with_context)


def _harden_response_headers(headers: List[Any], request_id: str) -> None:
    """
    Append the correlation id and the static hardening headers, without overwriting a
    header the route set itself (the chat SSE route sets its own Cache-Control).

    追加关联 id 与静态加固头，不覆盖路由自己设置的同名头（对话 SSE 路由自带
    Cache-Control）。
    """
    present = {key.lower() for key, _ in headers}
    if b"x-request-id" not in present:
        headers.append((b"x-request-id", request_id.encode("latin-1")))
    for name, value in _SECURITY_RESPONSE_HEADERS:
        if name not in present:
            headers.append((name, value))


app.add_middleware(RequestContextMiddleware)


# --------------------------------------------------------------------------- #
# Authentication
# 鉴权
# --------------------------------------------------------------------------- #
def _presented_proxy_key(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[len("Bearer ") :].strip()
    return request.headers.get("X-API-Key", "").strip()


def match_proxy_key(presented: str) -> Optional[str]:
    """
    Which configured proxy key the request presented, by name; None when it matches
    none (U-10).

    Every configured key is compared, without an early exit: returning on the first
    match would make the response time depend on the position of the matching key in
    the list, which is information a brute-forcer can use. The comparison itself is
    constant time, as before. Must encode to bytes -- compare_digest raises TypeError
    on non-ASCII str, which would turn a Chinese key's 401 into a 500.

    请求出示的是哪一把已配置的代理 Key（按名字）；一把都不匹配时返回 None（U-10）。

    每一把已配置的 Key 都会被比较，不提前返回：首个命中即返回会让响应耗时依赖命中
    Key 在列表中的位置，那是爆破者可以利用的信息。比较本身与以往一样是定长的。
    必须编码成 bytes：compare_digest 比较含非 ASCII 的 str 会抛 TypeError，
    那样一个中文 Key 就会把 401 变成 500。
    """
    if not presented:
        return None
    encoded = presented.encode("utf-8")
    matched: Optional[str] = None
    if settings.proxy_api_key and compare_digest(
        encoded, settings.proxy_api_key.encode("utf-8")
    ):
        matched = "default"
    for name, key in settings.proxy_api_keys.items():
        if compare_digest(encoded, key.encode("utf-8")):
            matched = name
    return matched


def is_proxy_key_valid(request: Request) -> bool:
    """
    Whether the request carries a valid proxy key; always True when auth is
    disabled (neither PROXY_API_KEY nor PROXY_API_KEYS provides one).

    Kept as a separate layer from require_proxy_key so the meta endpoints (/ and
    /healthz) stay accessible without auth, and only use this result to decide
    whether to expose sensitive fields such as the upstream address.

    请求是否携带有效代理 Key；未启用鉴权（PROXY_API_KEY 与 PROXY_API_KEYS 都为空）
    时恒为 True。

    与 require_proxy_key 分成两层：元信息端点（/ 与 /healthz）保持免鉴权
    可访问，只根据这个结果决定要不要暴露上游地址等敏感字段。
    """
    if not settings.authentication_enabled():
        return True
    return match_proxy_key(_presented_proxy_key(request)) is not None


def require_proxy_key(request: Request) -> None:
    if not is_proxy_key_valid(request):
        raise _http_error(401, lang.t("err_invalid_api_key"), code="invalid_api_key")


# --------------------------------------------------------------------------- #
# Errors
# 错误
# --------------------------------------------------------------------------- #
def openai_error(
    message: str,
    status_code: int = 400,
    *,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """
    Return an OpenAI-style error body instead of FastAPI's default {"detail": ...}.

    `headers` carries pass-through response headers -- currently Retry-After from an
    upstream 429, so clients can back off precisely instead of guessing.

    返回 OpenAI 风格的错误体，而不是 FastAPI 默认的 {"detail": ...}。

    `headers` 携带透传的响应头——目前是上游 429 的 Retry-After，
    让客户端能精确退避而不是靠猜。
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
        headers=headers,
    )


@app.exception_handler(HttpError)
async def http_error_handler(request: Request, exc: HttpError) -> JSONResponse:
    return openai_error(exc.message, exc.status_code, headers=exc.headers, **exc.error_fields)


def _upstream_error_message(status: int, text: str) -> str:
    """
    The client-facing message for an upstream error. With EXPOSE_UPSTREAM_ERROR
    disabled (the default) the upstream body stays in the log only, and the client
    gets a fixed sentence plus the request id that ties it to that log line (U-6).

    上游错误对客户端的消息。EXPOSE_UPSTREAM_ERROR 关闭（默认）时上游响应体只进日志，
    客户端收到固定语句 + 可用于在日志里定位这次请求的 request id（U-6）。
    """
    if settings.expose_upstream_error:
        return lang.t("err_upstream_http", status=status, text=text)
    logger.warning(lang.t("err_upstream_http", status=status, text=text))
    return lang.t("err_upstream_http_redacted", status=status) + request_id_suffix()


def _upstream_unavailable_error(exc: UpstreamUnavailable) -> HttpError:
    """
    Translate "the upstream could not be reached/handled" into a client-facing error.

    The local detail (host, port, network stack) goes to the log; the client gets it
    verbatim only when EXPOSE_UPSTREAM_ERROR is on, and otherwise a fixed message plus
    the request id (U-6).

    把"上游连不上/无法处理"翻译成对客户端的错误。

    本地细节（主机、端口、网络栈）进日志；只有在 EXPOSE_UPSTREAM_ERROR 打开时客户端
    才会看到原文，否则收到固定文案 + request id（U-6）。
    """
    logger.warning(lang.t("upstream_unavailable_log", exc=exc))
    if settings.expose_upstream_error:
        message = str(exc)
    else:
        message = lang.t("err_upstream_unavailable_redacted") + request_id_suffix()
    return _http_error(
        exc.status_code, message, error_type="server_error", code="upstream_unavailable"
    )


def _invalid_request_response(exc: UpstreamRequestInvalid) -> JSONResponse:
    """
    A request that could not even be built (a header value the HTTP stack rejects) is a
    bad request, not a broken upstream (B12): answering 400 with the reason keeps the
    diagnosis where it belongs instead of hiding it inside a 500.

    The reason itself may quote the offending header -- that is, the credential -- so
    it goes to the log; the client gets the detail only with EXPOSE_UPSTREAM_ERROR on,
    and a fixed sentence plus the request id otherwise (U-6).

    构造都构造不出来的请求（HTTP 栈拒绝的头值）属于请求有误，而不是上游故障（B12）：
    以 400 加原因作答，把诊断留在它该在的地方，而不是塞进 500 里藏起来。

    原因本身可能引用出问题的请求头——也就是凭证——因此它进日志；只有
    EXPOSE_UPSTREAM_ERROR 打开时客户端才看到细节，否则收到固定语句 + request id（U-6）。
    """
    logger.error(lang.t("upstream_request_invalid", exc=exc))
    if settings.expose_upstream_error:
        message = lang.t("err_upstream_request_invalid", exc=exc)
    else:
        message = lang.t("err_upstream_request_invalid_redacted") + request_id_suffix()
    return openai_error(
        message,
        400,
        error_type="invalid_request_error",
        code="invalid_request",
    )


def _passthrough_retry_headers(resp: httpx.Response) -> Optional[Dict[str, str]]:
    """Retry-After from the upstream (429 and friends) survives the wrapping."""
    retry_after = resp.headers.get("retry-after")
    return {"Retry-After": retry_after} if retry_after else None


def _as_response_headers(pairs: List[Tuple[str, str]]) -> Any:
    """
    Wrap forward_headers' pair list into a Starlette Headers object.

    StreamingResponse only accepts a Mapping (which would silently merge repeated
    headers) or a Headers instance, whose internal list keeps every occurrence. The
    credential headers (Set-Cookie, WWW-Authenticate) have already been dropped by
    forward_headers (U-4); what survives here is ordinary multi-valued headers.

    把 forward_headers 的键值对列表包装成 Starlette 的 Headers 对象。

    StreamingResponse 只接受 Mapping（会把重复头静默合并）或 Headers 实例——
    后者的内部列表保留每一次出现。凭证类响应头（Set-Cookie、WWW-Authenticate）
    已被 forward_headers 剔除（U-4）；这里留下的都是普通的多值头。
    """
    from starlette.datastructures import Headers as StarletteHeaders

    return StarletteHeaders(
        raw=[(name.encode("latin-1"), value.encode("latin-1")) for name, value in pairs]
    )


def _session_or_error() -> Any:
    try:
        return load_session(settings)
    except SessionMissing as exc:
        raise _http_error(
            503,
            str(exc),
            error_type="server_error",
            code="session_missing",
        ) from exc
    except SessionInvalid as exc:
        raise _http_error(
            500,
            str(exc),
            error_type="server_error",
            code="session_invalid",
        ) from exc


def _auth_failure_response(status_code: int) -> JSONResponse:
    logger.error(lang.t("auth_failure_log", status=status_code))
    return openai_error(
        lang.t("err_upstream_unauthorized"),
        status_code,
        error_type="invalid_request_error",
        code="upstream_unauthorized",
    )


# --------------------------------------------------------------------------- #
# Routes: meta info
# 路由：元信息
# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
async def index(request: Request) -> Dict[str, Any]:
    """
    Service metadata. Unauthenticated, but the upstream address is only returned
    when the request passes key validation -- so nobody can use this endpoint to
    map out the internal topology.

    服务元信息。免鉴权，但上游地址只在请求通过 Key 校验时返回——
    避免任何人都能借这个端点摸清内网拓扑。
    """
    response_body: Dict[str, Any] = {
        "service": "open-webui-to-openai-api",
        "version": VERSION,
        "session_ready": session_exists(settings),
        "endpoints": [
            "GET  /healthz",
            "GET  /v1/models",
            lang.t("endpoint_retrieve_model"),
            "POST /v1/chat/completions",
            "POST /v1/embeddings",
            lang.t("endpoint_passthrough"),
        ],
    }
    if is_proxy_key_valid(request):
        response_body["upstream"] = settings.open_webui_base_url
        response_body["upstream_prefix"] = upstream.prefix
    return response_body


@app.get("/healthz", tags=["meta"])
async def healthz(request: Request) -> Dict[str, Any]:
    """
    Health check. Stays unauthenticated and always 200 (probe/load-balancer
    friendly); the upstream address is likewise only returned when the request
    passes key validation.

    With a valid key it also carries the probe health (U-11): whether the last probe
    rounds succeeded, whether the credentials are being rejected, and the last
    recorded failure. Without it, a dead session was only visible as "N probes failed"
    somewhere in the log.

    健康检查。保持免鉴权 200（探针/负载均衡友好），上游地址同样只在
    请求通过 Key 校验时返回。

    带有效 Key 时还会附上探测健康状态（U-11）：最近几轮探测是否成功、凭证是否正被
    拒绝、以及最近一次失败记录。没有它时，死掉的会话只能从日志里某处的"N 个模型探测
    失败"间接推断。
    """
    response_body: Dict[str, Any] = {
        "status": "ok",
        "version": VERSION,
        "session_ready": session_exists(settings),
        "auth_required": settings.authentication_enabled(),
    }
    if is_proxy_key_valid(request):
        response_body["upstream"] = settings.open_webui_base_url
        response_body["upstream_prefix"] = upstream.prefix
        response_body["probe"] = probe_health.to_dict()
    return response_body


# --------------------------------------------------------------------------- #
# Routes: OpenAI compatible
# 路由：OpenAI 兼容
# --------------------------------------------------------------------------- #
def _models_with_probe_fields(
    raw_models: List[Any],
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Normalize the upstream model list and attach everything the probe established.

    Returns the normalized models plus their (id, engine fingerprint) summaries; the
    refresh path reuses the summaries so it does not fetch the list a second time.

    Also refreshes the instance-level default capability template, which comes free
    with the model list.

    规范化上游模型列表，并附上探测已确立的一切。

    返回规范化后的模型，以及它们的 (id, 引擎指纹) 摘要——刷新路径会复用摘要，
    避免第二次拉取模型列表。规范化与指纹推导统一走 _model_summaries，
    与刷新路径共用同一份定义。

    同时刷新实例级的默认能力模板，它是随模型列表免费得到的。
    """
    shared_capabilities = _shared_default_capabilities(raw_models)
    if shared_capabilities:
        _instance_meta.default_model_capabilities = shared_capabilities

    models, summaries = _model_summaries(raw_models, shared_capabilities)

    model_probe.load()
    for model in models:
        presented = model_probe.present(model["id"])
        if presented:
            model.update(presented)
    return models, summaries


@app.get("/v1/models", tags=["openai"])
async def list_models(_: None = Depends(require_proxy_key)) -> Response:
    """
    The model list, normalized to the OpenAI structure, with everything the probe
    established attached to each model and the deployment's own metadata in the
    envelope's `x_open_webui`.

    模型列表：规范化为 OpenAI 结构，每个模型附上探测已确立的信息，信封里的
    `x_open_webui` 承载部署自身的元信息。
    """
    session = _session_or_error()
    raw_models = await _get_raw_models_cached(session)
    await _ensure_instance_meta(session)
    models, summaries = _models_with_probe_fields(raw_models)

    # Models whose facts are not established yet (fresh upstream addition, changed
    # engine fingerprint, expired backoff) get a background refresh. /v1/models waits
    # for it only when a probe for one of them is genuinely in flight -- a model in
    # backoff, or one the upstream never validates, is served immediately.
    #
    # 尚未确立事实的模型（上游新增、引擎指纹变化、退避已过期）触发后台刷新。只有当
    # 确实有它们的探测在飞行中时 /v1/models 才等待——处于退避中、或上游从不校验的
    # 模型立即返回。
    pending_ids = [
        model_id
        for model_id, fingerprint in summaries
        if model_probe.needs_probe(model_id, fingerprint)
    ]
    if pending_ids:
        refresh_task = _spawn_model_probe_refresh(summaries=summaries)
        if refresh_task is not None and settings.model_probe_wait > 0:
            # Wait for the refresh to announce what it is probing. An explicit event
            # instead of sleep(0): the handshake must survive async steps being added
            # anywhere between task creation and the announcement. Bounded so a
            # wedged refresh degrades to "serve immediately".
            #
            # 等待刷新公布它正在探测什么。用显式事件而非 sleep(0)：从任务创建到
            # 公布之间无论新增多少异步步骤，握手都不会静默失效。加上限时，
            # 刷新卡死时退化为"立即返回"。
            try:
                await asyncio.wait_for(
                    _refresh_state.announcement().wait(), timeout=_PROBE_ANNOUNCE_TIMEOUT
                )
            except asyncio.TimeoutError:
                pass
            if _refresh_state.pending & set(pending_ids):
                try:
                    # shield: on timeout only the *wait* is cancelled, the refresh
                    # task keeps running for later requests.
                    #
                    # shield：超时只取消"等待"，刷新任务本身继续运行。
                    await asyncio.wait_for(
                        asyncio.shield(refresh_task), timeout=settings.model_probe_wait
                    )
                except asyncio.TimeoutError:
                    logger.warning(lang.t("probe_wait_timeout", wait=settings.model_probe_wait))
                # Merge again: models probed during the wait are now covered
                # 再合并一轮：等待期间探测完成的模型现在已有缓存
                models, _ = _models_with_probe_fields(raw_models)

    envelope: Dict[str, Any] = {"object": "list", "data": models}
    if settings.expose_instance_meta and _instance_meta.is_usable():
        envelope["x_open_webui"] = _instance_meta.to_dict()
    return JSONResponse(content=envelope)


@app.get("/v1/models/{model_id:path}", tags=["openai"])
async def retrieve_model(model_id: str, _: None = Depends(require_proxy_key)) -> Response:
    """
    OpenAI's "retrieve model": one normalized model object, or a 404.

    Implemented locally on purpose. The catch-all passthrough used to forward this to
    the upstream, where /api/v1/models/<id> is not a route and Open WebUI's SPA
    answers with HTTP 200 and an HTML page -- a 200 no OpenAI client can parse.

    OpenAI 的 "retrieve model"：返回单个规范化模型对象，找不到则 404。

    特意在本地实现。兜底透传以前会把它转发到上游，而 /api/v1/models/<id> 并不是
    路由，Open WebUI 的 SPA 会用 HTTP 200 + 一页 HTML 回答——一个任何 OpenAI
    客户端都无法解析的 200。
    """
    session = _session_or_error()
    raw_models = await _get_raw_models_cached(session)
    models, _ = _models_with_probe_fields(raw_models)
    for model in models:
        if model["id"] == model_id:
            return JSONResponse(content=model)
    return openai_error(
        lang.t("err_model_not_found", model=model_id),
        404,
        code="model_not_found",
        param="model",
    )


@app.post("/v1/chat/completions", tags=["openai"])
async def chat_completions(request: Request, _: None = Depends(require_proxy_key)) -> Response:
    session = _session_or_error()
    payload = await _read_json_body(request)
    _validate_chat_payload(payload)

    # Model aliases (optional)
    # 模型别名（可选）
    if payload.get("model"):
        payload["model"] = settings.resolve_model(payload["model"])

    # Strict identity, not truthiness (U-5): `{"stream": "false"}` (a non-empty string),
    # `{"stream": 1}` and `{"stream": []}` all used to flip the request into the
    # streaming branch, and the client then waited for SSE it never asked for. Only a
    # real JSON `true` streams -- which is what every OpenAI client sends.
    #
    # 严格判等，不用真值（U-5）：`{"stream": "false"}`（非空字符串）、`{"stream": 1}`、
    # `{"stream": []}` 过去都会把请求翻进流式分支，客户端于是等一个它从没要求的 SSE。
    # 只有真正的 JSON `true` 才走流式——那正是所有 OpenAI 客户端的发法。
    is_stream = payload.get("stream") is True
    logger.debug(lang.t("forward_chat", model=payload.get("model"), stream=is_stream))

    try:
        resp = await upstream.post(session, "chat/completions", payload, stream=is_stream)
    except UpstreamUnavailable as exc:
        raise _upstream_unavailable_error(exc) from exc
    except UpstreamRequestInvalid as exc:
        return _invalid_request_response(exc)

    if resp.status_code in AUTH_FAILURE_CODES:
        await resp.aclose()
        return _auth_failure_response(resp.status_code)

    if resp.status_code >= 400:
        text = await _safe_read(resp)
        await resp.aclose()
        # A 400 that blames the reasoning effort disproves what we advertised for
        # this model: drop the level and re-probe in the background. The error the
        # client gets is unchanged, and this request is not delayed by it.
        #
        # 归咎于思考挡位的 400 证伪了我们为该模型声明的内容：剔除该挡位并在后台重探。
        # 客户端收到的错误保持原样，这个请求也不会因此被拖延。
        if resp.status_code in VALIDATION_FAILURE_CODES and looks_like_effort_error(text):
            requested_effort = payload.get("reasoning_effort")
            _trigger_probe_heal(
                payload.get("model"),
                requested_effort if isinstance(requested_effort, str) else None,
            )
        return openai_error(
            _upstream_error_message(resp.status_code, text),
            resp.status_code if resp.status_code < 500 else 502,
            error_type="invalid_request_error" if resp.status_code < 500 else "server_error",
            code="upstream_error",
            headers=_passthrough_retry_headers(resp),
        )

    if not is_stream:
        raw_body = await resp.aread()
        await resp.aclose()
        try:
            return JSONResponse(content=json.loads(raw_body))
        except ValueError:
            # Same policy as the embeddings route (U-6): the body may quote upstream
            # internals, so it goes to the log and the client gets a fixed sentence.
            #
            # 与 embeddings 路由同一套策略（U-6）：响应体可能引用上游内部信息，
            # 因此它进日志，客户端收到固定语句。
            logger.warning(lang.t("err_upstream_not_json", body=raw_body[:500]))
            message = (
                lang.t("err_upstream_not_json", body=raw_body[:500])
                if settings.expose_upstream_error
                else lang.t("err_upstream_not_json_redacted") + request_id_suffix()
            )
            return openai_error(
                message,
                502,
                error_type="server_error",
                code="upstream_error",
            )

    media_type = resp.headers.get("content-type", "text/event-stream")
    # Only the headers this proxy really owns are added. Connection and friends are
    # hop-by-hop and stay the protocol layer's business (uvicorn decides on reuse);
    # stripping them from the upstream response only to re-add one here would be
    # self-contradictory.
    #
    # 只补上本代理真正拥有的头。Connection 之类的逐跳头属于协议层（是否复用由 uvicorn 决定），
    # 从上游响应里剔除却又在这里加回来，属于自相矛盾。
    headers = _as_response_headers(
        UpstreamClient.forward_headers(
            resp,
            {
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    )
    return StreamingResponse(
        _sse_iterator(resp, request),
        status_code=resp.status_code,
        media_type=media_type,
        headers=headers,
    )


@app.post("/v1/embeddings", tags=["openai"])
async def embeddings(request: Request, _: None = Depends(require_proxy_key)) -> Response:
    session = _session_or_error()
    payload = await _read_json_body(request)
    model = payload.get("model")
    if not model or (isinstance(model, str) and not model.strip()) or "input" not in payload:
        return openai_error(
            lang.t("err_missing_model_input"), 400, code="missing_required_field"
        )
    # Same guard as chat/completions: a non-string model must not reach resolve_model.
    # 与 chat/completions 相同的防护：非字符串的 model 不得进入 resolve_model。
    if not isinstance(model, str):
        return openai_error(
            lang.t("err_model_not_string"), 400, code="invalid_type", param="model"
        )
    payload["model"] = settings.resolve_model(model)

    try:
        resp = await upstream.post(session, "embeddings", payload, stream=False)
    except UpstreamUnavailable as exc:
        raise _upstream_unavailable_error(exc) from exc
    except UpstreamRequestInvalid as exc:
        return _invalid_request_response(exc)

    if resp.status_code in AUTH_FAILURE_CODES:
        await resp.aclose()
        return _auth_failure_response(resp.status_code)

    # Consistent with chat/completions: upstream errors must also be wrapped into an
    # OpenAI-style error body
    #
    # 与 chat/completions 保持一致：上游报错也要包成 OpenAI 风格错误体
    if resp.status_code >= 400:
        text = await _safe_read(resp)
        await resp.aclose()
        return openai_error(
            _upstream_error_message(resp.status_code, text),
            resp.status_code if resp.status_code < 500 else 502,
            error_type="invalid_request_error" if resp.status_code < 500 else "server_error",
            code="upstream_error",
            headers=_passthrough_retry_headers(resp),
        )

    raw_body = await resp.aread()
    await resp.aclose()
    try:
        content = json.loads(raw_body)
    except ValueError:
        logger.warning(lang.t("err_upstream_not_json", body=raw_body[:500]))
        message = (
            lang.t("err_upstream_not_json", body=raw_body[:500])
            if settings.expose_upstream_error
            else lang.t("err_upstream_not_json_redacted") + request_id_suffix()
        )
        return openai_error(
            message,
            502,
            error_type="server_error",
            code="upstream_error",
        )
    return JSONResponse(status_code=resp.status_code, content=content)


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    tags=["openai"],
    # Multiple methods sharing one function would produce duplicate operationIds
    # 多方法共用同一函数会产生重复的 operationId
    include_in_schema=False,
)
async def v1_passthrough(path: str, request: Request, _: None = Depends(require_proxy_key)) -> Response:
    """Catch-all passthrough: forward unimplemented /v1/* requests to the
    corresponding upstream path as-is.

    兜底透传：把未单独实现的 /v1/* 请求原样转发到上游对应路径。
    """
    if not path.strip("/"):
        return openai_error(lang.t("err_passthrough_path"), 404, code="not_found")

    # U-1: the catch-all forwards with the operator's captured credentials attached, so
    # it now sits behind an allowlist. The default covers the OpenAI-style add-on routes
    # (images/audio/files/responses); PASSTHROUGH_ALLOW extends or shrinks it, and
    # PASSTHROUGH_ALLOW=* is the explicit opt-in to the historical forward-everything
    # behavior.
    #
    # U-1：兜底透传会带着运维抓到的凭证转发，因此现在受白名单约束。默认覆盖
    # OpenAI 风格的附加路由（images/audio/files/responses）；PASSTHROUGH_ALLOW 可增可减，
    # PASSTHROUGH_ALLOW=* 是对历史上"全量透传"行为的显式选择。
    if not settings.passthrough_permits(path):
        logger.warning(lang.t("passthrough_forbidden_log", path=path))
        return openai_error(
            lang.t("passthrough_forbidden"), 403, code="passthrough_forbidden"
        )

    session = _session_or_error()
    headers = session.to_headers()
    # The client-declared Content-Type must override the default from to_headers
    # 客户端声明的 Content-Type 必须覆盖 to_headers 的默认值
    content_type = request.headers.get("content-type", "")
    if content_type:
        headers["Content-Type"] = content_type
    request_body = await request.body()
    subpath = f"{path}?{request.url.query}" if request.url.query else path

    # Consistent with chat/embeddings: fall back to the other candidate prefixes
    # when the primary prefix returns 404 (route not found)
    #
    # 与 chat/embeddings 一致：主前缀 404（无此路由）时自动回退其它候选前缀
    try:
        resp = await upstream.forward(
            session, request.method, subpath, headers=headers, content=request_body or None, stream=True
        )
    except UpstreamUnavailable as exc:
        raise _upstream_unavailable_error(exc) from exc
    except UpstreamRequestInvalid as exc:
        return _invalid_request_response(exc)

    if resp.status_code in AUTH_FAILURE_CODES:
        await resp.aclose()
        return _auth_failure_response(resp.status_code)

    media_type = resp.headers.get("content-type", "application/json")
    stream_headers = _as_response_headers(
        UpstreamClient.forward_headers(resp, {"X-Accel-Buffering": "no"})
    )
    return StreamingResponse(
        _sse_iterator(resp, request),
        status_code=resp.status_code,
        media_type=media_type,
        headers=stream_headers,
    )


# --------------------------------------------------------------------------- #
# Helpers
# 辅助
# --------------------------------------------------------------------------- #
async def _read_json_body(request: Request) -> Dict[str, Any]:
    """
    Read and parse the JSON request body under a hard size cap (U-2).

    The previous version only looked at a numeric Content-Length, so a chunked request
    -- which declares no length at all -- was read into memory in full before the cap
    could apply. The body is now consumed as a stream and the running total is enforced
    on every chunk, so the cap holds for every request shape; the declared length is
    still checked first because it rejects an oversized body without reading it.

    在硬性大小上限之下读取并解析 JSON 请求体（U-2）。

    旧实现只看纯数字的 Content-Length，因此不带长度声明的 chunked 请求会被完整读进
    内存之后上限才可能生效。现在改为流式消费请求体，并在每个分块上核对累计字节数，
    使上限对任何请求形态都成立；声明长度仍先检查——它能在不读取的前提下直接拒收超大请求体。
    """
    declared_length = request.headers.get("content-length", "")
    if declared_length.isdigit() and int(declared_length) > settings.max_body_bytes:
        raise _http_error(
            413,
            lang.t("err_body_too_large", limit=settings.max_body_bytes),
            code="body_too_large",
        )

    chunks: List[bytes] = []
    total = 0
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > settings.max_body_bytes:
                # Stop reading: leaving the rest of an oversized body on the wire is the
                # point -- buffering it first is exactly what this cap exists to prevent.
                #
                # 立即停止读取：把超限请求体的剩余部分留在网络上正是重点——先缓冲下来
                # 恰恰是这个上限要防的事。
                raise _http_error(
                    413,
                    lang.t("err_body_too_large", limit=settings.max_body_bytes),
                    code="body_too_large",
                )
            chunks.append(chunk)
    except ClientDisconnect as exc:
        # The client went away mid-body. Without this, the disconnect surfaces as an
        # unhandled ASGI-level error and lands in the log as a crash, which it is not.
        #
        # 客户端在发送请求体途中断开。若不处理，这次断开会以未处理的 ASGI 层错误浮现，
        # 在日志里被记成一次崩溃——而它并不是。
        raise _http_error(
            400, lang.t("err_client_disconnected"), code="client_disconnected"
        ) from exc

    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise _http_error(400, lang.t("err_invalid_json"), code="invalid_json")
    if not isinstance(payload, dict):
        raise _http_error(400, lang.t("err_json_object"), code="invalid_json")
    return payload


def _validate_chat_payload(payload: Dict[str, Any]) -> None:
    model = payload.get("model")
    if not model or (isinstance(model, str) and not model.strip()):
        raise _http_error(400, lang.t("err_missing_model"), code="missing_required_field", param="model")
    # A non-string model (a malformed body may carry a dict/list) must be rejected here:
    # it has no alias semantics and is not hashable, so it would otherwise blow up in
    # resolve_model as a TypeError -> HTTP 500.
    #
    # 非字符串的 model（畸形请求体里可能是 dict/list）必须在这里拦下：它没有别名语义、
    # 也不可哈希，否则会在 resolve_model 里抛 TypeError，变成 HTTP 500。
    if not isinstance(model, str):
        raise _http_error(400, lang.t("err_model_not_string"), code="invalid_type", param="model")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _http_error(400, lang.t("err_messages_empty"), code="missing_required_field", param="messages")


async def _safe_read(resp: httpx.Response, limit: int = 2000) -> str:
    try:
        data = await resp.aread()
    except Exception:  # pragma: no cover
        return lang.t("err_cannot_read_body")
    return data.decode("utf-8", errors="replace")[:limit]


async def _sse_iterator(resp: httpx.Response, request: Request):
    """
    Forward the upstream byte stream; close the upstream connection proactively
    when the client disconnects, so nothing hangs.

    The upstream dropping the connection mid-stream (process restart, gateway
    timeout, read timeout) is common in long conversations. Those exceptions are
    subclasses of httpx.TransportError (RemoteProtocolError / ReadTimeout /
    ReadError), not StreamError, so both must be caught: otherwise the exception
    reaches the ASGI layer, the client loses even the content it already received,
    and the server leaves a pile of tracebacks behind.

    转发上游字节流；客户端断开时主动关闭上游连接，避免悬挂。

    上游中途掐断连接（进程重启、网关超时、读超时）在长对话里很常见，这类异常
    抛的是 httpx.TransportError 的子类（RemoteProtocolError / ReadTimeout /
    ReadError），并不是 StreamError，必须一并兜住：否则异常会穿透到 ASGI 层，
    客户端连已经收到的部分内容都拿不到，服务端还会留下一大串 traceback。
    """
    try:
        # aiter_bytes() rather than aiter_raw(): httpx advertises gzip/deflate/br/zstd
        # and decodes them here, which is exactly what forward_headers promises when it
        # drops the upstream's Content-Encoding. Forwarding the raw bytes while
        # stripping that header would hand the client a compressed body it cannot
        # decode -- and /v1/* passthrough JSON is routinely compressed upstream.
        #
        # 用 aiter_bytes() 而不是 aiter_raw()：httpx 会自动协商并在此解码
        # gzip/deflate/br/zstd，这正是 forward_headers 丢弃上游 Content-Encoding 时
        # 所承诺的。若一边丢掉该头、一边转发未解码的原始字节，客户端拿到的响应将无法
        # 解析——而 /v1/* 兜底透传的 JSON 在上游通常就是被压缩的。
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            if await request.is_disconnected():
                logger.debug(lang.t("sse_client_disconnected"))
                break
            yield chunk
    except (httpx.StreamError, httpx.HTTPError) as exc:
        logger.warning(lang.t("sse_stream_ended", etype=type(exc).__name__, exc=exc))
    finally:
        if not resp.is_closed:
            await resp.aclose()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
async def _check_session() -> int:
    if not session_exists(settings):
        logger.error(lang.t("check_session_missing", path=settings.session_file))
        return 1
    try:
        session = load_session(settings)
    except SessionError as exc:
        logger.error(lang.t("creds_unusable", exc=exc))
        return 1

    logger.info(lang.t("check_summary", desc=session.describe()))
    # U-11: --check is exactly when an operator wants to know what the persisted probe
    # state says -- including the reason the last round failed, which previously was
    # only visible per-model inside the cache file.
    #
    # U-11：运维正需要在这时候知道磁盘上的探测状态是什么——包括上一轮失败的原因，
    # 它此前只出现在缓存文件里、按模型逐条记录。
    logger.info(lang.t("check_probe_cache", **probe_cache_status()))
    return 0 if await _startup_check(quiet_success=True) else 1


async def _run_check_cli() -> int:
    """--check with a bounded client lifetime (B1): the shared httpx client created
    by the probe must be closed before asyncio.run() tears the loop down."""
    try:
        return await _check_session()
    finally:
        await upstream.aclose()


async def _run_probe_cli() -> int:
    """--probe with a bounded client lifetime (B1): probes create the shared client
    many times over; leaving it open until loop shutdown leaks warnings on exit."""
    try:
        return 0 if await _refresh_model_probe(force=True) else 1
    finally:
        await upstream.aclose()


def _preconfigure_language(cli_args: List[str]) -> None:
    """
    Honor --lang for argparse's own output (--help): argparse exits during parsing,
    before the post-parse reconfigure in main(), so argv is scanned up front. Both
    the space form (--lang zh) and the equals form (--lang=zh) are recognized.

    让 --lang 也能作用于 argparse 自身的输出（--help）：argparse 在解析阶段就会退出，
    早于 main() 里解析后的重配置，因此先扫一遍 argv。空格形式（--lang zh）与
    等号形式（--lang=zh）都能识别。
    """
    for index, token in enumerate(cli_args):
        name, separator, value = token.partition("=")
        if name != "--lang":
            continue
        if separator:
            lang.configure(lang.resolve_language(value))
        elif index + 1 < len(cli_args):
            lang.configure(lang.resolve_language(cli_args[index + 1]))
        break


def main(argv: Optional[List[str]] = None) -> int:
    # --help exits during parsing, before the post-parse reconfigure below, so peek
    # at argv first to honor --lang for the help text as well.
    #
    # --help 会在参数解析阶段直接退出，晚于下方的重配置；因此先扫一遍 argv，
    # 让 --help 的输出也能跟随 --lang。
    cli_args = list(sys.argv[1:] if argv is None else argv)
    _preconfigure_language(cli_args)

    parser = argparse.ArgumentParser(description=lang.t("cli_description"))
    parser.add_argument("--login", action="store_true", help=lang.t("cli_login_help"))
    parser.add_argument("--check", action="store_true", help=lang.t("cli_check_help"))
    parser.add_argument("--probe", action="store_true", help=lang.t("cli_probe_help"))
    parser.add_argument("--host", default=None, help=lang.t("cli_host_help"))
    parser.add_argument("--port", type=int, default=None, help=lang.t("cli_port_help"))
    parser.add_argument(
        "--lang",
        choices=(lang.LANG_ZH, lang.LANG_EN, lang.LANG_AUTO),
        default=lang.LANG_AUTO,
        help=lang.t("cli_lang_help"),
    )
    args = parser.parse_args(argv)

    # --host/--port/--lang require rebuilding settings (the dataclass is frozen).
    # B3: falsy CLI values must not be swallowed -- "--port 0" is an explicit value,
    # not "unset". The rebuilt singletons are mirrored into probe_runner, which owns
    # the runtime state the orchestration functions actually read.
    #
    # --host/--port/--lang 需要重新构造 settings（dataclass 是 frozen 的）。
    # B3：falsy 的 CLI 值不能被吞掉——"--port 0" 是显式赋值，不是"未提供"。
    # 重建后的单例同步到 probe_runner——编排函数读取的运行时状态以它为准。
    global settings, upstream
    if args.host is not None or args.port is not None or args.lang != lang.LANG_AUTO:
        import dataclasses

        settings = dataclasses.replace(
            settings,
            proxy_host=args.host if args.host is not None else settings.proxy_host,
            proxy_port=args.port if args.port is not None else settings.proxy_port,
            language=lang.resolve_language(args.lang),
        )
        upstream = UpstreamClient(settings)
        probe_runner.settings = settings
        probe_runner.upstream = upstream

    # Apply the effective language before any user-facing output (logs, banner, errors)
    # 在产生任何用户可见输出（日志、横幅、错误）之前应用生效语言
    lang.configure(settings.language)
    configure_logging(settings)

    if args.check:
        return asyncio.run(_run_check_cli())

    if args.probe:
        # Force a full probe refresh, synchronously, then exit.
        # 强制完整刷新一次探测缓存（同步等待），然后退出。
        if not session_exists(settings):
            logger.error(lang.t("check_session_missing", path=settings.session_file))
            return 1
        try:
            load_session(settings)
        except SessionError as exc:
            logger.error(lang.t("creds_unusable", exc=exc))
            return 1
        return asyncio.run(_run_probe_cli())

    if args.login or not session_exists(settings):
        try:
            asyncio.run(perform_browser_login(settings))
        except SessionError as exc:
            logger.error("%s", exc)
            return 1

    uvicorn.run(
        app,
        host=settings.proxy_host,
        port=settings.proxy_port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
