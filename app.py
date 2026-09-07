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
import re
import sys
from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Any, Dict, List, Optional

import lang
import httpx
import uvicorn
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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
from upstream import UpstreamClient, UpstreamUnavailable

logger = logging.getLogger("webui-proxy")

VERSION = "1.0.0"

# The upstream returning these status codes means the credentials are dead and re-login is needed
# 上游返回这几个状态码说明凭证失效，需要重新登录
AUTH_FAILURE_CODES = (401, 403)


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
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


upstream = UpstreamClient(settings)


# --------------------------------------------------------------------------- #
# Lifecycle
# 生命周期
# --------------------------------------------------------------------------- #
async def _startup_check() -> bool:
    """Validate credentials and upstream connectivity at startup.

    启动时校验凭证与上游连通性。返回 True 表示凭证可用。
    """
    if not settings.proxy_api_key:
        logger.warning(lang.t("no_proxy_key"))

    if not session_exists(settings):
        logger.warning(
            lang.t("session_missing_hint", path=settings.session_file),
        )
        return False

    try:
        session = load_session(settings)
    except SessionError as exc:
        logger.warning(lang.t("creds_unusable", exc=exc))
        return False

    # The probe request itself is an authenticated GET /models, so a single request
    # performs both prefix discovery and credential validation (the old implementation
    # called detect_prefix + get_models and hit the upstream twice).
    #
    # 探测请求本身就是一次带凭证的 GET /models，一次请求同时完成
    # 找前缀 + 验凭证（旧实现 detect_prefix + get_models 会打两次）。
    try:
        prefix, status = await upstream.probe_models(session)
    except UpstreamUnavailable as exc:
        logger.warning(lang.t("startup_cant_connect", exc=exc))
        return False

    if status in AUTH_FAILURE_CODES:
        logger.error(
            lang.t("creds_expired", status=status)
        )
        return False

    if status == 404:
        # The old behavior misreported a 404 as "credentials valid"; give a clear error here
        # 旧行为会把 404 误报成"凭证校验通过"，这里给出明确错误
        logger.error(
            lang.t("models_404", prefix=prefix)
        )
        return False

    logger.info(lang.t("creds_ok", status=status, desc=session.describe()))
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings)
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
    await _startup_check()
    try:
        yield
    finally:
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
# Authentication
# 鉴权
# --------------------------------------------------------------------------- #
def _presented_proxy_key(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[len("Bearer ") :].strip()
    return request.headers.get("X-API-Key", "").strip()


def key_is_valid(request: Request) -> bool:
    """
    Whether the request carries a valid proxy key; always True when auth is
    disabled (PROXY_API_KEY empty).

    Kept as a separate layer from verify_proxy_key so the meta endpoints
        (/ and /healthz) stay accessible without auth, and only use this result to
        decide whether to expose sensitive fields such as the upstream address.
    
    请求是否携带有效代理 Key；未启用鉴权（PROXY_API_KEY 为空）时恒为 True。

    与 verify_proxy_key 分成两层：元信息端点（/ 与 /healthz）保持免鉴权
    可访问，只根据这个结果决定要不要暴露上游地址等敏感字段。
    """
    if not settings.proxy_api_key:
        return True
    presented = _presented_proxy_key(request)
    # Use a constant-time comparison to prevent byte-by-byte key brute-forcing via
    # response timing. Must encode to bytes: compare_digest raises TypeError on
    # non-ASCII str, which would turn a Chinese key's 401 into a 500.
    #
    # 用定长时间比较，避免通过响应耗时逐字节爆破 Key。
    # 必须编码成 bytes：compare_digest 比较含非 ASCII 的 str 会抛 TypeError，
    # 那样一个中文 Key 就会把 401 变成 500。
    return bool(presented) and compare_digest(
        presented.encode("utf-8"), settings.proxy_api_key.encode("utf-8")
    )


def verify_proxy_key(request: Request) -> None:
    if not key_is_valid(request):
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
) -> JSONResponse:
    """
    Return an OpenAI-style error body instead of FastAPI's default {"detail": ...}.

    返回 OpenAI 风格的错误体，而不是 FastAPI 默认的 {"detail": ...}。
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
    )


class HttpError(Exception):
    def __init__(self, status_code: int, message: str, **kwargs: Any):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.kwargs = kwargs


def _http_error(status_code: int, message: str, **kwargs: Any) -> HttpError:
    return HttpError(status_code, message, **kwargs)


@app.exception_handler(HttpError)
async def http_error_handler(request: Request, exc: HttpError) -> JSONResponse:
    return openai_error(exc.message, exc.status_code, **exc.kwargs)


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
# Model list normalization
# 模型列表规范化
# --------------------------------------------------------------------------- #
# Quantization tokens recognizable in model ids: NVFP4, FP8, FP16, INT8, GPTQ, AWQ, ...
# 模型名中可识别的量化标识：NVFP4、FP8、FP16、INT8、GPTQ、AWQ 等
_QUANT_PATTERN = re.compile(
    r"\b(NVFP4|FP4|FP8|FP16|INT8|INT4|GPTQ(?:-?[0-9]+BIT)?|AWQ|GGUF|Q[0-9](?:_[A-Z0-9]+)*)\b",
    re.IGNORECASE,
)


def normalize_model(raw: Any) -> Optional[Dict[str, Any]]:
    """Collapse an upstream model object into the OpenAI model structure.

    Standard fields stay intact; a whitelist of safe, useful extras aligned
    with the generic /v1/models template is preserved when present:
    max_model_len (kept for compatibility) plus max_context_length and
    context_length, quantization (parsed from the model id), capabilities
    (with a derived function_calling flag) and description. Private upstream
    fields (user_id, access_grants, permission, urlIdx, ...) are never exposed.

    把上游的模型对象收敛成 OpenAI 的 model 结构。

    标准字段原样保留，另有一份白名单按通用 /v1/models 模板透出安全且
    有用的扩展字段：max_model_len（兼容保留）+ max_context_length/
    context_length、quantization（从模型名解析）、capabilities（含派生的
    function_calling）与 description；上游私有字段（user_id、access_grants、
    permission、urlIdx 等）一律不透出。
    """
    if isinstance(raw, str):
        return {"id": raw, "object": "model", "created": 0, "owned_by": "openai"}

    if not isinstance(raw, dict):
        return None

    model_id = raw.get("id") or raw.get("name") or raw.get("model")
    if not model_id:
        return None

    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
    openai_obj = raw.get("openai") if isinstance(raw.get("openai"), dict) else {}

    # info.created_at is the model's real creation time; the "created" on the
    # OpenAI layer is the serving engine's start time, not the model's.
    #
    # info.created_at 才是模型真实创建时间；OpenAI 层的 created 是推理引擎
    # 的启动时间，并非模型本身的。
    created = info.get("created_at")
    if created is None:
        created = raw.get("created")
    if created is None:
        created = raw.get("created_at")
    try:
        created = int(created)
    except (TypeError, ValueError):
        created = 0

    # Prefer the inner engine attribution (e.g. "vllm") over the OpenAI-layer default
    # 优先取内层引擎归属（如 "vllm"），而非 OpenAI 层的默认值
    owned_by = openai_obj.get("owned_by") or raw.get("owned_by") or "openai"

    model: Dict[str, Any] = {
        "id": str(model_id),
        "object": "model",
        "created": created,
        "owned_by": str(owned_by),
    }

    # Whitelisted extras: only emitted when the upstream provides them, so
    # minimal/legacy model objects keep the exact 4-field OpenAI shape.
    #
    # 白名单扩展字段：上游提供时才输出，极简/老版本模型对象仍保持
    # 精确的 4 字段 OpenAI 结构。
    max_model_len = raw.get("max_model_len") or openai_obj.get("max_model_len")
    try:
        context_length = int(max_model_len)
    except (TypeError, ValueError):
        context_length = None
    if context_length is not None:
        # Generic-template field names; max_model_len stays as a compatibility alias
        # 通用模板字段名；max_model_len 作为兼容别名保留
        model["max_model_len"] = context_length
        model["max_context_length"] = context_length
        model["context_length"] = context_length

    # Quantization is not a dedicated upstream field; parse it from the model id
    # (e.g. "GLM-5.2-NVFP4" -> "NVFP4"). Omitted when nothing matches.
    #
    # 量化信息不是上游的独立字段，从模型名解析（如 "GLM-5.2-NVFP4" ->
    # "NVFP4"）。匹配不到时不输出该字段。
    quant_match = _QUANT_PATTERN.search(str(model_id))
    if quant_match:
        model["quantization"] = quant_match.group(1).upper()

    description = meta.get("description")
    if description:
        model["description"] = str(description)

    capabilities = meta.get("capabilities")
    if isinstance(capabilities, dict):
        caps = {k: v for k, v in capabilities.items() if isinstance(v, bool)}
        if caps:
            # Derived flag: builtin_tools maps onto the template's function_calling
            # 派生字段：builtin_tools 对应通用模板的 function_calling
            caps["function_calling"] = bool(caps.get("builtin_tools", False))
            model["capabilities"] = caps

    return model


def extract_model_list(payload: Any) -> List[Any]:
    """
    Upstream versions return inconsistent shapes: {"data": [...]} / {"items": [...]} / a bare list.

    上游不同版本返回结构不一致：{"data": [...]} / {"items": [...]} / 裸列表。
    """
    if isinstance(payload, dict):
        for key in ("data", "items", "models"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(payload, list):
        return payload
    return []


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
    body: Dict[str, Any] = {
        "service": "open-webui-to-openai-api",
        "version": VERSION,
        "session_ready": session_exists(settings),
        "endpoints": [
            "GET  /healthz",
            "GET  /v1/models",
            "POST /v1/chat/completions",
            "POST /v1/embeddings",
            lang.t("endpoint_passthrough"),
        ],
    }
    if key_is_valid(request):
        body["upstream"] = settings.open_webui_base_url
        body["upstream_prefix"] = upstream.prefix
    return body


@app.get("/healthz", tags=["meta"])
async def healthz(request: Request) -> Dict[str, Any]:
    """
    Health check. Stays unauthenticated and always 200 (probe/load-balancer
    friendly); the upstream address is likewise only returned when the request
    passes key validation.

    健康检查。保持免鉴权 200（探针/负载均衡友好），上游地址同样只在
    请求通过 Key 校验时返回。
    """
    body: Dict[str, Any] = {
        "status": "ok",
        "version": VERSION,
        "session_ready": session_exists(settings),
        "auth_required": bool(settings.proxy_api_key),
    }
    if key_is_valid(request):
        body["upstream"] = settings.open_webui_base_url
        body["upstream_prefix"] = upstream.prefix
    return body


# --------------------------------------------------------------------------- #
# Routes: OpenAI compatible
# 路由：OpenAI 兼容
# --------------------------------------------------------------------------- #
@app.get("/v1/models", tags=["openai"])
async def list_models(_: None = Depends(verify_proxy_key)) -> Response:
    session = _session_or_error()
    try:
        resp = await upstream.get_models(session)
    except UpstreamUnavailable as exc:
        raise _http_error(exc.status_code, str(exc), error_type="server_error", code="upstream_unavailable") from exc

    if resp.status_code in AUTH_FAILURE_CODES:
        await resp.aclose()
        return _auth_failure_response(resp.status_code)

    if resp.status_code != 200:
        text = resp.text[:500]
        await resp.aclose()
        return openai_error(
            lang.t("err_upstream_models_http", status=resp.status_code, text=text),
            502,
            error_type="server_error",
            code="upstream_error",
        )

    try:
        payload = resp.json()
    except ValueError:
        text = resp.text[:500]
        await resp.aclose()
        return openai_error(
            lang.t("err_upstream_models_not_json", text=text),
            502,
            error_type="server_error",
            code="upstream_error",
        )
    finally:
        if not resp.is_closed:
            await resp.aclose()

    models = [m for m in (normalize_model(x) for x in extract_model_list(payload)) if m]
    return JSONResponse(content={"object": "list", "data": models})


@app.post("/v1/chat/completions", tags=["openai"])
async def chat_completions(request: Request, _: None = Depends(verify_proxy_key)) -> Response:
    session = _session_or_error()
    payload = await _read_json_body(request)
    _validate_chat_payload(payload)

    # Model aliases (optional)
    # 模型别名（可选）
    if payload.get("model"):
        payload["model"] = settings.resolve_model(payload["model"])

    is_stream = bool(payload.get("stream"))
    logger.debug(lang.t("forward_chat", model=payload.get("model"), stream=is_stream))

    try:
        resp = await upstream.post("chat/completions", session, payload, stream=is_stream)
    except UpstreamUnavailable as exc:
        raise _http_error(exc.status_code, str(exc), error_type="server_error", code="upstream_unavailable") from exc

    if resp.status_code in AUTH_FAILURE_CODES:
        await resp.aclose()
        return _auth_failure_response(resp.status_code)

    if resp.status_code >= 400:
        text = await _safe_read(resp)
        await resp.aclose()
        return openai_error(
            lang.t("err_upstream_http", status=resp.status_code, text=text),
            resp.status_code if resp.status_code < 500 else 502,
            error_type="invalid_request_error" if resp.status_code < 500 else "server_error",
            code="upstream_error",
        )

    if not is_stream:
        body = await resp.aread()
        await resp.aclose()
        try:
            return JSONResponse(content=json.loads(body))
        except ValueError:
            return openai_error(
                lang.t("err_upstream_not_json", body=body[:500]),
                502,
                error_type="server_error",
                code="upstream_error",
            )

    media_type = resp.headers.get("content-type", "text/event-stream")
    headers = UpstreamClient.forward_headers(
        resp,
        {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    return StreamingResponse(
        _sse_iterator(resp, request),
        status_code=resp.status_code,
        media_type=media_type,
        headers=headers,
    )


@app.post("/v1/embeddings", tags=["openai"])
async def embeddings(request: Request, _: None = Depends(verify_proxy_key)) -> Response:
    session = _session_or_error()
    payload = await _read_json_body(request)
    if not payload.get("model") or "input" not in payload:
        return openai_error(
            lang.t("err_missing_model_input"), 400, code="missing_required_field"
        )
    payload["model"] = settings.resolve_model(payload.get("model"))

    try:
        resp = await upstream.post("embeddings", session, payload, stream=False)
    except UpstreamUnavailable as exc:
        raise _http_error(exc.status_code, str(exc), error_type="server_error", code="upstream_unavailable") from exc

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
            lang.t("err_upstream_http", status=resp.status_code, text=text),
            resp.status_code if resp.status_code < 500 else 502,
            error_type="invalid_request_error" if resp.status_code < 500 else "server_error",
            code="upstream_error",
        )

    body = await resp.aread()
    await resp.aclose()
    try:
        content = json.loads(body)
    except ValueError:
        return openai_error(
            lang.t("err_upstream_not_json", body=body[:500]),
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
async def v1_passthrough(path: str, request: Request, _: None = Depends(verify_proxy_key)) -> Response:
    """Catch-all passthrough: forward unimplemented /v1/* requests to the
    corresponding upstream path as-is.

    兜底透传：把未单独实现的 /v1/* 请求原样转发到上游对应路径。
    """
    if not path.strip("/"):
        return openai_error(lang.t("err_passthrough_path"), 404, code="not_found")

    session = _session_or_error()
    headers = session.to_headers()
    # The client-declared Content-Type must override the default from to_headers
    # 客户端声明的 Content-Type 必须覆盖 to_headers 的默认值
    content_type = request.headers.get("content-type", "")
    if content_type:
        headers["Content-Type"] = content_type
    body = await request.body()
    subpath = f"{path}?{request.url.query}" if request.url.query else path

    # Consistent with chat/embeddings: fall back to the other candidate prefixes
    # when the primary prefix returns 404 (route not found)
    #
    # 与 chat/embeddings 一致：主前缀 404（无此路由）时自动回退其它候选前缀
    try:
        resp = await upstream.forward(
            request.method, subpath, session, headers=headers, content=body or None, stream=True
        )
    except UpstreamUnavailable as exc:
        raise _http_error(
            exc.status_code, str(exc), error_type="server_error", code="upstream_unavailable"
        ) from exc

    if resp.status_code in AUTH_FAILURE_CODES:
        await resp.aclose()
        return _auth_failure_response(resp.status_code)

    media_type = resp.headers.get("content-type", "application/json")
    stream_headers = UpstreamClient.forward_headers(resp, {"X-Accel-Buffering": "no"})
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
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise _http_error(400, lang.t("err_invalid_json"), code="invalid_json")
    if not isinstance(payload, dict):
        raise _http_error(400, lang.t("err_json_object"), code="invalid_json")
    return payload


def _validate_chat_payload(payload: Dict[str, Any]) -> None:
    if not payload.get("model"):
        raise _http_error(400, lang.t("err_missing_model"), code="missing_required_field", param="model")
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
        async for chunk in resp.aiter_raw():
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
    return 0 if await _startup_check() else 1


def main(argv: Optional[List[str]] = None) -> int:
    # --help exits during parsing, before the post-parse reconfigure below, so peek
    # at argv first to honor --lang for the help text as well.
    #
    # --help 会在参数解析阶段直接退出，晚于下方的重配置；因此先扫一遍 argv，
    # 让 --help 的输出也能跟随 --lang。
    cli_args = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(cli_args):
        if token == "--lang" and i + 1 < len(cli_args):
            lang.configure(lang.resolve_language(cli_args[i + 1]))
            break

    parser = argparse.ArgumentParser(description=lang.t("cli_description"))
    parser.add_argument("--login", action="store_true", help=lang.t("cli_login_help"))
    parser.add_argument("--check", action="store_true", help=lang.t("cli_check_help"))
    parser.add_argument("--host", default=None, help=lang.t("cli_host_help"))
    parser.add_argument("--port", type=int, default=None, help=lang.t("cli_port_help"))
    parser.add_argument(
        "--lang",
        choices=(lang.LANG_ZH, lang.LANG_EN, lang.LANG_AUTO),
        default=lang.LANG_AUTO,
        help=lang.t("cli_lang_help"),
    )
    args = parser.parse_args(argv)

    # --host/--port/--lang require rebuilding settings (the dataclass is frozen)
    # --host/--port/--lang 需要重新构造 settings（dataclass 是 frozen 的）
    global settings, upstream
    if args.host or args.port or args.lang != lang.LANG_AUTO:
        import dataclasses

        settings = dataclasses.replace(
            settings,
            proxy_host=args.host or settings.proxy_host,
            proxy_port=args.port or settings.proxy_port,
            language=lang.resolve_language(args.lang),
        )
        upstream = UpstreamClient(settings)

    # Apply the effective language before any user-facing output (logs, banner, errors)
    # 在产生任何用户可见输出（日志、横幅、错误）之前应用生效语言
    lang.configure(settings.language)
    configure_logging(settings)

    if args.check:
        return asyncio.run(_check_session())

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
