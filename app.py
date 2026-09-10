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
import hashlib
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from secrets import compare_digest
from typing import Any, Dict, List, Optional, Set, Tuple

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
from model_probe import (
    EFFORT_ORDER,
    PROBE_SENTINEL,
    PROBED_PARAMETERS,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_UNPROBEABLE,
    ModelProbe,
    ModelProbeCache,
    baseline_payload,
    derive_reasoning_capability,
    effort_payload,
    extract_default_effort,
    extract_effort_candidates,
    looks_like_effort_error,
    parameter_of_error,
    parameter_payload,
    response_has_reasoning,
    vision_payload,
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

# Per-model probe cache (loaded lazily; persisted next to session.json)
# 逐模型探测缓存（惰性加载；持久化在 session.json 旁边）
model_probe = ModelProbeCache(settings.model_probe_cache_file)


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
        prefix, status = await upstream.probe_prefix(session)
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


# --------------------------------------------------------------------------- #
# Per-model probe: reasoning efforts, capabilities and request parameters
# 逐模型探测：思考挡位、能力与请求参数
# --------------------------------------------------------------------------- #
# Upstream statuses meaning "the engine rejected the request body", i.e. the probe
# learned something definite. Anything else (404, 5xx) is transient.
#
# 表示"引擎拒绝了请求体"的上游状态码，即探测学到了确定的东西。其它（404、5xx）
# 都是暂时性的。
VALIDATION_FAILURE_CODES = (400, 422)


class _ProbeAuthExpired(Exception):
    """
    Raised when the upstream rejects probe requests with 401/403: credentials
    died mid-probe, the whole refresh must stop instead of hammering a dead
    session once per model.

    上游对探测请求返回 401/403 时抛出：凭证在探测中途失效，整个刷新应立即
    停止，而不是对每个模型都拿着死凭证再撞一遍。
    """


class _ProbeTransient(RuntimeError):
    """
    The probe cannot reach a conclusion right now (network hiccup, 5xx, timeout):
    the caller records a failure with backoff and keeps the earlier facts.

    探测此刻得不出结论（网络抖动、5xx、超时）：调用方记录一次失败并退避，
    同时保留此前已确立的事实。
    """


async def _close(response: httpx.Response) -> None:
    """Close a response unless it is already closed. / 关闭响应（若尚未关闭）。"""
    if not response.is_closed:
        await response.aclose()


def _engine_build(body: str) -> str:
    """
    The engine build string a chat response carries (`system_fingerprint`), kept for
    diagnostics; "" when the body has none.

    聊天响应携带的引擎构建串（`system_fingerprint`），用于诊断；没有则为 ""。
    """
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("system_fingerprint") or "")


async def _probe_model(session: Any, model_id: str, fingerprint: str) -> ModelProbe:
    """
    Establish, by asking the engine, what one model accepts.

    Every step is a real request with max_tokens=1, so a full probe costs at most a
    handful of output tokens:

      1. candidate discovery -- a sentinel `reasoning_effort` makes the outer schema
         enumerate the levels it accepts;
      2. per-value verification -- only a 200 for a concrete level counts, which is
         what catches the second validation layer (Harmony, the model's own parser);
      3. request parameters -- one merged request, retried without whatever the 400
         blames; this also establishes function calling and structured outputs;
      4. vision -- one request carrying a 1x1 image;
      5. default behaviour -- one request with `reasoning_effort` omitted.

    Raises _ProbeAuthExpired (abort the whole refresh) or _ProbeTransient (retry the
    model later); every other outcome is a ModelProbe, complete or partial.

    通过"问引擎"确立单个模型接受什么。

    每一步都是 max_tokens=1 的真实请求，完整探测最多花费个位数输出 token：

      1. 候选发现 —— 用哨兵 `reasoning_effort` 让外层 schema 枚举它接受的挡位；
      2. 逐值实证 —— 只有具体挡位返回 200 才算数，这正是抓住第二层校验
         （Harmony、模型自带解析器）的关键；
      3. 请求参数 —— 一次合并请求，命中 400 就剔除被归因的参数后重试；这一步同时
         确立函数调用与结构化输出；
      4. 视觉 —— 一次携带 1x1 图片的请求；
      5. 默认行为 —— 一次省略 `reasoning_effort` 的请求。

    抛出 _ProbeAuthExpired（中止整轮刷新）或 _ProbeTransient（稍后重试该模型）；
    其它任何结果都返回 ModelProbe，可能是完整的也可能是部分的。
    """
    probe = ModelProbe(fingerprint=fingerprint, probed_at=time.time())
    unresolved = 0

    async def ask(payload: Dict[str, Any]) -> httpx.Response:
        """
        Send one probe request. An auth failure aborts the whole refresh; a transport
        failure is transient and never a claim about the model.

        发送一次探测请求。凭证失效中止整轮刷新；传输失败是暂时性的，绝不构成
        关于该模型的任何声明。
        """
        try:
            response = await upstream.post(session, "chat/completions", payload)
        except UpstreamUnavailable as exc:
            raise _ProbeTransient(str(exc)) from exc
        if response.status_code in AUTH_FAILURE_CODES:
            status_code = response.status_code
            await _close(response)
            raise _ProbeAuthExpired(str(status_code))
        return response

    # --- 1. candidate discovery: the sentinel makes the outer schema talk --------
    response = await ask(effort_payload(model_id, PROBE_SENTINEL))
    sentinel_status = response.status_code
    sentinel_body = response.text
    await _close(response)

    unprobeable = sentinel_status == 200
    if unprobeable:
        # The upstream ignored the sentinel: it does not validate the field, so a
        # per-value answer would not mean anything either.
        #
        # 上游忽略了哨兵值：它不校验该字段，逐值回答同样没有意义。
        probe.status = STATUS_UNPROBEABLE
        candidates: List[str] = []
    elif sentinel_status in VALIDATION_FAILURE_CODES:
        probe.default_effort = extract_default_effort(sentinel_body)
        # The enumeration is only the OUTER schema. When the phrasing is unknown,
        # sweep the whole canonical list rather than give up: verification is what
        # decides, so an unparsed candidate list costs requests, not correctness.
        #
        # 这个枚举只是**外层** schema。措辞不认识时改为遍历完整规范列表而不是放弃：
        # 结论由实证决定，因此候选解析不出只多花几次请求，不影响正确性。
        candidates = extract_effort_candidates(sentinel_body) or list(EFFORT_ORDER)
    else:
        raise _ProbeTransient(f"sentinel probe returned HTTP {sentinel_status}")

    # --- 2. per-value verification: only a 200 counts ---------------------------
    for effort in candidates:
        response = await ask(effort_payload(model_id, effort))
        status_code = response.status_code
        body = response.text
        await _close(response)
        if status_code == 200:
            probe.supported_efforts.append(effort)
            probe.system_fingerprint = probe.system_fingerprint or _engine_build(body)
        elif status_code in VALIDATION_FAILURE_CODES:
            # The model-level layer is the only place the engine sometimes names its
            # default level ("Supported types are xhigh (default), ..."), so mine it
            # here as well -- the outer schema error never carries that marker.
            #
            # 模型级那一层是引擎偶尔声明默认挡位的唯一地方（"Supported types are
            # xhigh (default), ..."），所以这里也要挖一遍——外层 schema 的报错从不带
            # 这个标注。
            if probe.default_effort is None:
                probe.default_effort = extract_default_effort(body)
            continue
        else:
            unresolved += 1
    probe.efforts_verified = not unprobeable and unresolved == 0

    # --- 3. request parameters: one merged request, retried without the offender --
    remaining = list(PROBED_PARAMETERS)
    parameter_accepted: Dict[str, bool] = {}
    for _ in range(len(PROBED_PARAMETERS) + 1):
        if not remaining:
            break
        response = await ask(parameter_payload(model_id, remaining))
        status_code = response.status_code
        body = response.text
        await _close(response)
        if status_code == 200:
            for parameter in remaining:
                parameter_accepted[parameter] = True
            probe.system_fingerprint = probe.system_fingerprint or _engine_build(body)
            remaining = []
            break
        if status_code not in VALIDATION_FAILURE_CODES:
            unresolved += 1
            break
        blamed = parameter_of_error(body)
        if blamed == "reasoning_effort":
            unresolved += 1
            break
        if blamed in ("tools", "tool_choice"):
            # `tools` and `tool_choice` are one feature: an engine built without a
            # tool-call parser rejects whichever of the two it sees first, so both
            # are disproved together.
            #
            # `tools` 与 `tool_choice` 属于同一特性：没带 tool-call parser 的引擎会拒绝
            # 先看到的那个，因此两者一起被证伪。
            blamed_set = {"tools", "tool_choice"}
        elif blamed:
            blamed_set = {blamed}
        else:
            # The 400 cannot be attributed to one parameter: claim nothing about the
            # ones still under test instead of guessing.
            #
            # 这个 400 无法归因到某个参数：对仍在测试的参数不做任何声明，而不是猜。
            unresolved += 1
            break
        for parameter in blamed_set & set(remaining):
            parameter_accepted[parameter] = False
        remaining = [item for item in remaining if item not in blamed_set]

    # --- 4. vision: does the engine accept image content at all? ----------------
    response = await ask(vision_payload(model_id))
    status_code = response.status_code
    body = response.text
    await _close(response)
    vision: Optional[bool]
    if status_code == 200:
        vision = True
        probe.system_fingerprint = probe.system_fingerprint or _engine_build(body)
    elif status_code in VALIDATION_FAILURE_CODES:
        vision = False
    else:
        vision = None
        unresolved += 1

    # --- 5. default behaviour: the same request without reasoning_effort --------
    response = await ask(baseline_payload(model_id))
    status_code = response.status_code
    body = response.text
    await _close(response)
    if status_code == 200:
        probe.default_enabled = response_has_reasoning(body)
        probe.system_fingerprint = probe.system_fingerprint or _engine_build(body)
    elif status_code not in VALIDATION_FAILURE_CODES:
        unresolved += 1

    # --- 6. assemble the facts --------------------------------------------------
    capabilities: Dict[str, bool] = {}
    if vision is not None:
        capabilities["vision"] = vision
    if "tools" in parameter_accepted:
        capabilities["function_calling"] = parameter_accepted["tools"]
    if "response_format" in parameter_accepted:
        capabilities["structured_outputs"] = parameter_accepted["response_format"]
    reasoning_capable = derive_reasoning_capability(
        None if unprobeable else probe.supported_efforts, probe.default_enabled
    )
    if reasoning_capable is not None:
        capabilities["reasoning"] = reasoning_capable
    probe.capabilities = capabilities

    accepted_parameters = [
        parameter for parameter, accepted in parameter_accepted.items() if accepted
    ]
    if probe.supported_efforts and not unprobeable:
        accepted_parameters.append("reasoning_effort")
    probe.supported_parameters = sorted(accepted_parameters)

    if unprobeable:
        probe.status = STATUS_UNPROBEABLE
    elif unresolved:
        probe.status = STATUS_PARTIAL
        probe.last_error = f"{unresolved} probe request(s) left the answer open"
    else:
        probe.status = STATUS_OK
    return probe


def _raw_model_capabilities(raw: Any) -> Dict[str, bool]:
    """
    The capability dictionary one upstream model object carries, boolean entries only.

    单个上游模型对象携带的能力字典，只取布尔项。
    """
    if not isinstance(raw, dict):
        return {}
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
    capabilities = meta.get("capabilities")
    if not isinstance(capabilities, dict):
        return {}
    return {
        str(key): value for key, value in capabilities.items() if isinstance(value, bool)
    }


def _shared_default_capabilities(raw_models: List[Any]) -> Optional[Dict[str, bool]]:
    """
    The capability keys every reporting upstream model agrees on -- that shared part
    is Open WebUI's "default model metadata" template, merged into each model.

    Keys the models disagree about (or that only some of them report) are left out;
    a model's own value for those is published as a deviation under that model's
    `x_open_webui`. Reporting the template once, as an instance-level fact, is
    honest; repeating it inside each model's `capabilities` would claim something
    about the model that is not true -- the same template was also handed to
    DeepSeek-V4-Flash, which then answered an image with "is not a multimodal model".

    上游每个上报能力的模型都一致同意的那些键——这部分共同值就是 Open WebUI 合并进
    每个模型的"默认模型元数据"模板。

    各模型不一致（或只有部分模型上报）的键不纳入模板；某个模型对这些键自己的取值，
    作为"偏离"放在该模型的 `x_open_webui` 里。把模板作为实例级事实输出一次是诚实的；
    重复放进每个模型的 `capabilities` 则是在声称模型具备它并不具备的能力——同一份
    模板也发给了 DeepSeek-V4-Flash，而它对图片的回答是 "is not a multimodal model"。
    """
    reported = [
        capabilities
        for capabilities in (_raw_model_capabilities(raw) for raw in raw_models)
        if capabilities
    ]
    if not reported:
        return None
    template: Dict[str, bool] = {}
    for key in reported[0]:
        values = {capabilities.get(key) for capabilities in reported}
        if len(values) == 1 and None not in values:
            template[key] = reported[0][key]
    return template or None


def _model_fingerprint(raw: Any, model_id: str) -> str:
    """
    A cheap identity for "the engine serving this model", derived purely from the
    model list so checking it costs no request.

    Deliberately excludes the top-level `created`: vLLM rebuilds its model card for
    every response and stamps it with the current time, so it changes on every fetch
    (verified: 1789036467 then 1789036470 three seconds later).

    仅从模型列表推导出的"服务该模型的引擎"廉价标识，检查它不需要任何请求。

    刻意排除顶层 `created`：vLLM 每次响应都会重建模型卡并打上当前时间，因此它每次
    拉取都会变（实测：1789036467，三秒后 1789036470）。
    """
    if not isinstance(raw, dict):
        return ""
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    engine = raw.get("openai") if isinstance(raw.get("openai"), dict) else {}
    identity = {
        "id": model_id,
        "root": engine.get("root") or raw.get("root") or "",
        "max_model_len": raw.get("max_model_len") or engine.get("max_model_len"),
        "owned_by": engine.get("owned_by") or raw.get("owned_by") or "",
        "base_model_id": info.get("base_model_id"),
        "updated_at": info.get("updated_at"),
    }
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


async def _fetch_raw_models(session: Any) -> List[Any]:
    """
    GET the upstream model list and return its raw entries.

    Every failure mode raises HttpError with an OpenAI-style body, so /v1/models and
    /v1/models/{id} share one code path.

    GET 上游模型列表并返回原始条目。

    所有失败模式都以 HttpError（OpenAI 风格错误体）抛出，使 /v1/models 与
    /v1/models/{id} 共用同一条代码路径。
    """
    try:
        resp = await upstream.get_models(session)
    except UpstreamUnavailable as exc:
        raise _http_error(
            exc.status_code, str(exc), error_type="server_error", code="upstream_unavailable"
        ) from exc

    try:
        if resp.status_code in AUTH_FAILURE_CODES:
            logger.error(lang.t("auth_failure_log", status=resp.status_code))
            raise _http_error(
                resp.status_code,
                lang.t("err_upstream_unauthorized"),
                error_type="invalid_request_error",
                code="upstream_unauthorized",
            )
        if resp.status_code != 200:
            raise _http_error(
                502,
                lang.t("err_upstream_models_http", status=resp.status_code, text=resp.text[:500]),
                error_type="server_error",
                code="upstream_error",
            )
        try:
            payload = resp.json()
        except ValueError:
            raise _http_error(
                502,
                lang.t("err_upstream_models_not_json", text=resp.text[:500]),
                error_type="server_error",
                code="upstream_error",
            )
    finally:
        await _close(resp)

    return extract_model_list(payload)


async def _fetch_model_summaries(session: Any) -> Optional[List[Tuple[str, str]]]:
    """
    Fetch the upstream model list and reduce it to (model_id, engine fingerprint)
    pairs; None means the list could not be retrieved (the caller skips this round).

    拉取上游模型列表并归约为 (模型 id, 引擎指纹) 对；None 表示拉取失败
    （调用方跳过本轮刷新）。
    """
    try:
        raw_models = await _fetch_raw_models(session)
    except HttpError as exc:
        logger.warning(lang.t("probe_models_failed", status=exc.status_code))
        return None

    summaries: List[Tuple[str, str]] = []
    for raw in raw_models:
        model = normalize_model(raw)
        if not model:
            continue
        summaries.append((model["id"], _model_fingerprint(raw, model["id"])))
    return summaries


# --------------------------------------------------------------------------- #
# Instance-level Open WebUI metadata (served as the envelope's "x_open_webui")
# 实例级 Open WebUI 元信息（作为信封的 "x_open_webui" 输出）
# --------------------------------------------------------------------------- #
# How long a /api/config snapshot is reused before being fetched again.
# /api/config 快照复用的时长，超过后才重新拉取。
INSTANCE_META_TTL = 300.0


@dataclass
class _InstanceMeta:
    """
    Facts about the Open WebUI deployment itself, as opposed to any single model:
    the feature switches it has turned on, and the default model metadata template
    it merges into every model.

    关于 Open WebUI 部署本身（而非任何单个模型）的事实：它开启了哪些功能开关，
    以及它合并进每个模型的默认模型元数据模板。
    """

    name: str = ""
    version: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    default_model_capabilities: Optional[Dict[str, bool]] = None
    fetched_at: float = 0.0

    def is_usable(self) -> bool:
        """Whether anything worth publishing is known. / 是否有值得输出的内容。"""
        return bool(self.name or self.version or self.features or self.default_model_capabilities)

    def is_fresh(self, now: float) -> bool:
        return bool(self.fetched_at) and (now - self.fetched_at) < INSTANCE_META_TTL

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.name:
            payload["name"] = self.name
        if self.version:
            payload["version"] = self.version
        if self.features:
            payload["features"] = self.features
        if self.default_model_capabilities:
            payload["default_model_capabilities"] = self.default_model_capabilities
        return payload


_instance_meta = _InstanceMeta()


async def _ensure_instance_meta(session: Any) -> None:
    """
    Refresh the /api/config snapshot when it is stale. Best effort: a failure keeps
    the previous snapshot for another TTL and never fails /v1/models.

    快照过期时刷新 /api/config。尽力而为：失败则保留上一份快照再等一个 TTL，
    绝不让 /v1/models 失败。
    """
    if _instance_meta.is_fresh(time.time()):
        return
    config = await upstream.get_instance_config(session)
    _instance_meta.fetched_at = time.time()
    if not isinstance(config, dict):
        return
    _instance_meta.name = str(config.get("name") or _instance_meta.name)
    _instance_meta.version = str(config.get("version") or _instance_meta.version)
    features = config.get("features")
    if isinstance(features, dict):
        _instance_meta.features = features


@dataclass
class _RefreshState:
    """
    What a running refresh is probing right now, so /v1/models can decide whether
    waiting for it could actually change the answer.

    正在运行的刷新此刻在探测什么，供 /v1/models 判断"等它"是否真能改变结果。
    """

    pending: Set[str] = field(default_factory=set)


_refresh_state = _RefreshState()


def _describe_probe(probe: ModelProbe) -> str:
    """
    A one-line, log-friendly summary of what a probe established.

    对一次探测所确立内容的单行、便于记日志的摘要。
    """
    if probe.status == STATUS_UNPROBEABLE:
        efforts = "upstream does not validate the field"
    else:
        efforts = ", ".join(probe.supported_efforts) or "none accepted"
    parts = [f"efforts[{efforts}]"]
    if probe.default_effort:
        parts.append(f"default={probe.default_effort}")
    if probe.default_enabled is not None:
        parts.append(f"thinking_by_default={str(probe.default_enabled).lower()}")
    if probe.capabilities:
        parts.append(
            "capabilities="
            + ",".join(f"{key}:{str(value).lower()}" for key, value in probe.capabilities.items())
        )
    if probe.status == STATUS_PARTIAL:
        parts.append(f"partial({probe.last_error})")
    return "; ".join(parts)


async def _refresh_model_probe(
    *,
    summaries: Optional[List[Tuple[str, str]]] = None,
    force: bool = False,
) -> bool:
    """
    Reconcile the probe cache with the current model list, probe whatever is missing
    or stale, persist, and report success.

    `summaries` may be supplied by a caller that has just fetched the model list (the
    /v1/models path), which saves one upstream request.

    With force=False this is a no-op when every current model already has a conclusive
    entry for its engine fingerprint -- the "unchanged engine -> serve cache" contract.

    将探测缓存与当前模型列表对齐，探测缺失或过期的模型，持久化，并汇报结果。

    `summaries` 可由刚刚拉过模型列表的调用方传入（/v1/models 路径），省一次上游请求。

    force=False 时，若每个当前模型都已针对其引擎指纹有结论性条目，则什么都不做——
    即"引擎未变 -> 直接用缓存"的约定。
    """
    try:
        session = load_session(settings)
    except SessionError as exc:
        logger.warning("%s", exc)
        return False

    if summaries is None:
        summaries = await _fetch_model_summaries(session)
        if summaries is None:
            return False

    model_probe.load()
    to_probe = model_probe.sync_with_models(summaries, force=force)
    if not to_probe:
        logger.info(lang.t("probe_cache_fresh", count=len(model_probe)))
        return True

    fingerprints = dict(summaries)
    _refresh_state.pending = set(to_probe)
    logger.info(lang.t("probe_begin", count=len(to_probe), models=", ".join(to_probe)))
    semaphore = asyncio.Semaphore(settings.model_probe_concurrency)
    counters = {"ok": 0, "partial": 0, "unprobeable": 0, "failed": 0}

    async def worker(model_id: str) -> None:
        async with semaphore:
            try:
                probe = await asyncio.wait_for(
                    _probe_model(session, model_id, fingerprints[model_id]),
                    timeout=settings.model_probe_timeout,
                )
            except _ProbeAuthExpired:
                raise
            except Exception as exc:  # noqa: BLE001 - per-model isolation
                model_probe.record_failure(model_id, fingerprints[model_id], str(exc))
                counters["failed"] += 1
                logger.warning(lang.t("probe_model_failed", model=model_id, exc=exc))
                return
            model_probe.record_result(model_id, probe)
            counters[probe.status] = counters.get(probe.status, 0) + 1
            logger.info(lang.t("probe_model_done", model=model_id, summary=_describe_probe(probe)))

    try:
        await asyncio.gather(*(worker(model_id) for model_id in to_probe))
    except _ProbeAuthExpired as exc:
        logger.error(lang.t("probe_auth_expired", status=exc))
        # Still save whatever was collected before the credentials died
        # 凭证失效前收集到的结果仍然落盘
        model_probe.save()
        return False
    finally:
        _refresh_state.pending.clear()

    model_probe.save()
    logger.info(
        lang.t(
            "probe_finished",
            ok=counters["ok"],
            partial=counters["partial"],
            unprobeable=counters["unprobeable"],
            failed=counters["failed"],
        )
    )
    logger.info(
        lang.t("probe_cache_saved", path=settings.model_probe_cache_file, count=len(model_probe))
    )
    return True


_refresh_task: Optional[asyncio.Task] = None


def _spawn_model_probe_refresh(
    *, summaries: Optional[List[Tuple[str, str]]] = None
) -> Optional[asyncio.Task]:
    """
    Launch a cache refresh in the background, at most one at a time, and return the
    running task (freshly started or already in flight) so callers may wait for it.

    后台启动一次缓存刷新，同一时刻至多一个实例，并返回运行中的任务
    （新启动的或已在跑的），调用方可选择性地等待它。
    """
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        return _refresh_task

    async def runner() -> None:
        try:
            await _refresh_model_probe(summaries=summaries)
        except Exception as exc:  # noqa: BLE001 - background task must not die silently
            logger.warning(lang.t("probe_task_error", exc=exc))

    _refresh_task = asyncio.create_task(runner())
    return _refresh_task


# Models that already have a heal probe scheduled, so a burst of bad requests cannot
# stampede the upstream with one refresh each.
#
# 已安排自愈探测的模型，避免一串坏请求各自触发一次刷新、把上游打爆。
_healing: Set[str] = set()


def _trigger_probe_heal(model_id: Optional[str], effort: Optional[str]) -> None:
    """
    React to a live upstream 400 about the reasoning effort: drop the disproved level
    from the cache immediately and re-probe that model in the background.

    The request that discovered the problem is never delayed by this.

    对线上"关于思考挡位"的上游 400 作出反应：立即从缓存里剔除被证伪的挡位，并在
    后台重探该模型。

    发现问题的那个请求绝不因此被拖延。
    """
    if not model_id or model_id in _healing:
        return
    entry = model_probe.entry(model_id)
    if entry is not None and not model_probe.invalidate_effort(model_id, effort):
        # Nothing was disproved (the level was not advertised): no re-probe needed.
        # 没有被证伪的东西（该挡位本就没被声明）：无需重探。
        return
    _healing.add(model_id)

    async def healer() -> None:
        try:
            model_probe.save()
            await _refresh_model_probe()
        except Exception as exc:  # noqa: BLE001 - background task must not die silently
            logger.warning(lang.t("probe_task_error", exc=exc))
        finally:
            _healing.discard(model_id)

    asyncio.create_task(healer())


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
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
    # Background per-model probe: covers the "first login / empty cache" case as well
    # as plain startups; never blocks the service from serving.
    #
    # 后台逐模型探测：既覆盖"首次登录 / 缓存为空"，也覆盖普通启动；
    # 绝不阻塞服务对外提供服务。
    _spawn_model_probe_refresh()
    try:
        yield
    finally:
        if _refresh_task is not None and not _refresh_task.done():
            _refresh_task.cancel()
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


def is_proxy_key_valid(request: Request) -> bool:
    """
    Whether the request carries a valid proxy key; always True when auth is
    disabled (PROXY_API_KEY empty).

    Kept as a separate layer from require_proxy_key so the meta endpoints (/ and
    /healthz) stay accessible without auth, and only use this result to decide
    whether to expose sensitive fields such as the upstream address.

    请求是否携带有效代理 Key；未启用鉴权（PROXY_API_KEY 为空）时恒为 True。

    与 require_proxy_key 分成两层：元信息端点（/ 与 /healthz）保持免鉴权
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
    """
    An error that the exception handler turns into an OpenAI-style error body.

    会被异常处理器转换成 OpenAI 风格错误体的异常。
    """

    def __init__(self, status_code: int, message: str, **error_fields: Any):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_fields = error_fields


def _http_error(status_code: int, message: str, **error_fields: Any) -> HttpError:
    return HttpError(status_code, message, **error_fields)


@app.exception_handler(HttpError)
async def http_error_handler(request: Request, exc: HttpError) -> JSONResponse:
    return openai_error(exc.message, exc.status_code, **exc.error_fields)


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


def normalize_model(
    raw: Any, shared_capabilities: Optional[Dict[str, bool]] = None
) -> Optional[Dict[str, Any]]:
    """Collapse an upstream model object into the OpenAI model structure.

    Standard fields stay intact; a whitelist of safe, useful extras is preserved
    when present: name, description, max_model_len (kept for compatibility) plus
    max_context_length and context_length, and quantization (parsed from the model
    id). Private upstream fields (user_id, access_grants, permission, urlIdx, ...)
    are never exposed.

    `capabilities`, `architecture`, `supported_parameters` and `reasoning` are NOT
    built here: they are established by probing the engine and attached by the
    caller, because the upstream's own capability dictionary is a deployment-wide
    default template rather than a fact about the model.

    把上游的模型对象收敛成 OpenAI 的 model 结构。

    标准字段原样保留，另有一份白名单在存在时透出安全且有用的扩展字段：
    name、description、max_model_len（兼容保留）+ max_context_length/
    context_length、quantization（从模型名解析）；上游私有字段（user_id、
    access_grants、permission、urlIdx 等）一律不透出。

    `capabilities`、`architecture`、`supported_parameters`、`reasoning` **不在这里
    构造**：它们由探测引擎得出、并由调用方附加，因为上游自带的能力字典是部署级的
    默认模板，而不是关于该模型的事实。
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

    # Standard fields first, then the whitelisted extras: only emitted when the
    # upstream provides them, so minimal/legacy model objects keep the exact
    # 4-field OpenAI shape.
    #
    # 先标准字段，后白名单扩展：上游提供时才输出，极简/老版本模型对象仍保持
    # 精确的 4 字段 OpenAI 结构。
    model: Dict[str, Any] = {
        "id": str(model_id),
        "object": "model",
        "created": created,
        "owned_by": str(owned_by),
    }

    # Human-readable name. Upstream keeps it separate from the id (workspace models
    # use a uuid as id and a friendly name here), and every mainstream provider that
    # publishes a list of models publishes one too.
    #
    # 人类可读的名称。上游把它与 id 分开保存（workspace 模型用 uuid 作 id，友好名放在
    # 这里），而所有会输出模型列表的主流供应商也都会输出这个字段。
    name = raw.get("name") or info.get("name")
    if name:
        model["name"] = str(name)

    # Generic-template field names; max_model_len stays as a compatibility alias
    # 通用模板字段名；max_model_len 作为兼容别名保留
    max_model_len = raw.get("max_model_len") or openai_obj.get("max_model_len")
    try:
        context_length = int(max_model_len)
    except (TypeError, ValueError):
        context_length = None
    if context_length is not None:
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

    own_capabilities = _raw_model_capabilities(raw)
    template = shared_capabilities or {}
    deviation = {
        key: value for key, value in own_capabilities.items() if template.get(key) != value
    }
    if deviation:
        # Open WebUI hands the deployment-wide template to every model, so a model
        # that deviates from it is worth keeping -- but under the instance namespace,
        # never inside `capabilities`, which holds probed facts only.
        #
        # Open WebUI 把同一份部署级模板发给每个模型，因此偏离模板的模型值得保留——
        # 但放在实例命名空间下，绝不放进只承载实证事实的 `capabilities`。
        model["x_open_webui"] = {"capabilities": deviation}

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

    健康检查。保持免鉴权 200（探针/负载均衡友好），上游地址同样只在
    请求通过 Key 校验时返回。
    """
    response_body: Dict[str, Any] = {
        "status": "ok",
        "version": VERSION,
        "session_ready": session_exists(settings),
        "auth_required": bool(settings.proxy_api_key),
    }
    if is_proxy_key_valid(request):
        response_body["upstream"] = settings.open_webui_base_url
        response_body["upstream_prefix"] = upstream.prefix
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
    避免第二次拉取模型列表。

    同时刷新实例级的默认能力模板，它是随模型列表免费得到的。
    """
    shared_capabilities = _shared_default_capabilities(raw_models)
    if shared_capabilities:
        _instance_meta.default_model_capabilities = shared_capabilities

    models: List[Dict[str, Any]] = []
    summaries: List[Tuple[str, str]] = []
    for raw in raw_models:
        model = normalize_model(raw, shared_capabilities)
        if not model:
            continue
        models.append(model)
        summaries.append((model["id"], _model_fingerprint(raw, model["id"])))

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
    raw_models = await _fetch_raw_models(session)
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
            # Let the refresh publish what it is probing before deciding to wait.
            # 先让刷新任务公布它在探测什么，再决定是否等待。
            await asyncio.sleep(0)
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
    raw_models = await _fetch_raw_models(session)
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

    is_stream = bool(payload.get("stream"))
    logger.debug(lang.t("forward_chat", model=payload.get("model"), stream=is_stream))

    try:
        resp = await upstream.post(session, "chat/completions", payload, stream=is_stream)
    except UpstreamUnavailable as exc:
        raise _http_error(exc.status_code, str(exc), error_type="server_error", code="upstream_unavailable") from exc

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
            lang.t("err_upstream_http", status=resp.status_code, text=text),
            resp.status_code if resp.status_code < 500 else 502,
            error_type="invalid_request_error" if resp.status_code < 500 else "server_error",
            code="upstream_error",
        )

    if not is_stream:
        raw_body = await resp.aread()
        await resp.aclose()
        try:
            return JSONResponse(content=json.loads(raw_body))
        except ValueError:
            return openai_error(
                lang.t("err_upstream_not_json", body=raw_body[:500]),
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
async def embeddings(request: Request, _: None = Depends(require_proxy_key)) -> Response:
    session = _session_or_error()
    payload = await _read_json_body(request)
    if not payload.get("model") or "input" not in payload:
        return openai_error(
            lang.t("err_missing_model_input"), 400, code="missing_required_field"
        )
    payload["model"] = settings.resolve_model(payload.get("model"))

    try:
        resp = await upstream.post(session, "embeddings", payload, stream=False)
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

    raw_body = await resp.aread()
    await resp.aclose()
    try:
        content = json.loads(raw_body)
    except ValueError:
        return openai_error(
            lang.t("err_upstream_not_json", body=raw_body[:500]),
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
    for index, token in enumerate(cli_args):
        if token == "--lang" and index + 1 < len(cli_args):
            lang.configure(lang.resolve_language(cli_args[index + 1]))
            break

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
        return 0 if asyncio.run(_refresh_model_probe(force=True)) else 1

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
