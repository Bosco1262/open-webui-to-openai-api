"""
Upstream data access and per-model probe orchestration.

This module owns the process-level runtime singletons (the shared UpstreamClient
and the ModelProbeCache) plus everything that orchestrates upstream data:

* the model-list fetch path (with the 404 prefix self-heal and the short TTL),
* the per-model probe (`_probe_model`) and its serialized refresh loop,
* the live-400 heal path,
* the instance-level `/api/config` snapshot.

`app.py` keeps the FastAPI surface (routes, CLI, model normalization) and imports
the runtime singletons from here, so the dependency graph stays one-way:
app -> probe_runner -> config / session_store / model_probe / upstream.

The singletons live here rather than in app.py so that tests can patch
`probe_runner.upstream` / `probe_runner.model_probe` and have the orchestration
functions actually see the replacement.


上游数据访问与逐模型探测编排。

本模块持有进程级运行时单例（共享的 UpstreamClient 与 ModelProbeCache），以及
编排上游数据的全部逻辑：

* 模型列表拉取路径（含 404 前缀自愈与短 TTL 缓存）；
* 逐模型探测（`_probe_model`）与其串行化刷新循环；
* 线上 400 的自愈路径；
* 实例级 `/api/config` 快照。

`app.py` 保留 FastAPI 表面（路由、CLI、模型规范化），并从这里导入运行时单例，
使依赖图保持单向：app -> probe_runner -> config / session_store / model_probe /
upstream。

单例放在这里而非 app.py，是为了让测试可以 patch `probe_runner.upstream` /
`probe_runner.model_probe` 并真正被编排函数看到。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

import lang
from config import Settings, settings
from model_probe import (
    EFFORT_ORDER,
    PROBE_SENTINEL,
    PROBED_PARAMETERS,
    STATUS_FAILED,
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
    parameter_of_error,
    parameter_payload,
    response_has_reasoning,
    vision_payload,
)
from models import _model_summaries
from request_context import bind_request_id, request_id_suffix, sanitize_log_text
from session_store import SessionError, load_session, session_exists
from upstream import (
    AUTH_FAILURE_CODES,
    UpstreamClient,
    UpstreamRequestInvalid,
    UpstreamUnavailable,
)

logger = logging.getLogger("webui-proxy.runner")

# Probe traffic identifies itself (D11). It shares the connection pool and the
# credentials with real chat traffic, so without a marker an operator reading upstream
# logs -- or a rate limiter deciding what to throttle -- cannot tell a probe from a user.
#
# 探测流量自报身份（D11）。它与真实对话流量共用连接池与凭证，因此若无标记，运维在上游
# 日志里（或限流方在决定限谁时）无法把探测与用户请求区分开。
PROBE_REQUEST_HEADERS = {"X-WebUI-Proxy-Probe": "1"}

# --------------------------------------------------------------------------- #
# Runtime singletons / 运行时单例
# --------------------------------------------------------------------------- #
upstream = UpstreamClient(settings)

# Per-model probe cache (loaded lazily; persisted next to session.json)
# 逐模型探测缓存（惰性加载；持久化在 session.json 旁边）
model_probe = ModelProbeCache(settings.model_probe_cache_file)


def use_settings(new_settings: Settings) -> UpstreamClient:
    """
    Adopt `new_settings` as this module's runtime settings, and return the fresh upstream
    client that goes with it (I9).

    `config.settings` is the value loaded at import time -- the startup default, not the
    runtime instance. When `app.main` rebuilds the frozen dataclass for --host/--port/
    --lang, both holders of the live object (the app module and this one) have to change
    together, and doing both halves in one place is what makes "adopt it here, forget it
    there" impossible: the module that owns the orchestration state is also the one that
    knows it needs a matching client.

    Logging level and language are applied by the caller, not here.


    把 `new_settings` 作为本模块的运行时设置，并返回与之配套的新上游客户端（I9）。

    `config.settings` 是 import 期加载的值——启动默认值，而非运行时实例。`app.main` 为
    --host/--port/--lang 重建这个 frozen dataclass 时，两个运行时持有者（app 模块与本模块）
    必须一起换；把两半放在同一处，才使"这边采纳了、那边忘了"成为不可能：持有编排状态的
    模块本来就知道自己需要一个与之匹配的客户端。

    日志级别与语言由调用方应用，不在此处。
    """
    global settings, upstream
    settings = new_settings
    upstream = UpstreamClient(new_settings)
    return upstream


# --------------------------------------------------------------------------- #
# Errors (OpenAI-style, raised by the upstream data path)
# 错误（上游数据路径抛出的 OpenAI 风格错误）
# --------------------------------------------------------------------------- #
class HttpError(Exception):
    """
    An error that the exception handler turns into an OpenAI-style error body.

    会被异常处理器转换成 OpenAI 风格错误体的异常。
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Dict[str, str]] = None,
        **error_fields: Any,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.headers = headers
        self.error_fields = error_fields


def _http_error(status_code: int, message: str, **error_fields: Any) -> HttpError:
    return HttpError(status_code, message, **error_fields)


# --------------------------------------------------------------------------- #
# Probe health (U-11)
# 探测健康状态（U-11）
# --------------------------------------------------------------------------- #
# Probe-health states. `unknown` until something is observed; `ok`/`degraded` describe
# the upstream as a whole, `auth_rejected`/`no_session` name a credential problem the
# operator has to fix.
#
#
# 探测健康状态。观察到任何事实前为 `unknown`；`ok`/`degraded` 描述上游整体状况，
# `auth_rejected`/`no_session` 则点名了必须由运维处理的凭证问题。
PROBE_HEALTH_UNKNOWN = "unknown"
PROBE_HEALTH_OK = "ok"
PROBE_HEALTH_DEGRADED = "degraded"
PROBE_HEALTH_AUTH_REJECTED = "auth_rejected"
PROBE_HEALTH_NO_SESSION = "no_session"

# Consecutive credential rejections before the single prominent warning is emitted. A
# dead session used to produce one line per round (or none at all, when the failure was
# only visible as "N models failed"); one warning that says what to do is actionable,
# the hundredth is noise.
#
# 连续多少次凭证被拒后发出那一条醒目告警。会话失效过去会每轮一行（或者干脆没有，
# 只在"N 个模型失败"里间接可见）；一条写明该怎么办的告警是有用的，第一百条只是噪声。
PROBE_AUTH_FAILURE_WARN = 3


@dataclass
class ProbeHealth:
    """
    What the probe subsystem last observed, ready to be exposed on /healthz and in
    `--check` output (U-11).

    The whole point is that "the session died" is otherwise invisible until someone
    reads the logs: the refresh loop retries quietly, every round logs a failed model
    or two, and nothing says "log in again".


    探测子系统最近观察到的情况，供 /healthz 与 --check 输出（U-11）。

    它存在的全部意义在于："会话死了"这件事过去只有翻日志的人才知道：刷新循环安静地
    重试，每轮记下几个失败的模型，却没有一处告诉你"请重新登录"。
    """

    status: str = PROBE_HEALTH_UNKNOWN
    consecutive_auth_failures: int = 0
    last_error: str = ""
    updated_at: float = 0.0
    # Counters of the last completed round: probed / ok / partial / unprobeable / failed.
    # 最近一轮完成的计数：probed / ok / partial / unprobeable / failed。
    last_round: Optional[Dict[str, Any]] = None
    # Whether the prominent warning was already emitted for the current streak, so the
    # failure is announced once instead of once per round.
    #
    # 当前这串连续失败是否已经发过那条醒目告警，使失败只被宣布一次而不是每轮一次。
    warned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """The /healthz view of this state. / 本状态在 /healthz 上的呈现。"""
        payload: Dict[str, Any] = {
            "status": self.status,
            "consecutive_auth_failures": self.consecutive_auth_failures,
        }
        if self.last_error:
            payload["last_error"] = self.last_error
        if self.last_round:
            payload["last_round"] = dict(self.last_round)
        if self.updated_at:
            payload["updated_at"] = int(self.updated_at)
        return payload

    def _announce_recovery(self) -> None:
        """
        Emit the one "credentials work again" line, so an operator who saw the warning
        does not have to guess whether it is still true.

        输出那唯一一行"凭证恢复可用"，使看到过告警的运维不必猜它是否仍然成立。
        """
        if self.warned:
            logger.info(lang.t("probe_health_recovered"))
        self.warned = False

    def record_auth_rejected(self, reason: str) -> None:
        """
        The upstream refused the credentials. Counts the streak and warns once at the
        threshold with the action to take.

        上游拒绝了凭证。累计连续次数，并在达到阈值时发出唯一一条带行动指引的告警。
        """
        self.status = PROBE_HEALTH_AUTH_REJECTED
        self.consecutive_auth_failures += 1
        self.last_error = reason[:300]
        self.updated_at = time.time()
        if self.consecutive_auth_failures >= PROBE_AUTH_FAILURE_WARN and not self.warned:
            self.warned = True
            logger.warning(
                lang.t(
                    "probe_auth_rejected_warning",
                    count=self.consecutive_auth_failures,
                )
            )

    def record_no_session(self, reason: str) -> None:
        """
        There is no usable credential file at all.

        根本没有可用的凭证文件。
        """
        self.status = PROBE_HEALTH_NO_SESSION
        self.last_error = reason[:300]
        self.updated_at = time.time()

    def record_degraded(self, reason: str) -> None:
        """
        A transient failure (the upstream could not be reached, the model list could
        not be fetched). Never overwrites a known credential problem: that one is more
        specific and needs operator action.

        暂时性失败（连不上上游、拉不到模型列表）。绝不覆盖已知的凭证问题：后者更具体，
        而且需要运维动手。
        """
        if self.status in (PROBE_HEALTH_AUTH_REJECTED, PROBE_HEALTH_NO_SESSION):
            return
        self.status = PROBE_HEALTH_DEGRADED
        self.last_error = reason[:300]
        self.updated_at = time.time()

    def record_credentials_ok(self) -> None:
        """
        A credential validation passed (startup self-check, --check). Clears the
        failure streak; a "degraded" state left over from a probe round is kept, since
        it is the newer information about the upstream as a whole.

        一次凭证校验通过（启动自检、--check）。清零连续失败计数；探测轮次留下的
        "degraded" 会保留——它是关于上游整体状况更新的信息。
        """
        recovered = self.status in (
            PROBE_HEALTH_AUTH_REJECTED,
            PROBE_HEALTH_NO_SESSION,
        )
        self.consecutive_auth_failures = 0
        self.last_error = ""
        self.updated_at = time.time()
        if recovered or self.status == PROBE_HEALTH_UNKNOWN:
            self.status = PROBE_HEALTH_OK
        if recovered:
            self._announce_recovery()

    def record_round(self, counts: Dict[str, int]) -> None:
        """
        A full probe round finished. This is the authoritative "the credentials work"
        signal: the round only completes when the upstream answered.

        一整轮探测完成。这是"凭证可用"的权威信号：只有上游应答了，这一轮才会走完。
        """
        recovered = self.status == PROBE_HEALTH_AUTH_REJECTED
        self.status = PROBE_HEALTH_OK
        self.consecutive_auth_failures = 0
        self.last_error = ""
        self.warned = False
        self.updated_at = time.time()
        self.last_round = {"at": int(self.updated_at), **counts}
        if recovered:
            self._announce_recovery()


probe_health = ProbeHealth()


def probe_cache_status() -> Dict[str, Any]:
    """
    A queryable summary of the persisted probe cache (U-11): how many models are known
    and how many of them the last rounds left inconclusive, plus the most recent
    failure reason. Read from disk, so a fresh `--check` process can report it.

    磁盘上探测缓存的可查询摘要（U-11）：已知多少个模型、其中多少个被最近几轮留成了
    未定性，以及最近一次失败原因。从磁盘读取，因此新起的 `--check` 进程也能报告它。
    """
    model_probe.load()
    inconclusive = model_probe.inconclusive()
    detail = ""
    if inconclusive:
        latest = max(inconclusive, key=lambda entry: entry.probed_at or entry.retry_after)
        detail = lang.t(
            "check_probe_cache_error",
            error=(latest.last_error or latest.status)[:200],
        )
    return {
        "models": len(model_probe),
        "inconclusive": len(inconclusive),
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# Lifecycle
# 生命周期
# --------------------------------------------------------------------------- #
async def _startup_check(*, quiet_success: bool = False) -> bool:
    """Validate credentials and upstream connectivity at startup.

    `quiet_success=True` (the --check path) skips the success line: the caller
    prints its own single summary instead of two overlapping messages (R11).


    启动时校验凭证与上游连通性。返回 True 表示凭证可用。

    `quiet_success=True`（--check 路径）跳过成功日志：由调用方输出一份
    合并摘要，而不是两条重叠的消息（R11）。
    """
    if not settings.proxy_api_key and not settings.proxy_api_keys:
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
        probe_health.record_no_session(str(exc))
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
        # U-11: an unreachable upstream is a degraded state the operator should be able
        # to query, not just a startup line that scrolls away.
        #
        # U-11：上游连不上属于可被运维查询到的降级状态，而不只是一行很快滚走的启动日志。
        probe_health.record_degraded(str(exc))
        return False

    if status in AUTH_FAILURE_CODES:
        logger.error(
            lang.t("creds_expired", status=status)
        )
        probe_health.record_auth_rejected(f"HTTP {status}")
        return False

    if status == 404:
        # The old behavior misreported a 404 as "credentials valid"; give a clear error here
        # 旧行为会把 404 误报成"凭证校验通过"，这里给出明确错误
        logger.error(
            lang.t("models_404", prefix=prefix)
        )
        return False

    if not 200 <= status < 300:
        # probe_prefix only accepts 2xx (with a real model list) and 401/403 as
        # conclusive, so anything reaching here means no candidate could be confirmed
        # (5xx, or a 200 that is not the model list). Reporting that as "credentials
        # valid" would be a claim nothing supports.
        #
        # probe_prefix 只把 2xx（且响应体确实是模型列表）与 401/403 当作结论，因此走到这里
        # 说明没有任何候选能被确认（5xx，或 200 但不是模型列表）。把它报成"凭证校验通过"
        # 是没有任何依据的结论。
        logger.error(lang.t("startup_bad_status", prefix=prefix, status=status))
        probe_health.record_degraded(f"{prefix}/models -> HTTP {status}")
        return False

    probe_health.record_credentials_ok()
    if not quiet_success:
        logger.info(lang.t("creds_ok", status=status, desc=session.describe()))
    return True


# --------------------------------------------------------------------------- #
# Per-model probe: reasoning efforts, capabilities and request parameters
# 逐模型探测：思考挡位、能力与请求参数
# --------------------------------------------------------------------------- #
# Upstream statuses meaning "the engine rejected the request body", i.e. the probe
# learned something definite. Anything else (404, 5xx) is transient.
#
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


async def _probe_model(
    session: Any,
    model_id: str,
    fingerprint: str,
    auth_dead: Optional[asyncio.Event] = None,
) -> ModelProbe:
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
        failure is transient and never a claim about the model. Once any worker has
        seen an auth failure, later requests short-circuit instead of hammering the
        dead session again.

        发送一次探测请求。凭证失效中止整轮刷新；传输失败是暂时性的，绝不构成
        关于该模型的任何声明。任一 worker 发现凭证失效后，后续请求直接短路，
        不再对着死会话多撞一次。
        """
        if auth_dead is not None and auth_dead.is_set():
            raise _ProbeAuthExpired("credentials already known dead")
        try:
            response = await upstream.post(
                session,
                "chat/completions",
                payload,
                extra_headers=PROBE_REQUEST_HEADERS,
            )
        except UpstreamUnavailable as exc:
            raise _ProbeTransient(str(exc)) from exc
        if response.status_code in AUTH_FAILURE_CODES:
            status_code = response.status_code
            await _close(response)
            if auth_dead is not None:
                auth_dead.set()
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
    # The 200 that cleared the merged parameter request. It carries no `reasoning_effort`
    # either, so it doubles as a default-behaviour observation (see step 5, D4).
    #
    # 通过合并参数请求的那个 200。它同样不带 `reasoning_effort`，因此也可以充当
    # 默认行为的观察结果（见第 5 步，D4）。
    parameter_body: Optional[str] = None
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
            parameter_body = body
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
        if not blamed_set & set(remaining):
            # The blame landed on a parameter that is no longer under test (typically one
            # already disproved in an earlier round). Claim nothing and stop, instead of
            # re-sending the very same request until the round budget runs out.
            #
            # 归因落在已不在测试范围内的参数上（通常是上一轮已被证伪的那个）：不做任何声明
            # 并立即停止，而不是把同一发请求重发到轮次耗尽。
            unresolved += 1
            break
        for parameter in blamed_set & set(remaining):
            parameter_accepted[parameter] = False
        remaining = [item for item in remaining if item not in blamed_set]

    if remaining:
        # Defensive invariant: should the loop ever end with parameters still undecided,
        # the probe must not claim they were established.
        #
        # 防御性不变式：循环若在仍有参数未定性的情况下结束，探测不得声称它们已确立。
        unresolved += 1

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
    if unprobeable and parameter_body is not None:
        # The merged parameter request omits `reasoning_effort` too, so its 200 already
        # answers "does thinking happen by default?" -- an unprobeable model (an upstream
        # that ignores the field) therefore needs one request fewer (D4).
        #
        # 合并参数请求同样省略了 `reasoning_effort`，它返回 200 就已经回答了
        # "默认是否思考"——因此不可探测的模型（上游忽略该字段）可以少发一次请求（D4）。
        probe.default_enabled = response_has_reasoning(parameter_body)
    else:
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
        if unresolved:
            # Keep the open questions visible even for unprobeable models: the
            # effort list is unknowable, but vision/parameters may be incomplete.
            #
            # 不可探测的模型也要让悬而未决的部分可见：挡位列表无从得知，
            # 但视觉/参数可能是残缺的。
            probe.last_error = f"{unresolved} probe request(s) left the answer open"
    elif unresolved:
        probe.status = STATUS_PARTIAL
        probe.last_error = f"{unresolved} probe request(s) left the answer open"
    else:
        probe.status = STATUS_OK
    return probe


# --------------------------------------------------------------------------- #
# Model-list fetch path / 模型列表拉取路径
# --------------------------------------------------------------------------- #
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


async def _get_models_or_unavailable(session: Any) -> httpx.Response:
    """
    GET /models, translating failures into an OpenAI-style HttpError.

    A request this process cannot even build (an unusable header value in
    session.json) answers 400 with the reason instead of escaping as an opaque 500 --
    the same classification the chat path uses (B12).


    GET /models，并把失败翻译成 OpenAI 风格的 HttpError。

    本进程连构造都做不到的请求（session.json 里有无法编码的头值）以 400 带上原因作答，
    而不是以一个语焉不详的 500 逃出去——与对话路径采用同一套归类（B12）。
    """
    try:
        return await upstream.get_models(session)
    except UpstreamUnavailable as exc:
        raise _http_error(
            exc.status_code, str(exc), error_type="server_error", code="upstream_unavailable"
        ) from exc
    except UpstreamRequestInvalid as exc:
        logger.error(lang.t("upstream_request_invalid", exc=exc))
        raise _http_error(
            400,
            lang.t("err_upstream_request_invalid", exc=exc),
            error_type="invalid_request_error",
            code="invalid_request",
        ) from exc


async def _fetch_raw_models(session: Any) -> List[Any]:
    """
    GET the upstream model list and return its raw entries.

    Every failure mode raises HttpError with an OpenAI-style body, so /v1/models and
    /v1/models/{id} share one code path.

    A 404 on a previously confirmed prefix self-heals: the upstream may have changed
    its API style (dropped /api/v1, or moved to it), and without this the prefix
    cache would pin every future models request to the dead route until restart. The
    prefix is re-probed once; the request is retried a single time when the probe
    settles on a different prefix.


    GET 上游模型列表并返回原始条目。

    所有失败模式都以 HttpError（OpenAI 风格错误体）抛出，使 /v1/models 与
    /v1/models/{id} 共用同一条代码路径。

    此前已确认的前缀返回 404 时会自愈：上游可能换了 API 风格（移除 /api/v1，
    或迁移到 /api/v1），若无此逻辑，前缀缓存会把之后的每次模型列表请求都钉死
    在死路由上，直到重启。这里重探一次前缀；若落点变了则重试一次。
    """
    resp = await _get_models_or_unavailable(session)

    if resp.status_code == 404 and upstream.prefix is not None:
        stale_prefix = upstream.prefix
        await _close(resp)
        try:
            await upstream.detect_prefix(session, refresh=True)
        except UpstreamUnavailable as exc:
            raise _http_error(
                exc.status_code, str(exc), error_type="server_error", code="upstream_unavailable"
            ) from exc
        if upstream.prefix != stale_prefix:
            logger.warning(
                lang.t("models_prefix_moved", old=stale_prefix, new=upstream.prefix)
            )
            resp = await _get_models_or_unavailable(session)

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
            # The detail goes to the log unconditionally (U-6): with the default
            # redaction it is the only place it exists, and without it "upstream /models
            # returned HTTP 502" would be the end of the trail.
            #
            # 细节无条件进日志（U-6）：默认脱敏时它是细节唯一存在的地方；没有它，
            # "上游 /models 返回 HTTP 502" 就是线索的尽头。上游响应体先清洗控制字符（L1）。
            logger.warning(
                lang.t(
                    "upstream_models_error_log",
                    status=resp.status_code,
                    text=sanitize_log_text(resp.text[:500]),
                )
            )
            message = (
                lang.t("err_upstream_models_http", status=resp.status_code, text=resp.text[:500])
                if settings.expose_upstream_error
                else lang.t("err_upstream_models_http_redacted", status=resp.status_code)
                + request_id_suffix()
            )
            raise _http_error(
                502,
                message,
                error_type="server_error",
                code="upstream_error",
            )
        try:
            payload = resp.json()
        except ValueError:
            logger.warning(
                lang.t(
                    "upstream_models_not_json_log",
                    text=sanitize_log_text(resp.text[:500]),
                )
            )
            message = (
                lang.t("err_upstream_models_not_json", text=resp.text[:500])
                if settings.expose_upstream_error
                else lang.t("err_upstream_models_not_json_redacted") + request_id_suffix()
            )
            raise _http_error(
                502,
                message,
                error_type="server_error",
                code="upstream_error",
            )
    finally:
        await _close(resp)

    return extract_model_list(payload)


# How long the raw upstream model list is reused before being fetched again
# (MODEL_LIST_TTL, 0 disables the cache). Polling clients would otherwise put one
# upstream /models request on the wire per /v1/models call. The list is small and
# "new model appears" is not latency-critical, so a short bounded staleness is a
# good trade. Only successful fetches populate the cache.
#
# 上游模型列表在重新拉取前的复用时长（MODEL_LIST_TTL，0 = 禁用缓存）。否则轮询型
# 客户端的每次 /v1/models 都会打一次上游 /models。列表很小、"新模型上架"也不是
# 延迟敏感事件，短而有界的陈旧度是划算的。只有成功拉取才写入缓存。
_models_cache_fetched_at: float = 0.0
_models_cache_entries: List[Any] = []


async def _get_raw_models_cached(session: Any) -> List[Any]:
    """
    _fetch_raw_models behind a short TTL. Concurrency note: several simultaneous
    misses each fetch (no in-flight dedup) -- harmless and rare; the TTL absorbs the
    steady-state traffic.
    """
    global _models_cache_fetched_at, _models_cache_entries
    now = time.time()
    if settings.model_list_ttl > 0 and _models_cache_entries and (
        now - _models_cache_fetched_at < settings.model_list_ttl
    ):
        return _models_cache_entries
    raw_models = await _fetch_raw_models(session)
    _models_cache_fetched_at = now
    _models_cache_entries = raw_models
    return raw_models


async def _fetch_model_summaries(session: Any) -> Optional[List[Tuple[str, str]]]:
    """
    Fetch the upstream model list and reduce it to (model_id, engine fingerprint)
    pairs; None means the list could not be retrieved (the caller skips this round).

    The normalized-model side of the reduction lives in models._model_summaries -- the
    normalization and fingerprint definitions stay in one place there, and this
    module only consumes them. That import used to go through app (a local import to
    dodge the cycle); with the normalization split out in R5 it is a plain module
    dependency, and this module no longer reaches back into the app layer at all.


    拉取上游模型列表并归约为 (模型 id, 引擎指纹) 对；None 表示拉取失败
    （调用方跳过本轮刷新）。

    规范化侧的归约在 models._model_summaries——规范化与指纹的定义在那里
    保持单一出处，本模块只是消费它。这个导入以前要经过 app（用局部导入绕开循环）；
    R5 把规范化拆出去之后它成了普通的模块依赖，本模块不再反向触碰 app 层。
    """
    try:
        raw_models = await _fetch_raw_models(session)
    except HttpError as exc:
        logger.warning(lang.t("probe_models_failed", status=exc.status_code))
        return None

    _, summaries = _model_summaries(raw_models)
    return summaries


# --------------------------------------------------------------------------- #
# Instance-level Open WebUI metadata (served as the envelope's "x_open_webui")
# 实例级 Open WebUI 元信息（作为信封的 "x_open_webui" 输出）
# --------------------------------------------------------------------------- #
# How long a /api/config snapshot is reused after a SUCCESSFUL fetch.
# /api/config 快照在成功拉取后的复用时长。
INSTANCE_META_TTL = 300.0
# How long to wait before retrying after a FAILED fetch: a short retry keeps the
# recovery quick instead of losing the metadata for a full TTL (the previous
# behavior pinned a failure for 300s).
#
# 失败后的重试间隔：短重试让恢复更快，而不是让一次失败把元信息
# 钉住整个 TTL（旧行为会把失败固化 300 秒）。
INSTANCE_META_RETRY = 30.0
# Bound on how long a caller waits for an in-flight refresh before giving up
# (the fetch itself is capped by INSTANCE_META_TIMEOUT in upstream.py).
INSTANCE_META_WAIT = 6.0


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
    # Timestamp of the last SUCCESSFUL refresh.
    # 最近一次成功刷新的时间。
    fetched_at: float = 0.0
    # Timestamp of the last attempt (success or failure) -- drives the retry pacing.
    # 最近一次尝试（无论成败）的时间——用于重试节流。
    attempted_at: float = 0.0

    def is_usable(self) -> bool:
        """Whether anything worth publishing is known. / 是否有值得输出的内容。"""
        return bool(self.name or self.version or self.features or self.default_model_capabilities)

    def is_fresh(self, now: float) -> bool:
        return bool(self.fetched_at) and (now - self.fetched_at) < INSTANCE_META_TTL

    def recently_attempted(self, now: float) -> bool:
        return bool(self.attempted_at) and (now - self.attempted_at) < INSTANCE_META_RETRY

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
# The in-flight /api/config refresh, so simultaneous stale /v1/models requests do
# not each hit the upstream (in-flight dedup; single-event-loop, check-then-set is
# atomic because no await sits between them).
#
# 在飞的 /api/config 刷新，使同时发现快照过期的多个 /v1/models 请求
# 不会各自打上游（in-flight 去重；单线程事件循环，检查与赋值之间
# 无 await，天然原子）。
_instance_meta_task: Optional[asyncio.Task] = None


async def _refresh_instance_meta(session: Any) -> None:
    config = await upstream.get_instance_config(session)
    if not isinstance(config, dict):
        # Failure: fetched_at stays untouched, so the next request retries after
        # INSTANCE_META_RETRY instead of waiting a full TTL.
        #
        # 失败：fetched_at 保持不变，下一次请求在 INSTANCE_META_RETRY 后
        # 重试，而不是等满整个 TTL。
        return
    _instance_meta.fetched_at = time.time()
    _instance_meta.name = str(config.get("name") or _instance_meta.name)
    _instance_meta.version = str(config.get("version") or _instance_meta.version)
    features = config.get("features")
    if isinstance(features, dict):
        _instance_meta.features = features


async def _await_instance_meta(task: asyncio.Task) -> None:
    """
    Wait for an in-flight /api/config refresh, at most INSTANCE_META_WAIT, without ever
    failing the caller.

    A timeout only ends the *wait* -- the shield keeps the refresh running for the next
    requester -- and is therefore expected and silent. Any other exception is a genuine
    defect (upstream failures are already turned into None by get_instance_config) and
    is logged rather than swallowed (I2): it used to disappear entirely, leaving
    "x_open_webui is always missing" as the only symptom to debug from.


    等待在飞的 /api/config 刷新，最多 INSTANCE_META_WAIT，且绝不让调用方失败。

    超时只结束**等待**——shield 让刷新继续运行、供下一个请求复用——因此它是预期内的，静默处理。
    其它异常属于真实缺陷（上游失败已被 get_instance_config 转成 None），这里记一行日志而不是
    吞掉（I2）：它过去会彻底消失，使"x_open_webui 一直缺失"成为唯一可用于排障的现象。
    """
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=INSTANCE_META_WAIT)
    except asyncio.TimeoutError:
        pass
    except Exception as exc:  # noqa: BLE001 - bonus metadata must never fail the caller
        logger.warning(lang.t("instance_meta_wait_failed", exc=exc))


async def _ensure_instance_meta(session: Any) -> None:
    """
    Refresh the /api/config snapshot when it is stale. Best effort: a failure keeps
    the previous snapshot (retried after a short INSTANCE_META_RETRY instead of a
    full TTL) and never fails /v1/models. Simultaneous refreshes share one task.

    快照过期时刷新 /api/config。尽力而为：失败保留上一份快照
    （短间隔 INSTANCE_META_RETRY 重试，而非等满 TTL），绝不让
    /v1/models 失败。并发刷新共用同一个任务。
    """
    global _instance_meta_task
    now = time.time()
    if _instance_meta.is_fresh(now) or _instance_meta.recently_attempted(now):
        return
    if _instance_meta_task is not None and not _instance_meta_task.done():
        await _await_instance_meta(_instance_meta_task)
        return
    _instance_meta.attempted_at = now
    _instance_meta_task = _spawn_background_task(_refresh_instance_meta(session))
    await _await_instance_meta(_instance_meta_task)


# --------------------------------------------------------------------------- #
# Refresh orchestration / 刷新编排
# --------------------------------------------------------------------------- #
@dataclass
class _RefreshState:
    """
    What a running refresh is probing right now, so /v1/models can decide whether
    waiting for it could actually change the answer.

    正在运行的刷新此刻在探测什么，供 /v1/models 判断"等它"是否真能改变结果。
    """

    pending: Set[str] = field(default_factory=set)
    # Set once the running refresh has published `pending`; lets /v1/models wait for
    # the announcement itself instead of assuming one event-loop turn is enough (any
    # async step added between task creation and the announcement would silently
    # break a sleep(0)-style handshake). Created lazily so it belongs to the running
    # event loop (same reasoning as _refresh_mutex below).
    #
    # 运行中的刷新公布 `pending` 后置位；让 /v1/models 等待"公布"这个事实本身，
    # 而不是假设一次事件循环轮转就够（在任务创建与公布之间新增任何异步步骤，都会
    # 让 sleep(0) 式握手静默失效）。惰性创建使其属于当前事件循环（理由同下面
    # 的 _refresh_mutex）。
    published: Optional[asyncio.Event] = None
    # Summaries handed over while a refresh was already in flight. The running
    # refresh consumes them when it finishes, so a freshly added model does not
    # have to wait for the *next* /v1/models to trigger its probe.
    #
    # 已有刷新在飞行时移交的 summaries。运行中的刷新收尾时会消化它们，
    # 使新上架的模型不必等下一次 /v1/models 才触发探测。
    deferred_summaries: Optional[List[Tuple[str, str]]] = None

    def announcement(self) -> asyncio.Event:
        """
        The announcement event, created on demand so waiters never see None.

        公布事件；按需创建，等待方永远拿不到 None。
        """
        if self.published is None:
            self.published = asyncio.Event()
        return self.published

    def announce(self, pending: Set[str]) -> None:
        """
        Publish what is being probed (an empty set is announced too) and wake the
        waiters.

        公布正在探测的集合（空集也公布），并唤醒等待方。
        """
        self.pending = set(pending)
        self.announcement().set()

    def reset_announcement(self) -> None:
        """
        Drop the previous run's announcement so waiters cannot mistake the old
        refresh's state for the new one.

        丢弃上一轮的公布，避免等待方把旧刷新的状态误当成新一轮的。
        """
        if self.published is not None:
            self.published.clear()


_refresh_state = _RefreshState()


# How long list_models waits for the running refresh to announce what it is probing
# before degrading to "serve immediately" (R7: no longer a magic number).
#
# list_models 等待在飞刷新公布探测内容、超时即退化为"立即返回"的时长
# （R7：不再是魔法数字）。
_PROBE_ANNOUNCE_TIMEOUT = 1.0


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


# A refresh must not run concurrently with itself: the heal path can fire while a
# startup / /v1/models refresh is still in flight, and two instances would probe the same
# models twice, interleave `_refresh_state.pending`, and make each other's backoff
# counters drift. The lock also makes "one refresh at a time" hold for the --probe CLI
# path.
#
# 刷新不得与自身并发：自愈可能在启动 / `/v1/models` 触发的刷新仍在飞行时触发，两个实例会
# 重复探测同一批模型、交错改写 `_refresh_state.pending`，并让双方的退避计数漂移。
# 这把锁同时让 `--probe` CLI 路径也遵守"同一时刻至多一个刷新"。
_refresh_lock: Optional[asyncio.Lock] = None


def _refresh_mutex() -> asyncio.Lock:
    """
    The refresh mutex, created lazily on first use.

    asyncio primitives bind to the event loop they are first used in, so a Lock built at
    import time could belong to a loop that `asyncio.run()` (--check / --probe) or
    uvicorn never uses -- and on Python 3.9 that binding happens even earlier. Creating
    it on first use always yields a lock of the current loop.


    刷新互斥锁，首次使用时惰性创建。

    asyncio 原语会绑定到首次使用它的事件循环，因此在 import 期构造的锁可能属于一个
    `asyncio.run()`（--check / --probe）或 uvicorn 都不会使用的循环——在 Python 3.9 上
    这种绑定甚至发生得更早。首次使用时创建，拿到的总是当前循环的锁。
    """
    global _refresh_lock
    if _refresh_lock is None:
        _refresh_lock = asyncio.Lock()
    return _refresh_lock


async def _refresh_model_probe(
    *,
    summaries: Optional[List[Tuple[str, str]]] = None,
    force: bool = False,
) -> bool:
    """
    Serialize cache refreshes; the actual work is in _refresh_model_probe_locked.

    串行化缓存刷新；实际工作见 _refresh_model_probe_locked。
    """
    async with _refresh_mutex():
        return await _refresh_model_probe_locked(summaries=summaries, force=force)


async def _refresh_model_probe_locked(
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
        probe_health.record_no_session(str(exc))
        return False

    if summaries is None:
        summaries = await _fetch_model_summaries(session)
        if summaries is None:
            probe_health.record_degraded(lang.t("probe_degraded_models"))
            return False

    model_probe.load()
    to_probe = model_probe.sync_with_models(summaries, force=force)
    # Announce unconditionally -- even an empty set: waiters must learn that this
    # refresh has nothing to probe instead of blocking until their own timeout.
    #
    # 无条件公布——空集也要公布：等待方必须得知这轮刷新没东西可探，
    # 而不是干等到自己超时。
    _refresh_state.announce(to_probe)
    if not to_probe:
        logger.info(lang.t("probe_cache_fresh", count=len(model_probe)))
        # A completed round with nothing to do is still a healthy round: the model list
        # came back and every entry was conclusive.
        #
        # 一轮"无活可干"的完成同样是健康的一轮：模型列表拿到了，而且每个条目都有结论。
        probe_health.record_round(
            {"probed": 0, "ok": 0, "partial": 0, "unprobeable": 0, "failed": 0}
        )
        return True

    fingerprints = dict(summaries)
    logger.info(lang.t("probe_begin", count=len(to_probe), models=", ".join(to_probe)))
    semaphore = asyncio.Semaphore(settings.model_probe_concurrency)
    counters = {"ok": 0, "partial": 0, "unprobeable": 0, "failed": 0}
    # Set as soon as any worker sees an auth failure: workers yet to send a request
    # short-circuit in ask() instead of hammering the dead session once more.
    #
    # 任一 worker 遇到凭证失效即置位：还没发请求的 worker 在 ask() 里直接短路，
    # 不再对着死会话多撞一次。
    auth_dead = asyncio.Event()

    async def worker(model_id: str) -> None:
        async with semaphore:
            try:
                probe = await asyncio.wait_for(
                    _probe_model(session, model_id, fingerprints[model_id], auth_dead),
                    timeout=settings.model_probe_timeout,
                )
            except _ProbeAuthExpired:
                auth_dead.set()
                raise
            except Exception as exc:  # noqa: BLE001 - per-model isolation
                model_probe.record_failure(model_id, fingerprints[model_id], str(exc))
                counters["failed"] += 1
                logger.warning(lang.t("probe_model_failed", model=model_id, exc=exc))
                return
            model_probe.record_result(model_id, probe)
            counters[probe.status] = counters.get(probe.status, 0) + 1
            logger.info(lang.t("probe_model_done", model=model_id, summary=_describe_probe(probe)))

    # TaskGroup (Python 3.11+): when one worker hits an auth failure the remaining
    # workers are CANCELLED -- gather() would let them keep hammering the dead
    # session, which is exactly what _ProbeAuthExpired promises to prevent.
    #
    # TaskGroup（Python 3.11+）：某个 worker 遇到凭证失效时其余 worker 会被取消——
    # gather() 会放任它们继续撞死会话，恰恰违背 _ProbeAuthExpired 的承诺。
    try:
        async with asyncio.TaskGroup() as task_group:
            for model_id in to_probe:
                task_group.create_task(worker(model_id))
    except _ProbeAuthExpired as exc:
        logger.error(lang.t("probe_auth_expired", status=exc))
        probe_health.record_auth_rejected(f"HTTP {exc}")
        # Still save whatever was collected before the credentials died
        # 凭证失效前收集到的结果仍然落盘
        model_probe.save()
        return False
    except BaseExceptionGroup as exc_group:
        # Several workers may fail with the same auth error; TaskGroup wraps them
        # (possibly alongside unrelated errors) in an ExceptionGroup.
        #
        # 多个 worker 可能因同一凭证问题失败；TaskGroup 会把异常
        # （可能混入无关异常）包成 ExceptionGroup。
        auth_errors = [
            sub
            for sub in _flatten_exception_group(exc_group)
            if isinstance(sub, _ProbeAuthExpired)
        ]
        if auth_errors:
            logger.error(lang.t("probe_auth_expired", status=auth_errors[0]))
            probe_health.record_auth_rejected(f"HTTP {auth_errors[0]}")
            model_probe.save()
            return False
        raise
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
    # U-11: the round reached the end, so the credentials answered for every model:
    # that is the "healthy" signal /healthz reports.
    #
    # U-11：这一轮走到了终点，说明凭证对每个模型都得到了应答：这就是 /healthz
    # 上报的"健康"信号。
    probe_health.record_round({"probed": len(to_probe), **counters})
    return True


def _flatten_exception_group(group: "BaseExceptionGroup[BaseException]") -> List[BaseException]:
    """Depth-first flatten of an (possibly nested) exception group. / 异常组的深度优先展平。"""
    flattened: List[BaseException] = []
    for exc in group.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            flattened.extend(_flatten_exception_group(exc))
        else:
            flattened.append(exc)
    return flattened


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
        # B8: hand this round's summaries to the running refresh so a freshly added
        # model is probed as soon as the current round finishes, instead of waiting
        # for a future /v1/models to notice.
        #
        # B8：把本轮 summaries 移交给在飞的刷新，使新上架的模型在当前一轮
        # 结束后即被探测，而不是等未来的某次 /v1/models 再发现。
        if summaries is not None:
            _refresh_state.deferred_summaries = summaries
        return _refresh_task

    # A fresh refresh will announce its own pending set: drop the previous run's
    # announcement first, so waiters cannot read stale state.
    #
    # 新一轮刷新会公布自己的 pending 集合：先丢弃上一轮的公布，
    # 等待方才不会读到过期状态。
    _refresh_state.reset_announcement()

    async def runner() -> None:
        try:
            await _refresh_model_probe(summaries=summaries)
            # Consume summaries that arrived while this refresh was running. Each
            # pass re-aligns the cache with the newest list; a pass that finds
            # nothing to probe is cheap.
            #
            # 消化本刷新运行期间到达的 summaries。每轮都把缓存与最新列表
            # 对齐；无东西可探的一轮代价很小。
            while True:
                deferred = _refresh_state.deferred_summaries
                if deferred is None:
                    break
                _refresh_state.deferred_summaries = None
                await _refresh_model_probe(summaries=deferred)
        except Exception as exc:  # noqa: BLE001 - background task must not die silently
            logger.warning(lang.t("probe_task_error", exc=exc))

    _refresh_task = asyncio.create_task(_in_detached_context(runner()))
    return _refresh_task


# Models that already have a heal probe scheduled, so a burst of bad requests cannot
# stampede the upstream with one refresh each.
#
# 已安排自愈探测的模型，避免一串坏请求各自触发一次刷新、把上游打爆。
_healing: Set[str] = set()

# Strong references to fire-and-forget background tasks. The event loop keeps only a
# weak reference to each task, so an unsaved task can be garbage-collected mid-run --
# its cleanup (cache save, _healing.discard) would silently never happen. Done tasks
# remove themselves from the set.
#
# 对"发射后不管"的后台任务持有强引用。事件循环只对每个 task 持有弱引用，不保存
# 引用的任务可能在运行途中被垃圾回收——其清理动作（缓存落盘、_healing.discard）
# 将静默永不发生。完成的任务会自行从集合中移除。
_background_tasks: Set[asyncio.Task] = set()


async def _in_detached_context(coro: Any) -> Any:
    """
    Run background work with no request id (L5).

    asyncio.create_task() copies the current context, so a task spawned while handling a
    request inherited that request's correlation id: the probe and heal log lines it
    produced were stamped with the id of whichever client happened to trigger them,
    which makes grepping by id misleading rather than useful. Clearing it inside the
    task touches only the task's own context, never the request's.

    在"没有请求 id"的上下文里运行后台工作（L5）。

    asyncio.create_task() 会复制当前上下文，因此在处理请求时派生的任务会继承该请求的关联
    id：它产生的探测与自愈日志会被打上"恰好触发了它的那个客户端"的 id，让按 id 检索日志
    从有用变成误导。在任务内部清空只影响该任务自己的上下文，不影响请求。
    """
    bind_request_id("")
    return await coro


def _spawn_background_task(coro: Any) -> asyncio.Task:
    """
    Create a fire-and-forget background task that cannot disappear to garbage
    collection before it finishes, and that carries no request id of its own (L5).

    创建一个"发射后不管"的后台任务：保证它在完成之前不会被垃圾回收掉，且不携带任何
    请求 id（L5）。
    """
    task = asyncio.create_task(_in_detached_context(coro))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _trigger_probe_heal(model_id: Optional[str], effort: Optional[str]) -> None:
    """
    React to a live upstream 400 about the reasoning effort: drop the disproved level
    from the cache immediately and re-probe that model in the background.

    The request that discovered the problem is never delayed by this.

    Ordering notes (B9/B14):
    * the cache is loaded FIRST -- the heal path can fire before any /v1/models or
      refresh has loaded the cache, and invalidating against an empty in-memory
      dict would silently do nothing while the later refresh would see a healthy
      on-disk entry and skip the re-probe entirely;
    * the invalidation happens BEFORE the re-entrancy check: a second live 400 for
      a model whose heal is already running must still drop its level, only the
      extra re-probe is skipped.


    对线上"关于思考挡位"的上游 400 作出反应：立即从缓存里剔除被证伪的挡位，并在
    后台重探该模型。发现问题的那个请求绝不因此被拖延。

    顺序说明（B9/B14）：
    * 必须先加载缓存——heal 可能在任何 /v1/models 或刷新加载缓存之前触发；
      若对空的内存字典做剔除，将静默无效，而随后的刷新看到磁盘上"健康"的条目
      又会跳过重探，整个自愈就此失效；
    * 剔除先于防重入判断：自愈进行中时同一模型的第二个线上 400 仍要剔除其挡位，
      被跳过的只是多余的重探。
    """
    if not model_id:
        return
    model_probe.load()
    entry = model_probe.entry(model_id)
    invalidated = model_probe.invalidate_effort(model_id, effort)
    if entry is not None and not invalidated:
        # Nothing was disproved (the level was not advertised): no re-probe needed.
        # 没有被证伪的东西（该挡位本就没被声明）：无需重探。
        return
    if model_id in _healing:
        # A heal refresh is already running; the invalidation above is the part
        # that matters right now.
        #
        # 自愈刷新已在跑；上面完成的剔除才是当下要紧的部分。
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

    _spawn_background_task(healer())


async def _shutdown_background_tasks() -> None:
    """
    Cancel and await the refresh/heal background tasks so uvicorn's shutdown does
    not leave them running against a closed httpx client.

    取消并等待刷新/自愈后台任务，避免 uvicorn 关停后它们仍拿着已关闭的
    httpx client 继续运行。
    """
    global _refresh_task
    tasks = [
        task
        for task in ([_refresh_task, *_background_tasks])
        if task is not None and not task.done()
    ]
    for task in tasks:
        task.cancel()
    if not tasks:
        _refresh_task = None
        return
    try:
        # Bounded: a wedged probe must not hold up the shutdown for long. On timeout
        # the gather itself is cancelled, which cancels the stragglers as well.
        #
        # 设上限：卡住的探测不能长时间拖住关停流程。超时后 gather 自身被取消，
        # 残留任务也会随之被取消。
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
        )
    except asyncio.TimeoutError:
        pass
    _refresh_task = None
