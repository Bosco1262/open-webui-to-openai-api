"""
Configuration module: read all settings from environment variables / a .env file.

All tunables live here to avoid os.getenv calls scattered across business code.

配置模块：统一从环境变量 / .env 文件读取设置。

所有可调项都集中在这里，避免在业务代码里散落 os.getenv。
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

import lang

load_dotenv()

logger = logging.getLogger("webui-proxy.config")

# Fallback value when no User-Agent was captured
# 未抓到 User-Agent 时的兜底值
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

# Upstream API styles:
#   auto   - auto-detect /api/v1 -> /api
#   v1     - /api/v1 only (OpenAI-compatible routes of Open WebUI >= 0.6)
#   legacy - /api only (old versions)
#
# 上游 API 风格：
#   auto   - 自动探测 /api/v1 -> /api
#   v1     - 仅用 /api/v1（Open WebUI >= 0.6 的 OpenAI 兼容路由）
#   legacy - 仅用 /api（老版本）
STYLE_AUTO = "auto"
STYLE_V1 = "v1"
STYLE_LEGACY = "legacy"
VALID_STYLES = (STYLE_AUTO, STYLE_V1, STYLE_LEGACY)

# Must be understood by both logging and uvicorn, otherwise uvicorn's log level goes unchecked
# 同时要喂给 logging 和 uvicorn，必须是两边都认得的值，否则 uvicorn 的日志级别会失控
VALID_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE")

# Default subpaths the /v1/* catch-all passthrough forwards (U-1). Least privilege is
# the default: every path listed here is forwarded with the operator's captured
# credentials attached, so an entry is a grant of "whatever this account can do at
# that upstream route" to everyone holding a proxy key.
#
# "*" as the whole value restores the historical forward-everything behavior and is
# the only way to get it -- an explicit, reviewable decision.
#
# /v1/* 兜底透传默认转发的子路径（U-1）。默认最小权限：此处每列一条，就等于把
# "运维账号在该上游路由上能做的一切"授予每一位代理 Key 持有者。
#
# 整体取值为 "*" 时恢复历史上"全量透传"的行为，且这是唯一的获取方式——必须显式、
# 可审计地做出该决定。
DEFAULT_PASSTHROUGH_ALLOW = ("images", "audio", "files", "responses")
PASSTHROUGH_ALLOW_WILDCARD = "*"


def _strip_host_brackets(host: str) -> str:
    """
    Normalize a URL host for comparison: lowercase, no IPv6 brackets.

    规范化 URL 主机名以便比较：小写、去掉 IPv6 方括号。
    """
    return host.strip().lower().strip("[]")


def _is_loopback_host(host: str) -> bool:
    """
    Whether a host is loopback in the forms that matter here: "localhost" (and
    anything under .localhost), or a loopback IP literal.

    主机是否为回环形态：localhost（及 .localhost 子域），或回环 IP 字面量。
    """
    lowered = _strip_host_brackets(host)
    if not lowered:
        return False
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


def _validate_base_url(base_url: str) -> None:
    """
    Validate OPEN_WEBUI_BASE_URL: it must parse as an http(s) URL with a host, and
    carry a usable port.

    That is the whole rule. An earlier revision of this function also refused private
    / link-local / metadata addresses, plain http on a non-loopback host and
    non-standard ports -- copied from the Cloudflare Workers port, where those really
    are constraints (the Worker sits on the public internet). Here they are not: this
    is a local/LAN process, "http://localhost:8080" is the default, and
    "http://192.168.x.x:3000" is one of the main ways it is deployed. Refusing those
    would break the most typical setups to buy nothing; a LAN link being in cleartext
    is reported once at startup instead (see Settings.upstream_is_plain_http_nonloopback).

    校验 OPEN_WEBUI_BASE_URL 并原样返回：它必须能解析为带主机的 http(s) 地址，
    且端口可解析。

    规则就这些。本函数的早期版本还会拒绝私有 / 链路本地 / 元数据地址、非回环明文 http
    与非标准端口——那是从 Cloudflare Worker 移植版抄来的，那边确实是硬约束（Worker 跑在
    公网上）。这里不是：本仓库是本地 / 局域网进程，"http://localhost:8080" 就是默认值，
    "http://192.168.x.x:3000" 是主要用法之一。拒绝它们会打断最典型的部署方式却什么也换不来；
    局域网链路明文这件事改为启动时提示一次（见 Settings.upstream_is_plain_http_nonloopback）。
    """
    try:
        url = urlparse(base_url)
        url.port  # a malformed port ("host:abc") raises ValueError here
    except ValueError as exc:
        raise RuntimeError(lang.t("base_url_invalid", url=base_url)) from exc
    if url.scheme not in ("http", "https") or not url.netloc:
        raise RuntimeError(lang.t("base_url_invalid", url=base_url))


def _get_named_keys(env_var: str) -> Dict[str, str]:
    """
    Parse PROXY_API_KEYS: a comma-separated list of "name:key" entries, so each
    client can be told apart, rotated and revoked on its own (U-10).

    A bare entry (no colon) is accepted and auto-named; the colon in "name:key" is
    the first one, so a key containing colons must be written with an explicit name.
    Later duplicates of a name win, with a warning: silently keeping the first would
    make a rotation that edits only the key half appear to work.

    解析 PROXY_API_KEYS：逗号分隔的 "name:key" 条目，使每个客户端可被单独识别、
    轮换与撤销（U-10）。

    不带冒号的条目也可接受，会自动命名；"name:key" 以第一个冒号切分，因此键中含冒号
    时必须显式写名字。同名条目后者覆盖前者并给出告警：若静默保留前者，一次只改了
    键那半边的轮换会看起来"生效了"。
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return {}
    keys: Dict[str, str] = {}
    for index, item in enumerate(raw.split(","), start=1):
        entry = item.strip()
        if not entry:
            continue
        name, separator, value = entry.partition(":")
        if separator:
            name, value = name.strip(), value.strip()
        else:
            name, value = "", entry
        if not name:
            name = f"key{index}"
        if not value:
            logger.warning(lang.t("proxy_key_empty", name=name))
            continue
        if name in keys:
            logger.warning(lang.t("proxy_keys_duplicate", name=name))
        keys[name] = value
    return keys


def _get_bool(env_var: str, default: bool) -> bool:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(env_var: str, default: int, minimum: Optional[int] = None) -> int:
    raw = os.getenv(env_var)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(lang.t("int_invalid", name=env_var, raw=raw, default=default))
        return default
    if minimum is not None and value < minimum:
        logger.warning(
            lang.t("int_below_min", name=env_var, raw=raw, minimum=minimum, default=default)
        )
        return default
    return value


def _get_float(env_var: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(env_var)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(lang.t("float_invalid", name=env_var, raw=raw, default=default))
        return default
    if value < minimum:
        logger.warning(lang.t("float_below_min", name=env_var, raw=raw, minimum=minimum, default=default))
        return default
    return value


def _get_log_level(debug: bool) -> str:
    raw = os.getenv("LOG_LEVEL", "").strip().upper()
    if not raw:
        return "DEBUG" if debug else "INFO"
    if raw not in VALID_LOG_LEVELS:
        logger.warning(
            lang.t("log_level_invalid", raw=raw, options="/".join(VALID_LOG_LEVELS))
        )
        return "INFO"
    return raw


def _get_aliases(env_var: str) -> Dict[str, str]:
    """
    Parse model aliases, shaped like '{"gpt-4o": "gpt-4o-mini"}'.

    解析模型别名，形如 '{"gpt-4o": "gpt-4o-mini"}'。
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(lang.t("aliases_not_json", name=env_var))
        return {}
    if not isinstance(parsed, dict):
        logger.warning(lang.t("aliases_not_object", name=env_var))
        return {}
    return {str(key): str(value) for key, value in parsed.items()}


def _get_str_list(env_var: str) -> List[str]:
    """
    Parse a comma-separated list of strings, shaped like 'http://a,http://b'.

    解析逗号分隔的字符串列表，形如 'http://a,http://b'。
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # ---------- Upstream ----------
    # ---------- 上游 ----------
    open_webui_base_url: str
    upstream_api_style: str
    upstream_verify_ssl: bool
    upstream_trust_env: bool
    request_timeout: float
    connect_timeout: float

    # ---------- Public service ----------
    # ---------- 对外服务 ----------
    proxy_host: str
    proxy_port: int
    proxy_api_key: str
    # Named proxy keys (PROXY_API_KEYS), name -> key. The legacy single PROXY_API_KEY
    # keeps working and is treated as one more named key; either source enables auth
    # (U-10). Declared without a default so it stays grouped with the other public
    # service fields; load_settings is the only constructor.
    #
    # 具名代理 Key（PROXY_API_KEYS），名字 -> 密钥。旧的单个 PROXY_API_KEY 继续可用，
    # 视作又一把具名 Key；任一来源存在即启用鉴权（U-10）。刻意不给默认值，以便与其余
    # 对外服务字段放在一组；load_settings 是唯一的构造点。
    proxy_api_keys: Dict[str, str]

    # ---------- Credentials ----------
    # ---------- 凭证 ----------
    session_file: Path

    # ---------- Per-model probe (reasoning efforts + capabilities) ----------
    # ---------- 逐模型探测（思考挡位 + 能力） ----------
    model_probe_cache_file: Path
    model_probe_concurrency: int = 4
    model_probe_timeout: float = 30.0
    model_probe_wait: float = 5.0
    expose_instance_meta: bool = True
    # How long the raw upstream model list is reused before being fetched again
    # (0 = fetch on every request). Bounded staleness trades freshness for far
    # fewer upstream /models requests under polling clients.
    #
    # 上游模型列表的复用时长（0 = 每次请求都重新拉取）。用有界的陈旧度
    # 换取轮询型客户端下大幅减少的上游 /models 请求。
    model_list_ttl: float = 10.0
    # Whether upstream error bodies are quoted inside the error message returned
    # to the client. `false` (the default) keeps the details in the log only, and
    # the client gets a fixed message plus the request id that ties it to the log
    # line (U-6).
    #
    # 是否把上游错误响应体原文放进返回给客户端的错误消息。默认 `false`：细节只进日志，
    # 客户端收到固定文案 + 可与日志对上的 request id（U-6）。
    expose_upstream_error: bool = False
    # Reject JSON request bodies larger than this (bytes); guards the memory of an
    # unauthenticated / widely exposed deployment against oversized payloads.
    #
    # 拒绝超过该字节数的 JSON 请求体；防止超大载荷打爆无鉴权/广泛暴露部署的内存。
    max_body_bytes: int = 10 * 1024 * 1024
    # Safety interlock (D13): when PROXY_API_KEY is empty and PROXY_HOST is not a
    # loopback address, startup is refused unless this is explicitly set.
    #
    # 安全联锁（D13）：PROXY_API_KEY 为空且 PROXY_HOST 非回环地址时拒绝启动，
    # 除非显式设置本项。
    allow_insecure: bool = False
    # Subpaths the /v1/* catch-all passthrough may forward (U-1). Defaults to
    # DEFAULT_PASSTHROUGH_ALLOW; empty means "nothing" (deny all), and
    # passthrough_allow_all restores the historical forward-everything behavior for
    # an explicit PASSTHROUGH_ALLOW=*.
    #
    # /v1/* 兜底透传允许转发的子路径（U-1）。默认取 DEFAULT_PASSTHROUGH_ALLOW；
    # 为空表示"一条都不放行"（全拒），passthrough_allow_all 则对应显式的
    # PASSTHROUGH_ALLOW=*，恢复历史上"全量透传"的行为。
    passthrough_allow: List[str] = field(
        default_factory=lambda: list(DEFAULT_PASSTHROUGH_ALLOW)
    )
    passthrough_allow_all: bool = False

    # ---------- Behavior ----------
    # ---------- 行为 ----------
    model_aliases: Dict[str, str] = field(default_factory=dict)
    cors_origins: List[str] = field(default_factory=list)
    debug: bool = False
    log_level: str = "INFO"
    # Output language: zh / en (resolved at startup; see lang.resolve_language)
    # 输出语言：zh / en（启动时解析；见 lang.resolve_language）
    language: str = "en"

    # ---------- Browser login ----------
    # ---------- 浏览器登录 ----------
    login_timeout: int = 600
    login_quiet_period: float = 6.0
    login_headless: bool = False

    def authentication_enabled(self) -> bool:
        """
        Whether any proxy key is configured. Auth is off only when neither
        PROXY_API_KEY nor PROXY_API_KEYS provides one.

        是否配置了任何代理 Key。只有 PROXY_API_KEY 与 PROXY_API_KEYS 都没提供时才关闭鉴权。
        """
        return bool(self.proxy_api_key) or bool(self.proxy_api_keys)

    def upstream_is_plain_http_nonloopback(self) -> bool:
        """
        Whether the upstream is reached over plain http on a non-loopback host (U-8):
        the moment credentials and chat content cross the network in clear text.

        Deliberately a *notice*, not a refusal. A LAN Open WebUI on
        "http://192.168.x.x:3000" is one of this project's main deployment shapes, and
        the Cloudflare Workers port's "https on 443 only" rule is a constraint of that
        platform, not a defect of this one. Loopback ("http://localhost:8080", the
        default) stays silent -- it is not on the network at all, and warning about it
        would only train the operator to ignore the warning.

        上游是否通过非回环主机上的明文 http 访问（U-8）：此刻凭证与对话内容会明文过网。

        刻意只是**提示**，不是拒绝。局域网里的 Open WebUI（"http://192.168.x.x:3000"）
        是本项目主要部署形态之一，而 Cloudflare Worker 移植版"仅 443 端口 https"的规则
        是那个平台的约束，不是本仓库的缺陷。回环地址（默认的 "http://localhost:8080"）
        保持静默——它根本不在网络上，为它告警只会让运维学会忽略这条告警。
        """
        try:
            url = urlparse(self.open_webui_base_url)
        except ValueError:
            return False
        if url.scheme != "http" or not url.hostname:
            return False
        return not _is_loopback_host(url.hostname)

    def passthrough_permits(self, path: str) -> bool:
        """
        Whether the /v1/* catch-all may forward `path` (U-1): exact match or subpath
        of an allowlisted entry ("responses" permits /v1/responses and
        /v1/responses/...). An empty allowlist denies everything.

        兜底透传是否允许转发 `path`（U-1）：与白名单条目精确相等，或为其子路径
        （"responses" 放行 /v1/responses 与 /v1/responses/...）。白名单为空即全拒。
        """
        if self.passthrough_allow_all:
            return True
        normalized = path.strip("/")
        return any(
            normalized == allowed or normalized.startswith(allowed + "/")
            for allowed in self.passthrough_allow
        )

    def upstream_url(self, prefix: str, subpath: str) -> str:
        subpath = subpath.lstrip("/")
        return f"{self.open_webui_base_url}{prefix}/{subpath}" if subpath else f"{self.open_webui_base_url}{prefix}"

    def prefix_candidates(self) -> List[str]:
        """Return the candidate upstream prefixes in probe priority order.

        按优先级返回待探测的上游前缀。
        """
        if self.upstream_api_style == STYLE_V1:
            return ["/api/v1"]
        if self.upstream_api_style == STYLE_LEGACY:
            return ["/api"]
        return ["/api/v1", "/api"]

    def resolve_model(self, model: Any) -> Any:
        """
        Map a client-requested model name through MODEL_ALIASES.

        A malformed body may carry a non-string `model` (dict/list). Such a value has
        no alias semantics and is not hashable either, so it is returned unchanged:
        callers validate it and answer 400, instead of this lookup raising TypeError
        and turning a client mistake into an HTTP 500.

        用 MODEL_ALIASES 映射客户端请求的模型名。

        畸形请求体可能携带非字符串的 `model`（dict/list）：它没有别名语义，也不可哈希，
        因此原样返回——由调用方校验后以 400 拒绝，而不是在这里抛 TypeError、
        把客户端的错误变成 HTTP 500。
        """
        if not isinstance(model, str):
            return model
        return self.model_aliases.get(model, model)


def load_settings() -> Settings:
    base_url = os.getenv("OPEN_WEBUI_BASE_URL", "http://localhost:8080").strip().rstrip("/")
    if not base_url:
        raise RuntimeError(lang.t("base_url_empty"))
    # Only "is this a usable http(s) URL" is enforceable here (U-8). Whether the link
    # is cleartext is reported once at startup instead of being refused: see
    # Settings.upstream_is_plain_http_nonloopback.
    #
    # 这里只强制"是不是可用的 http(s) URL"（U-8）。链路是否明文改为启动时提示一次，
    # 而不是拒绝：见 Settings.upstream_is_plain_http_nonloopback。
    _validate_base_url(base_url)

    style = os.getenv("UPSTREAM_API_STYLE", STYLE_AUTO).strip().lower()
    if style not in VALID_STYLES:
        logger.warning(lang.t("style_invalid", style=style, options="/".join(VALID_STYLES)))
        style = STYLE_AUTO

    debug = _get_bool("DEBUG", False)
    log_level = _get_log_level(debug)

    # The /v1/* catch-all allowlist (U-1): unset -> the least-privilege default;
    # "*" -> the historical forward-everything behavior; anything else -> that exact
    # list (an explicitly empty value therefore denies every passthrough path).
    #
    # /v1/* 兜底透传白名单（U-1）：未设置 -> 默认最小权限；"*" -> 历史上"全量透传"；
    # 其它 -> 恰好是列出的那些（因此显式空值意味着一条都不放行）。
    raw_passthrough = os.getenv("PASSTHROUGH_ALLOW")
    if raw_passthrough is None:
        passthrough_allow = list(DEFAULT_PASSTHROUGH_ALLOW)
        passthrough_allow_all = False
    else:
        entries = [item.strip() for item in raw_passthrough.split(",") if item.strip()]
        passthrough_allow_all = PASSTHROUGH_ALLOW_WILDCARD in entries
        passthrough_allow = [
            item for item in entries if item != PASSTHROUGH_ALLOW_WILDCARD
        ]

    return Settings(
        open_webui_base_url=base_url,
        upstream_api_style=style,
        upstream_verify_ssl=_get_bool("UPSTREAM_VERIFY_SSL", True),
        # If HTTP_PROXY is configured system-wide, a loopback upstream would wrongly be
        # routed through the proxy; set to false to bypass when the upstream is intranet.
        #
        # 系统里若配置了 HTTP_PROXY，回环地址的上游会被错误地送去走代理，
        # 上游在内网时可设为 false 绕过。
        upstream_trust_env=_get_bool("UPSTREAM_TRUST_ENV", True),
        # Timeouts must be positive; 0 makes httpx time out immediately
        # 超时必须是正数，0 会让 httpx 立刻超时
        request_timeout=_get_float("REQUEST_TIMEOUT", 300.0, minimum=1.0),
        connect_timeout=_get_float("CONNECT_TIMEOUT", 10.0, minimum=0.1),
        proxy_host=os.getenv("PROXY_HOST", "0.0.0.0").strip() or "0.0.0.0",
        proxy_port=_get_int("PROXY_PORT", 8000, minimum=1),
        proxy_api_key=os.getenv("PROXY_API_KEY", "").strip(),
        proxy_api_keys=_get_named_keys("PROXY_API_KEYS"),
        session_file=Path(os.getenv("SESSION_FILE", "session.json")).expanduser(),
        # Per-model probe cache: reused as-is while the engine fingerprint of a
        # model is unchanged; concurrency/timeout bound the probe requests and
        # `model_probe_wait` bounds how long /v1/models may block on a probe that
        # is actually in flight.
        #
        # 逐模型探测缓存：只要模型的引擎指纹不变就直接复用；并发数/超时约束探测
        # 请求，`model_probe_wait` 约束 /v1/models 在"确有探测在飞行中"时的等待上限。
        model_probe_cache_file=Path(
            os.getenv("MODEL_PROBE_CACHE_FILE", "model_probe_cache.json")
        ).expanduser(),
        model_probe_concurrency=max(1, _get_int("MODEL_PROBE_CONCURRENCY", 4)),
        model_probe_timeout=_get_float("MODEL_PROBE_TIMEOUT", 30.0, minimum=1.0),
        # How long /v1/models may block waiting for an in-flight probe before
        # serving without the probe fields (0 = never wait).
        #
        # /v1/models 在确有探测飞行中时最多等待多久再返回（0 = 从不等待）。
        model_probe_wait=_get_float("MODEL_PROBE_WAIT", 5.0, minimum=0.0),
        # Instance-level Open WebUI metadata (feature switches, the default model
        # metadata template) is exposed in the /v1/models envelope under
        # "x_open_webui"; turn off for clients that reject unknown envelope keys.
        #
        # 实例级 Open WebUI 元信息（功能开关、默认模型元数据模板）以 "x_open_webui"
        # 放在 /v1/models 信封里；对拒绝未知信封键的客户端可关闭。
        expose_instance_meta=_get_bool("EXPOSE_INSTANCE_META", True),
        model_list_ttl=_get_float("MODEL_LIST_TTL", 10.0, minimum=0.0),
        # U-6: upstream error bodies stay in this service's log by default. Those
        # bodies routinely name internal hosts, file paths and network details, which
        # is exactly the material a follow-up attack wants; a client debugging its own
        # request is served better by the request id than by an internal English error.
        #
        # U-6：上游错误体默认只进本服务日志。那些响应体经常暴露内部主机名、文件路径与
        # 网络细节，正是后续定向攻击所需的材料；对着 request id 查日志，比把一段内部
        # 英文错误原样丢给客户端更有用。
        expose_upstream_error=_get_bool("EXPOSE_UPSTREAM_ERROR", False),
        max_body_bytes=_get_int("MAX_BODY_BYTES", 10 * 1024 * 1024, minimum=1),
        allow_insecure=_get_bool("ALLOW_INSECURE", False),
        passthrough_allow=passthrough_allow,
        passthrough_allow_all=passthrough_allow_all,
        model_aliases=_get_aliases("MODEL_ALIASES"),
        cors_origins=_get_str_list("PROXY_CORS_ORIGINS"),
        debug=debug,
        log_level=log_level,
        # No CLI flag here (config loads at import time): use system detection, the
        # --lang flag overrides later via dataclasses.replace in app.main.
        #
        # 此处拿不到 CLI 参数（config 在 import 期加载）：用系统检测，
        # --lang 参数之后在 app.main 里通过 dataclasses.replace 覆盖。
        language=lang.detect_system_language(),
        login_timeout=_get_int("LOGIN_TIMEOUT", 600, minimum=1),
        login_quiet_period=_get_float("LOGIN_QUIET_PERIOD", 6.0),
        login_headless=_get_bool("LOGIN_HEADLESS", False),
    )


settings = load_settings()
