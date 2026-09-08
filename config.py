"""
Configuration module: read all settings from environment variables / a .env file.

All tunables live here to avoid os.getenv calls scattered across business code.

配置模块：统一从环境变量 / .env 文件读取设置。

所有可调项都集中在这里，避免在业务代码里散落 os.getenv。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
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


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(lang.t("int_invalid", name=name, raw=raw, default=default))
        return default


def _get_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(lang.t("float_invalid", name=name, raw=raw, default=default))
        return default
    if value < minimum:
        logger.warning(lang.t("float_below_min", name=name, raw=raw, minimum=minimum, default=default))
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


def _get_aliases(name: str) -> Dict[str, str]:
    """
    Parse model aliases, shaped like '{"gpt-4o": "gpt-4o-mini"}'.

    解析模型别名，形如 '{"gpt-4o": "gpt-4o-mini"}'。
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(lang.t("aliases_not_json", name=name))
        return {}
    if not isinstance(parsed, dict):
        logger.warning(lang.t("aliases_not_object", name=name))
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def _get_str_list(name: str) -> List[str]:
    """
    Parse a comma-separated list of strings, shaped like 'http://a,http://b'.

    解析逗号分隔的字符串列表，形如 'http://a,http://b'。
    """
    raw = os.getenv(name, "").strip()
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

    # ---------- Credentials ----------
    # ---------- 凭证 ----------
    session_file: Path

    # ---------- Reasoning-effort probe ----------
    # ---------- 思考挡位探测 ----------
    reasoning_cache_file: Path
    reasoning_probe_concurrency: int = 4
    reasoning_probe_timeout: float = 30.0
    reasoning_probe_wait: float = 5.0

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

    def resolve_model(self, model: Optional[str]) -> Optional[str]:
        if model is None:
            return None
        return self.model_aliases.get(model, model)


def load_settings() -> Settings:
    base_url = os.getenv("OPEN_WEBUI_BASE_URL", "http://localhost:8080").strip().rstrip("/")
    if not base_url:
        raise RuntimeError(lang.t("base_url_empty"))
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError(lang.t("base_url_invalid", url=base_url))

    style = os.getenv("UPSTREAM_API_STYLE", STYLE_AUTO).strip().lower()
    if style not in VALID_STYLES:
        logger.warning(lang.t("style_invalid", style=style, options="/".join(VALID_STYLES)))
        style = STYLE_AUTO

    debug = _get_bool("DEBUG", False)
    log_level = _get_log_level(debug)

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
        proxy_port=_get_int("PROXY_PORT", 8000),
        proxy_api_key=os.getenv("PROXY_API_KEY", "").strip(),
        session_file=Path(os.getenv("SESSION_FILE", "session.json")).expanduser(),
        # Reasoning-effort cache: refreshed only when the model list changes;
        # concurrency/timeout bound the per-model probe requests.
        #
        # 思考挡位缓存：仅在模型列表变化时刷新；并发数/超时约束逐模型的
        # 探测请求。
        reasoning_cache_file=Path(
            os.getenv("REASONING_CACHE_FILE", "reasoning_cache.json")
        ).expanduser(),
        reasoning_probe_concurrency=max(1, _get_int("REASONING_PROBE_CONCURRENCY", 4)),
        reasoning_probe_timeout=_get_float("REASONING_PROBE_TIMEOUT", 30.0, minimum=1.0),
        # How long /v1/models may block waiting for a missing-models probe to
        # finish before serving without the reasoning field (0 = never wait).
        #
        # /v1/models 在返回前最多等待缺失模型的探测完成多久
        # （0 = 从不等待，直接返回）。
        reasoning_probe_wait=_get_float("REASONING_PROBE_WAIT", 5.0, minimum=0.0),
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
        login_timeout=_get_int("LOGIN_TIMEOUT", 600),
        login_quiet_period=_get_float("LOGIN_QUIET_PERIOD", 6.0),
        login_headless=_get_bool("LOGIN_HEADLESS", False),
    )


settings = load_settings()
