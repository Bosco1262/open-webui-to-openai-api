"""
User-facing message localization (zh / en).

Language selection priority: CLI --lang flag > system language detection > English.

At import time the module initializes with the system language, so messages logged
during import (e.g. config warnings) are localized too. The CLI flag overrides it
later via configure().

用户可见消息的本地化（zh / en）。

语言选择优先级：CLI --lang 参数 > 系统语言检测 > 默认英文。

模块导入时即按系统语言初始化，因此 import 期的日志（如配置告警）也会被本地化；
CLI 参数随后通过 configure() 覆盖。
"""

from __future__ import annotations

import locale
import os
import sys
from typing import Dict, NamedTuple, Optional, Tuple

LANG_ZH = "zh"
LANG_EN = "en"
LANG_AUTO = "auto"


class Message(NamedTuple):
    """
    One translatable message, with both languages as named fields.

    Replaces the previous (en, zh) tuple that callers indexed with 0/1 -- a magic
    index that quietly breaks the moment a third language is added.

    一条可翻译消息，两种语言各有具名字段。

    取代此前用 0/1 下标访问的 (en, zh) 元组——那是"魔法下标"，
    一旦加入第三种语言就会静默出错。
    """

    en: str
    zh: str


# Raw message table: key -> (english, chinese). Kept as plain tuples for
# readability at this size; converted into Message objects below so nothing
# downstream ever indexes a language by position.
#
# 原始消息表：key -> (英文, 中文)。这个体量下用普通元组更好读；下面会转换成
# Message 对象，使下游代码永远不必用位置下标去取某种语言。
_MESSAGE_TEXTS: Dict[str, Tuple[str, str]] = {
    # ---------------- config.py ----------------
    "int_invalid": (
        "{name}={raw!r} is not a valid integer, falling back to default {default}",
        "{name}={raw!r} 不是合法整数，回退为默认值 {default}",
    ),
    "float_invalid": (
        "{name}={raw!r} is not a valid number, falling back to default {default}",
        "{name}={raw!r} 不是合法数字，回退为默认值 {default}",
    ),
    "float_below_min": (
        "{name}={raw!r} is below the minimum {minimum}, falling back to default {default}",
        "{name}={raw!r} 小于下界 {minimum}，回退为默认值 {default}",
    ),
    "log_level_invalid": (
        "LOG_LEVEL={raw!r} is invalid (options: {options}), falling back to INFO",
        "LOG_LEVEL={raw!r} 非法（可选 {options}），回退为 INFO",
    ),
    "aliases_not_json": (
        "{name} is not valid JSON, ignoring model alias config",
        "{name} 不是合法 JSON，已忽略模型别名配置",
    ),
    "aliases_not_object": (
        "{name} must be a JSON object, ignoring it",
        "{name} 需要是一个 JSON 对象，已忽略",
    ),
    "style_invalid": (
        "UPSTREAM_API_STYLE={style!r} is invalid (options: {options}), falling back to auto",
        "UPSTREAM_API_STYLE={style!r} 非法（可选 {options}），回退为 auto",
    ),
    "base_url_empty": (
        "OPEN_WEBUI_BASE_URL cannot be empty",
        "OPEN_WEBUI_BASE_URL 不能为空",
    ),
    "base_url_invalid": (
        "OPEN_WEBUI_BASE_URL={url!r} is not a valid address; use e.g. "
        "http://localhost:8080 or https://webui.example.com (include http:// or https://)",
        "OPEN_WEBUI_BASE_URL={url!r} 不是合法地址，"
        "需要形如 http://localhost:8080 或 https://webui.example.com（记得带 http:// 或 https://）。",
    ),
    # ---------------- app.py: startup / banner ----------------
    "startup_failed": (
        "Startup failed: {exc}",
        "启动失败：{exc}",
    ),
    "no_proxy_key": (
        "PROXY_API_KEY is empty -- auth is disabled; anyone who can reach this port can call the service.",
        "PROXY_API_KEY 为空 —— 代理服务未启用鉴权，任何能访问该端口的人都可调用。",
    ),
    "session_missing_hint": (
        "Credential file {path} not found. Run `python app.py --login` to complete the browser login first.",
        "未找到凭证文件 {path}，请先运行 `python app.py --login` 完成浏览器登录。",
    ),
    "creds_unusable": (
        "Credentials unusable: {exc}",
        "凭证不可用：{exc}",
    ),
    "startup_cant_connect": (
        "Cannot connect to upstream at startup: {exc}",
        "启动时无法连接上游：{exc}",
    ),
    "creds_expired": (
        "Credentials have expired (upstream returned HTTP {status}). Run `python app.py --login` to log in again.",
        "凭证已失效（上游返回 HTTP {status}）。请运行 `python app.py --login` 重新登录。",
    ),
    "models_404": (
        "Upstream {prefix}/models does not exist (HTTP 404). Make sure OPEN_WEBUI_BASE_URL "
        "points to Open WebUI, and set UPSTREAM_API_STYLE explicitly if necessary.",
        "上游 {prefix}/models 不存在（HTTP 404）。请确认 OPEN_WEBUI_BASE_URL 指向 Open WebUI，"
        "必要时显式设置 UPSTREAM_API_STYLE。",
    ),
    "creds_ok": (
        "Credential validation passed (upstream HTTP {status}, {desc})",
        "凭证校验通过（上游 HTTP {status}，{desc}）",
    ),
    "banner_start": (
        "open-webui-to-openai-api v{version} starting",
        "open-webui-to-openai-api v{version} 启动中",
    ),
    "banner_upstream": (
        "  Upstream  : {url}",
        "  上游地址 : {url}",
    ),
    "banner_listen_all": (
        "  Listen    : 0.0.0.0:{port} (all interfaces, server-side only)",
        "  监听地址 : 0.0.0.0:{port}（监听所有网卡，仅服务端配置）",
    ),
    "banner_local": (
        "  Local     : http://127.0.0.1:{port}/v1  <-- put this into the client",
        "  本机接入 : http://127.0.0.1:{port}/v1  <-- 填进客户端的地址",
    ),
    "banner_lan": (
        "  LAN       : http://<machine-IP>:{port}/v1",
        "  局域网接入: http://<本机IP>:{port}/v1",
    ),
    "banner_host": (
        "  Local     : http://{host}:{port}/v1  <-- put this into the client",
        "  本机接入 : http://{host}:{port}/v1  <-- 填进客户端的地址",
    ),
    "banner_session": (
        "  Credentials: {path}",
        "  凭证文件 : {path}",
    ),
    "banner_style": (
        "  API style : {style}",
        "  API 风格 : {style}",
    ),
    # ---------------- app.py: OpenAI-style errors ----------------
    "err_invalid_api_key": (
        "Invalid proxy API key.",
        "无效的代理 API Key。",
    ),
    "err_invalid_json": (
        "Request body is not valid JSON.",
        "请求体不是合法 JSON。",
    ),
    "err_json_object": (
        "Request body must be a JSON object.",
        "请求体必须是 JSON 对象。",
    ),
    "err_missing_model": (
        "Missing required field: model.",
        "缺少必填字段：model。",
    ),
    "err_messages_empty": (
        "messages must be a non-empty array.",
        "messages 必须是非空数组。",
    ),
    "err_missing_model_input": (
        "Missing required fields: model / input.",
        "缺少必填字段：model / input。",
    ),
    "err_passthrough_path": (
        "Specify which upstream endpoint to forward in the path.",
        "请在路径中指定要转发的上游接口。",
    ),
    "err_upstream_http": (
        "Upstream returned HTTP {status}: {text}",
        "上游返回 HTTP {status}：{text}",
    ),
    "err_upstream_models_http": (
        "Upstream /models returned HTTP {status}: {text}",
        "上游 /models 返回 HTTP {status}：{text}",
    ),
    "err_upstream_not_json": (
        "Upstream returned invalid JSON: {body!r}",
        "上游返回的不是合法 JSON：{body!r}",
    ),
    "err_upstream_models_not_json": (
        "Upstream /models returned invalid JSON: {text}",
        "上游 /models 返回的不是合法 JSON：{text}",
    ),
    "err_model_not_found": (
        "The model '{model}' does not exist",
        "模型 '{model}' 不存在",
    ),
    "err_upstream_unauthorized": (
        "Open WebUI rejected this request (credentials may have expired). "
        "Run `python app.py --login` to log in again.",
        "Open WebUI 拒绝了本次请求（凭证可能已过期）。请运行 `python app.py --login` 重新登录。",
    ),
    "err_cannot_read_body": (
        "<cannot read response body>",
        "<无法读取响应体>",
    ),
    # ---------------- app.py: logging / CLI ----------------
    "auth_failure_log": (
        "Upstream returned HTTP {status}; credentials may have expired.",
        "上游返回 HTTP {status}，凭证可能已过期。",
    ),
    "forward_chat": (
        "Forwarding chat request model={model} stream={stream}",
        "转发聊天请求 model={model} stream={stream}",
    ),
    "sse_client_disconnected": (
        "Client disconnected, terminating upstream stream.",
        "客户端已断开，终止上游流。",
    ),
    "sse_stream_ended": (
        "Upstream stream ended early ({etype}): {exc}",
        "上游流提前结束（{etype}）：{exc}",
    ),
    "check_session_missing": (
        "Credential file {path} not found",
        "未找到凭证文件 {path}",
    ),
    "check_summary": (
        "Credential summary: {desc}",
        "凭证摘要：{desc}",
    ),
    "endpoint_passthrough": (
        "ANY  /v1/{path}  (passthrough)",
        "ANY  /v1/{path}  (透传)",
    ),
    "endpoint_retrieve_model": (
        "GET  /v1/models/{id}",
        "GET  /v1/models/{id}",
    ),
    "cli_description": (
        "Open WebUI -> OpenAI API reverse proxy",
        "Open WebUI -> OpenAI API 反向代理",
    ),
    "app_description": (
        "Reverse-proxy an Open WebUI instance as an OpenAI-compatible API.",
        "将 Open WebUI 实例反向代理为 OpenAI 兼容的 API 接口。",
    ),
    "cli_login_help": (
        "Force a browser re-login and refresh credentials",
        "强制重新进行浏览器登录并刷新凭证",
    ),
    "cli_check_help": (
        "Only validate credentials and upstream connectivity, then exit",
        "只校验凭证与上游连通性，然后退出",
    ),
    "cli_host_help": (
        "Override PROXY_HOST",
        "覆盖 PROXY_HOST",
    ),
    "cli_port_help": (
        "Override PROXY_PORT",
        "覆盖 PROXY_PORT",
    ),
    "cli_lang_help": (
        "Output language: zh / en / auto (default: auto = follow the system language, English if undetectable)",
        "输出语言：zh / en / auto（默认 auto = 系统语言，检测不到时用英文）",
    ),
    # ---------------- per-model probe / 逐模型探测 ----------------
    "probe_cache_fresh": (
        "Per-model probe cache is up to date ({count} model(s)); skipping probe",
        "逐模型探测缓存已是最新（{count} 个模型），跳过探测",
    ),
    "probe_cache_saved": (
        "Per-model probe cache saved to {path} ({count} model(s))",
        "逐模型探测缓存已保存到 {path}（{count} 个模型）",
    ),
    "probe_begin": (
        "Probing {count} model(s): {models}",
        "开始探测 {count} 个模型：{models}",
    ),
    "probe_model_done": (
        "  {model}: {summary}",
        "  {model}：{summary}",
    ),
    "probe_model_failed": (
        "  {model}: probe failed ({exc}); will retry with backoff",
        "  {model}：探测失败（{exc}），将按退避重试",
    ),
    "probe_auth_expired": (
        "Credentials expired while probing (HTTP {status}); aborting",
        "探测期间凭证失效（HTTP {status}），已中止",
    ),
    "probe_finished": (
        "Probe finished: {ok} complete, {partial} partial, {unprobeable} unprobeable, {failed} failed",
        "探测完成：完整 {ok} 个，部分 {partial} 个，不可探测 {unprobeable} 个，失败 {failed} 个",
    ),
    "probe_models_failed": (
        "Cannot fetch the model list for probing (HTTP {status})",
        "无法获取模型列表以进行探测（HTTP {status}）",
    ),
    "probe_task_error": (
        "Probe refresh task crashed: {exc}",
        "探测刷新任务异常终止：{exc}",
    ),
    "probe_wait_timeout": (
        "Probe refresh did not finish within {wait}s; serving the model list without "
        "the unfinished fields (they appear once the background refresh lands)",
        "探测刷新未在 {wait} 秒内完成，本次模型列表先不带未完成的字段返回"
        "（后台刷新完成后即可看到）",
    ),
    "instance_meta_failed": (
        "Instance metadata (/api/config) unavailable: {exc}",
        "实例元信息（/api/config）不可用：{exc}",
    ),
    "cli_probe_help": (
        "Force a refresh of the per-model probe cache, then exit",
        "强制刷新逐模型探测缓存后退出",
    ),
    # ---------------- session_store.py ----------------
    "playwright_missing": (
        "playwright is not installed; cannot start browser login.\n"
        "    pip install -r requirements-browser.txt\n"
        "    playwright install chromium\n"
        "Or prepare session.json manually and start the service directly.",
        "未安装 playwright，无法启动浏览器登录。\n"
        "    pip install -r requirements-browser.txt\n"
        "    playwright install chromium\n"
        "或者手动准备 session.json 后直接启动服务。",
    ),
    "login_banner_open": (
        "  Opening browser, please log in: {url}",
        "  即将打开浏览器，请在窗口中登录：{url}",
    ),
    "login_banner_auto": (
        "  Credentials are captured and the browser closes automatically after login (no extra steps needed).",
        "  登录成功后脚本会自动抓取凭证并关闭浏览器（无需任何额外操作）。",
    ),
    "login_banner_portal": (
        "  If the browser jumps to a campus/corporate network auth page, finish network authentication first, then return to log in.",
        "  若浏览器跳到校园网 / 公司网认证页，请先完成网络认证再回到登录。",
    ),
    "login_banner_timeout": (
        "  Waiting at most {timeout} seconds.",
        "  最长等待 {timeout} 秒。",
    ),
    "session_missing_file": (
        "Credential file {path} not found. Run `python app.py --login` to complete a browser login.",
        "未找到凭证文件 {path}。请运行 `python app.py --login` 完成一次浏览器登录。",
    ),
    "session_unparseable": (
        "Credential file {path} cannot be parsed: {exc}",
        "凭证文件 {path} 无法解析：{exc}",
    ),
    "session_wrong_shape": (
        "Credential file {path} has an invalid format; expected a JSON object.",
        "凭证文件 {path} 内容格式不正确，应为 JSON 对象。",
    ),
    "session_no_creds": (
        "Credential file {path} has neither Authorization nor Cookie; run `python app.py --login` again.",
        "凭证文件 {path} 里既没有 Authorization 也没有 Cookie，请重新运行 `python app.py --login`。",
    ),
    "login_timeout": (
        "Login timed out ({timeout} seconds); no valid credentials captured, please retry.\n"
        "If the browser jumped to a campus/corporate network auth page, finish network authentication "
        "first, then return to Open WebUI and log in.",
        "登录超时（{timeout} 秒），未捕获到有效凭证，请重试。\n"
        "如果浏览器跳到了校园网 / 公司网认证页，请先完成网络认证，"
        "再回到 Open WebUI 完成登录。",
    ),
    "login_interrupted": (
        "Browser login interrupted: {exc}",
        "浏览器登录中断：{exc}",
    ),
    "login_browser_exit": (
        "Browser exited abnormally ({exc}); continuing with captured credentials.",
        "浏览器异常退出（{exc}），使用已抓到的凭证继续。",
    ),
    "login_no_creds": (
        "No credentials captured; check whether you completed the login in the browser.",
        "未能捕获到任何凭证，请检查是否在浏览器中完成了登录。",
    ),
    "creds_saved": (
        "Credentials saved to {path} ({desc})",
        "凭证已保存到 {path}（{desc}）",
    ),
    "capture_error_debug": (
        "Error while capturing request headers: {exc}",
        "抓取请求头时出错：{exc}",
    ),
    "validate_failed": (
        "Captured credentials failed upstream validation (possibly an expired token, or a campus "
        "portal not yet authenticated); keep waiting for login...",
        "抓到的凭证未通过上游校验（可能是过期 Token，或校园网等门户尚未完成网络认证），继续等待登录...",
    ),
    "goto_failed": (
        "Failed to open the homepage ({exc}); ignore if already logged in.",
        "打开首页失败（{exc}），如已登录可忽略。",
    ),
    "localStorage_failed": (
        "Failed to read localStorage: {exc}",
        "读取 localStorage 失败：{exc}",
    ),
    "cookies_failed": (
        "Failed to read the cookie jar: {exc}",
        "读取 Cookie Jar 失败：{exc}",
    ),
    "describe_empty": (
        "<empty>",
        "<空>",
    ),
    # ---------------- upstream.py ----------------
    "unavailable_connect": (
        "Cannot connect to upstream {url}: {exc}",
        "无法连接上游 {url}：{exc}",
    ),
    "probe_404": (
        "Upstream {url} does not exist (404), trying the next candidate prefix",
        "上游 {url} 不存在（404），尝试下一个候选前缀",
    ),
    "probe_result": (
        "Upstream API prefix probe result: {prefix} (HTTP {status})",
        "上游 API 前缀探测结果：{prefix}（HTTP {status}）",
    ),
    "all_404": (
        "All candidate prefixes returned 404 (last status {status}); falling back to {prefix}. "
        "Make sure OPEN_WEBUI_BASE_URL points to Open WebUI, not another service.",
        "所有候选前缀均返回 404（最后状态 {status}），回退为 {prefix}。请确认 OPEN_WEBUI_BASE_URL 指向 Open WebUI 而非其他服务。",
    ),
    "resp_fragment": (
        "Upstream response fragment: {text}",
        "上游响应片段：{text}",
    ),
    "request_failed": (
        "Request to {url} failed ({exc}); trying another prefix",
        "请求 {url} 失败（{exc}），尝试其它前缀",
    ),
    "fallback_404": (
        "Upstream {url} returned 404; falling back to {next}",
        "上游 {url} 返回 404，回退到 {next}",
    ),
    "unavailable_base": (
        "Cannot connect to upstream {base}: {exc}",
        "无法连接上游 {base}：{exc}",
    ),
}

# Named-field view of the table above; this is what the rest of the module uses.
# 上表的具名字段视图；本模块其余部分只使用这一份。
_MESSAGES: Dict[str, Message] = {
    key: Message(*texts) for key, texts in _MESSAGE_TEXTS.items()
}

# Current language; initialized with the system language at import time
# 当前语言；导入时按系统语言初始化
_current: str = LANG_EN


def detect_system_language() -> str:
    """
    Detect the system language from env vars / Windows UI language; fall back to English.

    从环境变量 / Windows UI 语言检测系统语言；检测不到时回退英文。
    """
    # POSIX-style env vars (highest confidence)
    # POSIX 风格环境变量（置信度最高）
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = (os.environ.get(var) or "").strip()
        if value:
            # Strip encoding (.UTF-8) and region (_CN / -CN), keep the primary code
            # 去掉编码（.UTF-8）与地区（_CN / -CN），保留主语言码
            code = value.replace("-", "_").split(".")[0].split("_")[0].lower()
            if code.startswith("zh"):
                return LANG_ZH
            return LANG_EN

    # Windows: GetUserDefaultUILanguage returns an LCID; primary language ID 0x04 = Chinese
    # Windows：GetUserDefaultUILanguage 返回 LCID；主语言 ID 0x04 = 中文
    if sys.platform == "win32":
        try:
            import ctypes

            lcid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            if (lcid & 0x3FF) == 0x04:
                return LANG_ZH
        except Exception:
            # pragma: no cover - depends on the platform
            # 取决于平台
            pass

    # Final fallback: the locale module
    # 最后兜底：locale 模块
    try:
        code, _encoding = locale.getlocale() or (None, None)
        if code and code.replace("-", "_").split("_")[0].lower().startswith("zh"):
            return LANG_ZH
    except Exception:
        # pragma: no cover - locale-dependent
        # 取决于语言环境
        pass

    return LANG_EN


def resolve_language(cli_lang: Optional[str]) -> str:
    """
    Resolve the effective language: CLI flag > system detection > English.

    解析生效语言：CLI 参数 > 系统检测 > 英文。
    """
    if cli_lang in (LANG_ZH, LANG_EN):
        return cli_lang
    return detect_system_language()


def configure(language: str) -> None:
    """
    Force the output language (used by the --lang CLI flag).

    强制设置输出语言（供 --lang CLI 参数使用）。
    """
    global _current
    _current = language if language in (LANG_ZH, LANG_EN) else LANG_EN


def current() -> str:
    """
    The currently active language.

    当前生效的语言。
    """
    return _current


def t(key: str, **fmt: object) -> str:
    """
    Translate a message key into the current language, then format it.

    把消息 key 翻译成当前语言，再按参数格式化。

    Unknown keys fall back to the key itself, so a typo degrades gracefully instead
    of crashing at runtime.
    未知 key 回退为 key 本身，拼写错误只会退化为原文而不会在运行时崩溃。
    """
    message = _MESSAGES.get(key)
    if message is None:
        return key
    text = message.zh if _current == LANG_ZH else message.en
    return text.format(**fmt) if fmt else text


_current = detect_system_language()
