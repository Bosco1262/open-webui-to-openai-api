"""
Per-request correlation id (U-7): one id per incoming request, carried through every
log line the request produces and echoed back in the response headers.

Without it, a client reporting "the upstream errored" cannot be tied to the log line
that explains what happened -- the operator's only option was to guess from timestamps.

The id lives in a contextvar rather than being threaded through every function
signature: the logging formatter and the error builders read it at the point of use,
which is exactly the set of places that need it.

每个进入请求的关联 id（U-7）：随该请求产生的每一行日志携带，并在响应头里回显。

没有它时，客户端报告的"上游报错了"无法与解释原因的那行日志对上——运维只能靠时间戳猜。

id 放在 contextvar 而不是逐层传参：日志格式化器与错误消息构造在使用点直接读取，
而它们恰好就是需要这个值的全部位置。
"""

from __future__ import annotations

import contextvars
import logging
import re
import uuid
from typing import Optional

import lang

# Longest accepted client-supplied id. Long enough for a UUID with a prefix, short
# enough that a hostile client cannot push log lines around.
#
# 接受的最长客户端 id。够放下带前缀的 UUID，又不至于让恶意客户端把日志行顶乱。
REQUEST_ID_MAX_LENGTH = 64

# Only characters that are safe in a header value and readable in a log line. Anything
# else (newlines above all: classic log forging) is dropped instead of escaped, so a
# forged id can never fake a second log entry.
#
# 仅保留在响应头里安全、在日志行里可读的字符。其余（首当其冲是换行——经典的日志伪造）
# 一律丢弃而不是转义，使伪造的 id 永远无法造出第二行日志。
_SAFE_REQUEST_ID = re.compile(r"[^A-Za-z0-9._:-]")

# The id of the request currently being handled; "" outside a request (startup, CLI).
# Each request runs in its own task, so setting without resetting cannot leak into
# another request -- the next request overwrites the value in its own context.
#
# 当前正在处理的请求 id；请求之外（启动、CLI）为空串。每个请求在自己的任务里运行，
# 因此只设置不重置不会泄漏到别的请求——下一个请求会在自己的上下文里覆盖该值。
_current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "webui_proxy_request_id", default=""
)


def sanitize_request_id(raw: Optional[str]) -> str:
    """
    Reduce a client-supplied id to something safe to log and echo back; "" when
    nothing usable is left (the caller then generates a fresh id).

    把客户端提供的 id 收敛为可安全记日志、可安全回显的形式；没有可用内容时返回 ""
    （调用方随后会生成一个新 id）。
    """
    if not raw:
        return ""
    cleaned = _SAFE_REQUEST_ID.sub("", raw.strip())[:REQUEST_ID_MAX_LENGTH]
    return cleaned


def new_request_id(raw: Optional[str] = None) -> str:
    """
    The id for a request: the client's own (sanitized) if it supplied a usable one,
    a fresh UUID otherwise.

    请求的 id：客户端提供了可用值时用它的（已清洗），否则生成新的 UUID。
    """
    return sanitize_request_id(raw) or uuid.uuid4().hex


def bind_request_id(request_id: str) -> None:
    """
    Make `request_id` the current id for this request's context.

    把 `request_id` 设为当前请求上下文中的 id。
    """
    _current_request_id.set(request_id)


def current_request_id() -> str:
    """
    The current request's id; "" outside a request context.

    当前请求的 id；请求上下文之外为空串。
    """
    return _current_request_id.get()


def request_id_suffix() -> str:
    """
    " (request id: ...)" for a client-facing message, or "" when there is no request
    context (background work, CLI output).

    客户端可见消息末尾的 " (request id: ...)"；没有请求上下文时（后台任务、CLI 输出）
    返回空串。
    """
    request_id = current_request_id()
    if not request_id:
        return ""
    return lang.t("request_id_suffix", request_id=request_id)


class RequestIdFormatter(logging.Formatter):
    """
    A formatter that guarantees `record.request_id` exists, so the log format can
    carry the correlation id without every call site having to pass it.

    保证 `record.request_id` 一定存在的格式化器，使日志格式可以携带关联 id，
    而不要求每个调用点都传它。
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - stdlib name
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id() or "-"
        return super().format(record)
