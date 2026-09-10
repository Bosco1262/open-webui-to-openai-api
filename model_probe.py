"""
Per-model probe cache: what the upstream engine *really* accepts, established by
asking it instead of trusting metadata.

Two kinds of facts are discovered and cached together, because they come out of
the same minimal requests:

1. Reasoning efforts (`reasoning_effort`). Sending a sentinel value that cannot be
   a real level makes the engine's schema validation fail with a 400 whose text
   enumerates the accepted levels. That enumeration is only the OUTER schema
   though: the model's own reasoning parser (Harmony for gpt-oss, Qwen's own
   parser, ...) validates a second time and rejects a subset of it. Every
   candidate is therefore verified with a real one-token request, and only a 200
   counts as "supported". Real example: Qwen3.8-27B advertises
   none/minimal/low/medium/high/xhigh/max but rejects minimal, high and max.

2. Capabilities and request parameters (vision, function calling, structured
   outputs, ...). Here the engine is the only source of truth: Open WebUI's
   `info.meta.capabilities` is a deployment-wide default template that is byte-for
   -byte identical for every model, so it says nothing about any single model.

The cache also stores an engine fingerprint (derived from the model list, so it
costs no request) and a negative cache with exponential backoff, so a model whose
probe keeps failing is not re-probed -- and does not make `/v1/models` wait -- on
every single request.

This module is deliberately free of HTTP: it parses upstream error text, builds
the probe payloads, and owns the cache. Sending the requests is the caller's job.

逐模型探测缓存：通过"问引擎"而不是"信元数据"来确定上游真正接受什么。

两类事实一起获取并一起缓存，因为它们来自同一批最小请求：

1. 思考挡位（`reasoning_effort`）。发送一个绝不可能是真实挡位的哨兵值，会让引擎的
   schema 校验以 400 失败，错误文本里枚举了可接受的挡位。但那只是**外层** schema：
   模型自带的推理解析器（gpt-oss 的 Harmony、Qwen 自己的解析器等）还会校验第二次，
   并拒绝其中的一个子集。因此每个候选值都会再用一次真实的单 token 请求实证，
   只有 200 才算"支持"。真实例子：Qwen3.8-27B 广告了
   none/minimal/low/medium/high/xhigh/max，但实际拒绝 minimal、high 与 max。

2. 能力与请求参数（视觉、函数调用、结构化输出等）。这里引擎是唯一的事实来源：
   Open WebUI 的 `info.meta.capabilities` 是部署级的默认模板，对每个模型都逐字节
   相同，因此说明不了任何单个模型的能力。

缓存还保存引擎指纹（从模型列表推导，零请求成本）与带指数退避的负缓存，
使"探测一直失败"的模型不会被每次请求反复重探，也不会让 `/v1/models` 每次都等待。

本模块刻意不碰 HTTP：它只解析上游报错文本、构造探测载荷、并持有缓存；
真正发请求是调用方的事。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("webui-proxy.probe")

# Sentinel value that can never be a real effort level; the upstream's schema
# validation rejects it and names the accepted values in the error text.
#
# 哨兵值，绝不可能是真实挡位；上游的 schema 校验会拒绝它，并在错误文本里点名
# 可接受的值。
PROBE_SENTINEL = "__probe__"

# Canonical effort order, from fully off to maximum thinking. Used to sort the
# emitted list and to run the fallback candidate sweep. Values unknown to this
# list (other upstreams may invent their own) still pass through, sorted last.
#
# 规范挡位顺序：从全关到最大思考。用于输出排序与兜底候选遍历。不在该列表中的
# 未知挡位（其他上游可能自创）照样透传，只是排在末尾。
EFFORT_ORDER: List[str] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

_KNOWN_EFFORTS = frozenset(EFFORT_ORDER)

# Capability keys established by probing. All of them are always emitted together
# so the object keeps a stable schema (the previous implementation echoed an
# upstream dict whose key set varied per model).
#
# 由探测确立的能力键。它们总是整体输出，保证对象结构稳定
# （旧实现透传上游字典，键集会随模型变化）。
CAPABILITY_KEYS: Tuple[str, ...] = (
    "vision",
    "function_calling",
    "reasoning",
    "structured_outputs",
)

# Request parameters verified by probing, named in OpenRouter's parameter
# namespace. A 400 is a definite "not supported"; a 200 only means "the engine did
# not reject it", which is what this list claims and no more.
#
# 通过探测验证的请求参数，采用 OpenRouter 的参数命名。400 是明确的"不支持"；
# 200 只意味着"引擎没有拒绝该参数"——本列表声称的也仅此而已。
PROBED_PARAMETERS: Tuple[str, ...] = (
    "tools",
    "tool_choice",
    "response_format",
    "logprobs",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "parallel_tool_calls",
)

# Probe outcomes. `status` describes the DATA (how conclusive the cached facts
# are); failure of a single attempt is expressed by `retry_after` + `last_error`,
# so a failed re-probe never throws away facts that were already established.
#
# 探测结果状态。`status` 描述的是**数据**（缓存事实有多确定）；单次尝试的失败由
# `retry_after` + `last_error` 表达，因此一次失败的重探不会丢掉已经确立的事实。
STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_UNPROBEABLE = "unprobeable"
STATUS_FAILED = "failed"

CACHE_VERSION = 2

# Exponential backoff for retrying a model whose probe failed (or is incomplete).
# Bounds bound how often a broken model can make the service re-probe.
#
# 探测失败（或未完成）后重试的指数退避上下界，约束坏模型被重探的频率。
BACKOFF_BASE_SECONDS = 60.0
BACKOFF_MAX_SECONDS = 6 * 3600.0

# A 1x1 transparent PNG: the cheapest possible multimodal input, used to establish
# whether the engine accepts image content at all.
#
# 1x1 透明 PNG：最便宜的多模态输入，用来确定引擎是否接受图片内容。
VISION_PROBE_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF/8Fj4AAAAAElFTkSuQmCC"
)

# --------------------------------------------------------------------------- #
# Error-text parsing
# 报错文本解析
# --------------------------------------------------------------------------- #
# pydantic/vLLM style: "Input should be 'none', 'low', 'medium' or 'high'"
_INPUT_SHOULD_RE = re.compile(r"Input should be ((?:'[^']+'(?:\s*,\s*|\s+or\s+)?)+)")
# Harmony style: "Supported values are: high, medium, low."
# Qwen style:     "Supported types are xhigh (default), medium, and low."
_SUPPORTED_LIST_RE = re.compile(
    r"supported\s+(?:values|types)\s+(?:are|is)\s*:?\s*([^.\n]+)", re.IGNORECASE
)
# A non-matching effort value echoed back by the engine, e.g.
# "reasoning_effort='max' is not supported by Harmony"
_REJECTED_VALUE_RE = re.compile(
    r"reasoning[_ ]effort\s*[=:]\s*'?([A-Za-z0-9_-]+)'?", re.IGNORECASE
)
# "(default)" marker, e.g. "Supported types are xhigh (default), medium, and low."
_DEFAULT_MARKER_RE = re.compile(r"([A-Za-z0-9_-]+)\s*\(\s*default\s*\)", re.IGNORECASE)
# "the default is xhigh" / "default: xhigh"
_DEFAULT_PHRASE_RE = re.compile(
    r"default\s*(?:is|:)\s*'?([A-Za-z0-9_-]+)'?", re.IGNORECASE
)
# A bare effort-looking token inside a comma/and separated list
_LIST_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,15}")

# The error text must be about the reasoning effort; otherwise an "Input should be"
# clause belonging to a different field would be misparsed.
#
# 报错文本必须与思考挡位有关，否则别的字段的 "Input should be" 子句会被误解析。
_EFFORT_TOPIC_RE = re.compile(r"reasoning[_ ]effort", re.IGNORECASE)


def _looks_like_effort(token: str) -> bool:
    """
    Whether a raw token is plausible as an effort level.

    判断一个原始 token 是否可能是挡位值。
    """
    lowered = token.strip().strip("'\"").lower()
    return bool(lowered) and lowered not in {"default", "and", "or", "the", "is", "are"}


def extract_effort_candidates(error_text: str) -> List[str]:
    """
    Pull every plausible accepted effort level out of an upstream error.

    Three real-world phrasings are understood:

        Input should be 'none', 'low', 'medium' or 'high'      (pydantic/vLLM)
        Supported values are: high, medium, low.               (Harmony / gpt-oss)
        Supported types are xhigh (default), medium, and low.  (Qwen)

    Guards: the text must mention the reasoning effort (otherwise a same-shaped
    error about an unrelated enum slips through), and at least one extracted value
    must be a known level.

    This enumeration is only the outer schema -- callers MUST verify each candidate
    with a real request before advertising it.

    Returns [] when nothing could be extracted; the caller then falls back to
    sweeping the full canonical list, so an unparseable upstream stays usable.

    从上游报错里提取所有可能的可接受挡位。

    理解三种真实措辞：

        Input should be 'none', 'low', 'medium' or 'high'      （pydantic/vLLM）
        Supported values are: high, medium, low.               （Harmony / gpt-oss）
        Supported types are xhigh (default), medium, and low.  （Qwen）

    两道防线：文本必须提到思考挡位（否则同构的无关枚举错误会漏进来），
    且提取值中至少有一个是已知挡位。

    这个枚举只是外层 schema —— 调用方**必须**逐个用真实请求实证后才能对外声明。

    提取不到时返回 []；调用方随后会兜底遍历完整规范列表，因此即使上游换了措辞
    也仍然可用。
    """
    if not error_text or not _EFFORT_TOPIC_RE.search(error_text):
        return []

    found: List[str] = []

    for match in _INPUT_SHOULD_RE.finditer(error_text):
        found.extend(re.findall(r"'([^']+)'", match.group(1)))

    for match in _SUPPORTED_LIST_RE.finditer(error_text):
        for token in _LIST_TOKEN_RE.findall(match.group(1)):
            if _looks_like_effort(token):
                found.append(token)

    # The value that was rejected is itself informative when the engine also names
    # what it does accept, e.g. "reasoning_effort='max' is not supported by Harmony.
    # Supported values are: high, medium, low." -- here only the list matters, so
    # the rejected value is used solely to decide whether the text is on topic.
    #
    # 被拒绝的那个值本身也有信息量（当引擎同时给出可接受列表时）。上面那条 Harmony
    # 报错里真正有用的是列表，因此被拒绝的值只用于确认文本切题。
    if not found:
        return []

    normalized: List[str] = []
    for raw in found:
        value = raw.strip().strip("'\"").lower()
        if not value or value == PROBE_SENTINEL:
            continue
        if value not in normalized:
            normalized.append(value)

    if not any(value in _KNOWN_EFFORTS for value in normalized):
        return []
    return sort_efforts(normalized)


def extract_default_effort(error_text: str) -> Optional[str]:
    """
    Pull the engine-declared default effort out of an upstream error, if it names
    one ("xhigh (default)" / "the default is xhigh").

    The previous implementation guessed this ("medium" or the median of the
    accepted list); the honest answer is None when the engine does not say.

    从上游报错里提取引擎自己声明的默认挡位（如 "xhigh (default)" / "default is xhigh"）。

    旧实现靠猜（"medium" 或可接受列表的中位数）；引擎没说时，诚实的答案是 None。
    """
    if not error_text:
        return None
    for pattern in (_DEFAULT_MARKER_RE, _DEFAULT_PHRASE_RE):
        match = pattern.search(error_text)
        if match:
            value = match.group(1).strip().strip("'\"").lower()
            if value and value != PROBE_SENTINEL:
                return value
    return None


def looks_like_effort_error(error_text: str) -> bool:
    """
    Whether an upstream failure is about the reasoning effort -- used to decide
    whether a live 400 should invalidate the cached effort list for that model.

    上游的失败是否与思考挡位有关——用于判断一次线上 400 是否应当作废该模型的
    挡位缓存。
    """
    return bool(error_text) and bool(_EFFORT_TOPIC_RE.search(error_text))


# pydantic loc tuples, e.g. "'loc': ('body', 'reasoning_effort')"
_LOC_RE = re.compile(r"'loc'\s*:\s*\(([^)]*)\)")
_LOC_TOKEN_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")

# Keyword -> parameter name, for engines that do not return a pydantic loc
# (e.g. vLLM's tool-call-parser complaint).
#
# 关键词 -> 参数名，用于不返回 pydantic loc 的引擎
# （例如 vLLM 关于 tool-call-parser 的报错）。
_PARAMETER_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("tool_choice", "tool_choice"),
    ("tool choice", "tool_choice"),
    ("tool-call-parser", "tools"),
    ("tool_call_parser", "tools"),
    ("tools", "tools"),
    ("function", "tools"),
    ("json_schema", "response_format"),
    ("response_format", "response_format"),
    ("guided", "response_format"),
    ("structured", "response_format"),
    ("logprob", "logprobs"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("stop", "stop"),
    ("seed", "seed"),
    ("parallel_tool_calls", "parallel_tool_calls"),
    ("reasoning", "reasoning_effort"),
)


def parameter_of_error(error_text: str) -> Optional[str]:
    """
    Attribute an upstream 400 to the request parameter that caused it.

    Prefers the pydantic `loc` (exact), falls back to keyword matching, and returns
    None when the error cannot be attributed -- the caller then probes the merged
    parameters one by one instead of guessing.

    把上游 400 归因到引发它的请求参数。

    优先使用 pydantic 的 `loc`（精确），其次关键词匹配；无法归因时返回 None ——
    调用方随后改为逐个探测参数，而不是猜。
    """
    if not error_text:
        return None
    match = _LOC_RE.search(error_text)
    if match:
        tokens = [
            first or second
            for first, second in _LOC_TOKEN_RE.findall(match.group(1))
        ]
        for token in reversed(tokens):
            if token in PROBED_PARAMETERS or token == "reasoning_effort":
                return token
    lowered = error_text.lower()
    for keyword, parameter in _PARAMETER_KEYWORDS:
        if keyword in lowered:
            return parameter
    return None


# --------------------------------------------------------------------------- #
# Probe payloads
# 探测载荷
# --------------------------------------------------------------------------- #
def effort_payload(model_id: str, effort: str) -> Dict[str, Any]:
    """
    A minimal completion request carrying one reasoning-effort value.

    max_tokens=1 bounds the worst case where an upstream ignores the field and
    actually generates.

    携带一个思考挡位值的最小补全请求。

    max_tokens=1 兜住最坏情况——上游完全忽略该字段并真的生成时，代价也只有 1 个 token。
    """
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
        "reasoning_effort": effort,
    }


def baseline_payload(model_id: str) -> Dict[str, Any]:
    """
    The same minimal request with `reasoning_effort` omitted, used to observe what
    the engine does by default (whether thinking is on).

    同样的最小请求，但不带 `reasoning_effort`，用于观察引擎的默认行为（思考是否默认开启）。
    """
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }


def parameter_payload(model_id: str, parameters: Sequence[str]) -> Dict[str, Any]:
    """
    One request carrying every parameter still under test, so a single 200 clears
    them all and a single 400 can be attributed (then retried without the offender).

    一个携带全部待测参数的请求：一次 200 就全部通过，一次 400 可被归因
    （剔除冒犯者后重试）。
    """
    payload: Dict[str, Any] = baseline_payload(model_id)
    for parameter in parameters:
        if parameter == "tools":
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        elif parameter == "tool_choice":
            payload["tool_choice"] = "auto"
        elif parameter == "response_format":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "probe",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    },
                },
            }
        elif parameter == "logprobs":
            payload["logprobs"] = True
            payload["top_logprobs"] = 1
        elif parameter == "temperature":
            payload["temperature"] = 0.7
        elif parameter == "top_p":
            payload["top_p"] = 0.9
        elif parameter == "stop":
            payload["stop"] = ["\n\n"]
        elif parameter == "seed":
            payload["seed"] = 42
        elif parameter == "parallel_tool_calls":
            payload["parallel_tool_calls"] = True
    return payload


def vision_payload(model_id: str) -> Dict[str, Any]:
    """
    The minimal request that asks the engine whether it accepts image content.

    询问引擎是否接受图片内容的最小请求。
    """
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ping"},
                    {"type": "image_url", "image_url": {"url": VISION_PROBE_IMAGE}},
                ],
            }
        ],
        "max_tokens": 1,
        "stream": False,
    }


def response_has_reasoning(body: str) -> Optional[bool]:
    """
    Read a chat-completion body and report whether the model produced thinking text.

    None means "cannot tell" (unparseable body, or a choice that carries neither
    content nor reasoning), which callers must not turn into a claim.

    读取补全响应体，判断模型是否产出了思考文本。

    None 表示"看不出来"（响应体无法解析，或该 choice 既没有 content 也没有
    reasoning），调用方不得把它变成结论。
    """
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    content = message.get("content")
    if reasoning:
        return True
    if content:
        return False
    return None


def sort_efforts(efforts: Iterable[str]) -> List[str]:
    """
    Sort effort levels into canonical order (none -> max); unknown values keep
    their original relative order at the end.

    把挡位按规范顺序（none -> max）排序；未知值按原相对顺序排在末尾。
    """
    given = list(efforts)
    known = [level for level in EFFORT_ORDER if level in given]
    seen = set(known)
    unknown = [level for level in given if level not in seen]
    return known + unknown


def build_reasoning_info(
    efforts: Sequence[str],
    default_effort: Optional[str] = None,
    default_enabled: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Assemble the per-model "reasoning" object served on /v1/models, in OpenRouter's
    shape: supported_efforts / default_effort / default_enabled / mandatory.

    Unknown facts are omitted rather than guessed (OpenRouter does the same: some of
    its models carry nothing but {"mandatory": false}).

    组装 /v1/models 上每个模型的 "reasoning" 对象，采用 OpenRouter 的形状：
    supported_efforts / default_effort / default_enabled / mandatory。

    拿不准的事实一律省略而不是猜（OpenRouter 也是如此：它有些模型只带
    {"mandatory": false}）。
    """
    if not efforts:
        return None
    info: Dict[str, Any] = {
        "supported_efforts": sort_efforts(efforts),
        "mandatory": "none" not in efforts,
    }
    if default_effort:
        info["default_effort"] = default_effort
    if default_enabled is not None:
        info["default_enabled"] = default_enabled
    return info


def build_architecture(vision: Optional[bool]) -> Optional[Dict[str, Any]]:
    """
    OpenRouter-shaped modality description derived from the vision probe. Only the
    three keys we can actually establish are emitted; `tokenizer`/`instruct_type`
    have no source here and are left out.

    由视觉探测推导出的 OpenRouter 形状模态描述。只输出三个确实能确立的键；
    `tokenizer`/`instruct_type` 在本项目里没有来源，故不输出。
    """
    if vision is None:
        return None
    inputs = ["text", "image"] if vision else ["text"]
    return {
        "modality": ("text+image->text" if vision else "text->text"),
        "input_modalities": inputs,
        "output_modalities": ["text"],
    }


def derive_reasoning_capability(
    supported_efforts: Optional[Sequence[str]], default_enabled: Optional[bool]
) -> Optional[bool]:
    """
    Whether the model is reasoning-capable: the engine accepts at least one level
    other than "off", or thinking was observed on a request without the field.

    Returns None when neither source is conclusive.

    模型是否具备思考能力：引擎接受至少一个"非关闭"挡位，或在未携带该字段的请求里
    观察到了思考内容。两个来源都得不出结论时返回 None。
    """
    if supported_efforts:
        if any(level != "none" for level in supported_efforts):
            return True
        return False
    if default_enabled is not None:
        return default_enabled
    return None


# --------------------------------------------------------------------------- #
# Cache
# 缓存
# --------------------------------------------------------------------------- #
@dataclass
class ModelProbe:
    """
    Everything established about one model, plus how to treat it next time.

    关于单个模型已确立的一切，以及下次该如何对待它。
    """

    fingerprint: str = ""
    probed_at: float = 0.0
    status: str = STATUS_FAILED
    attempts: int = 0
    retry_after: float = 0.0
    last_error: str = ""
    supported_efforts: List[str] = field(default_factory=list)
    efforts_verified: bool = False
    default_effort: Optional[str] = None
    default_enabled: Optional[bool] = None
    capabilities: Dict[str, bool] = field(default_factory=dict)
    supported_parameters: List[str] = field(default_factory=list)
    # Engine build string reported in chat responses; diagnostics only.
    # 聊天响应里上报的引擎构建串；仅供诊断。
    system_fingerprint: str = ""

    def has_facts(self) -> bool:
        """Whether anything presentable was ever established. / 是否已确立任何可呈现的事实。"""
        return self.status in (STATUS_OK, STATUS_PARTIAL, STATUS_UNPROBEABLE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "probed_at": self.probed_at,
            "status": self.status,
            "attempts": self.attempts,
            "retry_after": self.retry_after,
            "last_error": self.last_error,
            "supported_efforts": list(self.supported_efforts),
            "efforts_verified": self.efforts_verified,
            "default_effort": self.default_effort,
            "default_enabled": self.default_enabled,
            "capabilities": dict(self.capabilities),
            "supported_parameters": list(self.supported_parameters),
            "system_fingerprint": self.system_fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelProbe":
        capabilities = raw.get("capabilities")
        parameters = raw.get("supported_parameters")
        return cls(
            fingerprint=str(raw.get("fingerprint") or ""),
            probed_at=_as_float(raw.get("probed_at")),
            status=str(raw.get("status") or STATUS_FAILED),
            attempts=int(_as_float(raw.get("attempts"))),
            retry_after=_as_float(raw.get("retry_after")),
            last_error=str(raw.get("last_error") or ""),
            supported_efforts=[str(v) for v in (raw.get("supported_efforts") or [])],
            efforts_verified=bool(raw.get("efforts_verified")),
            default_effort=raw.get("default_effort") or None,
            default_enabled=raw.get("default_enabled")
            if isinstance(raw.get("default_enabled"), bool)
            else None,
            capabilities={
                str(key): bool(value)
                for key, value in (capabilities or {}).items()
                if isinstance(value, bool)
            },
            supported_parameters=[str(v) for v in (parameters or [])],
            system_fingerprint=str(raw.get("system_fingerprint") or ""),
        )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def backoff_seconds(attempts: int) -> float:
    """
    Exponential backoff for the Nth consecutive failure, capped.

    第 N 次连续失败后的指数退避，带上限。
    """
    if attempts <= 1:
        return BACKOFF_BASE_SECONDS
    return min(BACKOFF_BASE_SECONDS * (2 ** (attempts - 1)), BACKOFF_MAX_SECONDS)


class ModelProbeCache:
    """
    model_id -> ModelProbe, persisted as JSON next to the credential file.

    The cache is the contract between "what we verified once" and "what /v1/models
    advertises": a model is re-probed only when its engine fingerprint changes, when
    a previous probe left it inconclusive and its backoff has expired, or when the
    operator forces a refresh.

    model_id -> ModelProbe，以 JSON 持久化在凭证文件旁边。

    缓存是"我们实证过一次的结论"与"/v1/models 对外声明"之间的契约：只有当引擎指纹
    变化、上次探测结论不完整且退避已过期、或运维强制刷新时，才会重探该模型。
    """

    def __init__(self, path: Path):
        self._path = path
        self._entries: Dict[str, ModelProbe] = {}
        self._loaded = False

    def load(self) -> None:
        """
        Read the cache file (idempotent). A missing, corrupt or older-version file
        simply starts empty -- a re-probe is annoying, not fatal.

        读取缓存文件（幂等）。文件缺失、损坏或版本较旧时都从空缓存开始——重探一遍
        很烦，但不致命。
        """
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("probe cache unreadable (%s); starting empty", exc)
            return
        if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
            # Version 1 stored only effort lists and no fingerprint, so its entries
            # cannot be trusted to be complete; re-probe instead of migrating.
            #
            # 版本 1 只存了挡位列表、没有指纹，其条目无法保证完整；直接重探而不做迁移。
            if raw:
                logger.info(
                    "ignoring probe cache version %r (expected %s); will re-probe",
                    raw.get("version") if isinstance(raw, dict) else type(raw).__name__,
                    CACHE_VERSION,
                )
            return
        models = raw.get("models")
        if not isinstance(models, dict):
            return
        for model_id, entry in models.items():
            if isinstance(entry, dict):
                self._entries[str(model_id)] = ModelProbe.from_dict(entry)

    def save(self) -> None:
        payload = {
            "version": CACHE_VERSION,
            "models": {
                model_id: probe.to_dict() for model_id, probe in self._entries.items()
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #
    # Reads
    # 读取
    # ------------------------------------------------------------------ #
    def entry(self, model_id: str) -> Optional[ModelProbe]:
        """The raw cached entry, without interpretation. / 原始缓存条目，不做解释。"""
        return self._entries.get(model_id)

    def present(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        The model-level fields /v1/models should carry for this model:
        capabilities, supported_parameters and reasoning -- each only when it was
        actually established. None when nothing is known yet.

        /v1/models 应为本模型携带的模型级字段：capabilities、supported_parameters
        与 reasoning —— 各自只在确实确立后才出现。什么都还不知道时返回 None。
        """
        entry = self._entries.get(model_id)
        if entry is None or not entry.has_facts():
            return None

        presented: Dict[str, Any] = {}
        if entry.capabilities:
            # Only the keys that were actually established are emitted; a partial
            # probe therefore yields a partial object instead of a false "false".
            #
            # 只输出确实被确立的键；因此部分成功的探测产出的是部分对象，
            # 而不是一个虚假的 false。
            presented["capabilities"] = {
                key: bool(entry.capabilities[key])
                for key in CAPABILITY_KEYS
                if key in entry.capabilities
            }
        if entry.supported_parameters:
            presented["supported_parameters"] = list(entry.supported_parameters)

        reasoning = build_reasoning_info(
            entry.supported_efforts, entry.default_effort, entry.default_enabled
        )
        if reasoning is not None:
            presented["reasoning"] = reasoning
        if entry.capabilities:
            architecture = build_architecture(entry.capabilities.get("vision"))
            if architecture is not None:
                presented["architecture"] = architecture
        return presented or None

    def needs_probe(self, model_id: str, fingerprint: str, now: Optional[float] = None) -> bool:
        """
        Whether this model should be (re-)probed right now.

        该模型现在是否应当（重新）探测。
        """
        moment = time.time() if now is None else now
        entry = self._entries.get(model_id)
        if entry is None:
            return True
        if entry.fingerprint != fingerprint:
            return True
        if entry.status in (STATUS_FAILED, STATUS_PARTIAL):
            return moment >= entry.retry_after
        # STATUS_OK / STATUS_UNPROBEABLE are conclusive: nothing more to learn until
        # the engine fingerprint changes.
        #
        # STATUS_OK / STATUS_UNPROBEABLE 已是结论：在引擎指纹变化前没有更多可学的。
        return False

    # ------------------------------------------------------------------ #
    # Writes
    # 写入
    # ------------------------------------------------------------------ #
    def replace(self, model_id: str, probe: ModelProbe) -> None:
        self._entries[model_id] = probe

    def record_result(self, model_id: str, probe: ModelProbe) -> ModelProbe:
        """
        Store a completed probe attempt.

        A result that is still incomplete (`partial`) carries the failure streak
        over, so a model whose probe can never be fully resolved backs off instead of
        being re-probed on every request. A conclusive result resets the streak.

        保存一次完成的探测尝试。

        仍不完整（`partial`）的结果会继承失败计数，使"永远无法完全探清"的模型按
        退避重试，而不是每次请求都重探；结论性结果则清零计数。
        """
        previous = self._entries.get(model_id)
        same_engine = previous is not None and previous.fingerprint == probe.fingerprint
        if probe.status == STATUS_PARTIAL:
            probe.attempts = (previous.attempts if same_engine else 0) + 1
            probe.retry_after = time.time() + backoff_seconds(probe.attempts)
        else:
            probe.attempts = 0
            probe.retry_after = 0.0
        self._entries[model_id] = probe
        return probe

    def record_failure(self, model_id: str, fingerprint: str, error: str) -> ModelProbe:
        """
        Record a failed attempt without discarding facts established earlier.

        The entry keeps its previous status/data when it had any, and always gets a
        fresh backoff deadline so a broken model is not re-probed on every request.

        记录一次失败尝试，但不丢弃此前已确立的事实。

        若条目本来就有结论，则保留原状态/数据；无论如何都会写入新的退避截止时间，
        使坏模型不会被每次请求重探。
        """
        known = self._entries.get(model_id)
        attempts = (known.attempts if known and known.fingerprint == fingerprint else 0) + 1
        entry = known if known is not None else ModelProbe(fingerprint=fingerprint)
        if entry.fingerprint != fingerprint:
            # The engine changed under us: previous facts are void.
            # 引擎在我们脚下换了：此前的事实作废。
            entry = ModelProbe(fingerprint=fingerprint)
            attempts = 1
        entry.fingerprint = fingerprint
        entry.attempts = attempts
        entry.retry_after = time.time() + backoff_seconds(attempts)
        entry.last_error = error[:300]
        if not entry.has_facts():
            entry.status = STATUS_FAILED
        self._entries[model_id] = entry
        logger.debug(
            "probe of %s failed (attempt %s); retry in %.0fs: %s",
            model_id,
            attempts,
            backoff_seconds(attempts),
            entry.last_error,
        )
        return entry

    def invalidate_effort(self, model_id: str, effort: Optional[str]) -> bool:
        """
        Remove one effort level from a cached list (a live 400 just disproved it) and
        make the model eligible for a background re-probe.

        Returns whether anything changed.

        从缓存列表里移除某个挡位（线上 400 刚刚证伪了它），并让该模型可以被后台重探。
        返回是否有改动。
        """
        entry = self._entries.get(model_id)
        if entry is None or not effort:
            return False
        if effort not in entry.supported_efforts:
            return False
        entry.supported_efforts = [v for v in entry.supported_efforts if v != effort]
        if entry.default_effort == effort:
            # The default cannot be a level the engine just rejected.
            # 默认值不可能是一个刚被引擎拒绝的挡位。
            entry.default_effort = None
        entry.status = STATUS_PARTIAL
        entry.retry_after = 0.0
        entry.last_error = f"upstream rejected reasoning_effort={effort!r}"
        logger.info(
            "model %s: dropping reasoning_effort=%r after an upstream 400; re-probing",
            model_id,
            effort,
        )
        return True

    def sync_with_models(
        self,
        models: Sequence[Tuple[str, str]],
        *,
        force: bool = False,
        now: Optional[float] = None,
    ) -> List[str]:
        """
        Reconcile the cache with the current model list and return the ids that still
        need probing.

        `models` is a sequence of (model_id, engine fingerprint). Entries for models
        that no longer exist upstream are dropped (a disappearing model frees its
        slot; a re-added model is probed again because its entry was removed).

        With force=True every current model is returned for a full re-probe.

        将缓存与当前模型列表对齐，返回仍需探测的模型 id。

        `models` 是 (模型 id, 引擎指纹) 序列。上游已不存在的模型条目会被清除
        （模型消失即释放槽位；重新上架的模型因条目已删会再次探测）。
        force=True 时返回全部当前模型做完整重探。
        """
        current = {model_id for model_id, _ in models}
        for model_id in [key for key in self._entries if key not in current]:
            del self._entries[model_id]
        if force:
            return [model_id for model_id, _ in models]
        return [
            model_id
            for model_id, fingerprint in models
            if self.needs_probe(model_id, fingerprint, now)
        ]

    def __len__(self) -> int:
        return len(self._entries)
