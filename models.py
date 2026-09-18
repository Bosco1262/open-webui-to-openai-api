"""
Model-list normalization: upstream model objects in, OpenAI model objects out.

Everything here is pure: no HTTP, no cache, no configuration. It is the single place
that decides which upstream entries survive normalization and how the engine
fingerprint is derived, so the probe-refresh path and the /v1/models path can never
drift apart on those two definitions.

`app.py` re-exports these names, so routes and tests keep addressing them through the
app module.


模型列表规范化：把上游模型对象收敛为 OpenAI 模型对象。

这里的一切都是纯函数：不发 HTTP、不碰缓存、不读配置。这里是唯一决定"哪些上游条目
能通过规范化"与"引擎指纹如何推导"的地方，使探测刷新路径与 /v1/models 路径在这两个
定义上永远不会口径漂移。

`app.py` 重新导出这些名字，因此路由与测试继续通过 app 模块访问它们。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Quantization tokens recognizable in model ids: NVFP4, FP8, FP16, INT8, GPTQ, AWQ, ...
#
# `Q<n>` alone is deliberately NOT matched: ids like "q4-summary" are not quantized
# models, and the token is far too generic to claim one. Requiring at least one
# underscore segment after it (Q4_K_M, Q5_0) keeps the real llama.cpp-style names and
# drops the standalone form (R4).
#
#
# 模型名中可识别的量化标识：NVFP4、FP8、FP16、INT8、GPTQ、AWQ 等
#
# 刻意**不**匹配单独的 `Q<n>`：像 "q4-summary" 这类 id 并不是量化模型，而这个 token
# 本身也过于通用，不足以据此下断言。要求其后至少带一个下划线段（Q4_K_M、Q5_0），
# 既保留 llama.cpp 风格的真实名称，又排除独立词形式（R4）。
_QUANT_PATTERN = re.compile(
    r"\b(NVFP4|FP4|FP8|FP16|INT8|INT4|GPTQ(?:-?[0-9]+BIT)?|AWQ|GGUF|Q[0-9](?:_[A-Z0-9]+)+)\b",
    re.IGNORECASE,
)


def _parse_timestamp(value: Any) -> Optional[int]:
    """
    Parse a creation timestamp: epoch seconds (int/float, or a numeric string)
    directly; an ISO-8601 string via fromisoformat; anything else is not a
    timestamp. Returns None for values that cannot be interpreted -- the caller
    falls through to the next candidate.

    解析创建时间戳：epoch 秒（int/float，或纯数字字符串）直接返回；ISO-8601
    字符串用 fromisoformat 解析；其余一律不算时间戳。无法解释的值返回 None，
    由调用方落到下一个候选。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            # Naive timestamps are read as UTC: the upstream omitted the zone, and
            # interpreting them in local time would make the same model's `created`
            # depend on where the proxy happens to run.
            #
            # 无时区的时间戳按 UTC 解释：上游省略了时区，若按本地时间解释，
            # 同一模型的 created 会随代理所在机器的时区漂移。
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    return None


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
    `x_open_webui_deviations`. Reporting the template once, as an instance-level fact
    (under the envelope's `x_open_webui`), is honest; repeating it inside each model's
    `capabilities` would claim something about the model that is not true -- the same
    template was also handed to DeepSeek-V4-Flash, which then answered an image with
    "is not a multimodal model".


    上游每个上报能力的模型都一致同意的那些键——这部分共同值就是 Open WebUI 合并进
    每个模型的"默认模型元数据"模板。

    各模型不一致（或只有部分模型上报）的键不纳入模板；某个模型对这些键自己的取值，
    作为"偏离"放在该模型的 `x_open_webui_deviations` 里。把模板作为实例级事实
    （放在信封的 `x_open_webui` 下）输出一次是诚实的；重复放进每个模型的
    `capabilities` 则是在声称模型具备它并不具备的能力——同一份模板也发给了
    DeepSeek-V4-Flash，而它对图片的回答是 "is not a multimodal model"。
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
    # OpenAI layer is the serving engine's start time, not the model's. A candidate
    # that cannot be parsed (an ISO-8601 string, garbage) falls through to the next
    # one instead of zeroing the whole field: a usable integer further down the
    # chain must not be swallowed by one unparseable value at the top.
    #
    # info.created_at 才是模型真实创建时间；OpenAI 层的 created 是推理引擎
    # 的启动时间，并非模型本身的。解析不了的候选（ISO-8601 字符串、垃圾值）
    # 落到下一个候选而不是整个归零：链下游可用的整数不能被链顶一个解析
    # 不了的值吞掉。
    created = 0
    for candidate in (info.get("created_at"), raw.get("created"), raw.get("created_at")):
        parsed = _parse_timestamp(candidate)
        if parsed is not None:
            created = parsed
            break

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
        # Open WebUI hands the deployment-wide template to every model, so a model that
        # deviates from it is worth keeping -- but under its own key, never inside
        # `capabilities` (which holds probed facts only) and never as `x_open_webui`
        # (which names the instance-level metadata in the /v1/models envelope, R9).
        #
        # Open WebUI 把同一份部署级模板发给每个模型，因此偏离模板的模型值得保留——
        # 但用自己的键，绝不放进只承载实证事实的 `capabilities`，也不叫
        # `x_open_webui`（后者在 /v1/models 信封里指实例级元信息，R9）。
        model["x_open_webui_deviations"] = {"capabilities": deviation}

    return model


def _model_summaries(
    raw_models: List[Any], shared_capabilities: Optional[Dict[str, bool]] = None
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Normalize a raw model list into (normalized models, (model_id, engine
    fingerprint) summaries). The single place that decides which entries survive
    normalization and how the engine fingerprint is derived, so the refresh path and
    the /v1/models path can never drift apart on those two definitions.

    把原始模型列表规范化为 (规范化模型, (模型 id, 引擎指纹) 摘要)。这里是唯一
    决定"哪些条目通过规范化"与"引擎指纹如何推导"的地方，使刷新路径与
    /v1/models 路径在这两个定义上永远不会口径漂移。
    """
    models: List[Dict[str, Any]] = []
    summaries: List[Tuple[str, str]] = []
    for raw in raw_models:
        model = normalize_model(raw, shared_capabilities)
        if not model:
            continue
        models.append(model)
        summaries.append((model["id"], _model_fingerprint(raw, model["id"])))
    return models, summaries
