"""
Reasoning-effort cache: discover, per model, which reasoning_effort levels the
upstream accepts, and surface them on /v1/models.

Discovery trick: the upstream (vLLM and friends) validates reasoning_effort as a
Literal enum. Sending the sentinel "__probe__" makes it fail with a 400 whose
error text enumerates every accepted value:

    Input should be 'none', 'low', 'medium' or 'high'

Validation happens before generation, so a probe costs no tokens. Levels are
model-specific (each backend instance validates its own set), hence one probe
per model; results are persisted to reasoning_cache.json and refreshed only when
the model list changes.

思考挡位缓存：逐模型探测上游接受哪些 reasoning_effort 挡位，并在 /v1/models 上透出。

探测原理：上游（vLLM 等）把 reasoning_effort 声明为 Literal 枚举校验，发送哨兵值
"__probe__" 会得到 400，错误文本里恰好枚举了全部可接受值：

    Input should be 'none', 'low', 'medium' or 'high'

校验发生在生成之前，因此一次探测的 token 成本为零。挡位与模型挂钩（每个后端
实例校验各自的集合），所以按模型逐一探测；结果持久化到 reasoning_cache.json，
仅在模型列表变化时刷新。
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("webui-proxy.reasoning")

# Sentinel value that can never be a real effort level; the upstream's Literal
# validation rejects it and names the accepted values in the error text.
#
# 哨兵值，绝不可能是真实挡位；上游的 Literal 校验会拒绝它，并在错误文本里
# 点名可接受的值。
PROBE_SENTINEL = "__probe__"

# Canonical effort order from fully-off to maximum thinking. Used to sort the
# emitted list and to derive defaults. Values unknown to this list (other
# upstreams may invent their own) still pass through, sorted to the end.
#
# 规范挡位顺序：从全关到最大思考。用于输出排序与默认值推导。不在该列表中的
# 未知挡位（其他上游可能自创）照样透传，只是排在末尾。
EFFORT_ORDER: List[str] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

_KNOWN_EFFORTS = frozenset(EFFORT_ORDER)

# "Input should be 'a', 'b' or 'c'" -> captures the run of quoted values.
# Handles both the "or"-joined tail and comma-separated middles, and tolerates
# the same text appearing (escaped) inside a JSON detail string.
#
# "Input should be 'a', 'b' or 'c'" -> 捕获连续的引号值序列。兼容末尾的 or
# 连接与中间的逗号分隔，也容忍同一段文本（带转义）出现在 JSON detail 里。
_INPUT_SHOULD_RE = re.compile(r"Input should be ((?:'[^']+'(?:\s*,\s*|\s+or\s+)?)+)")
_QUOTED_RE = re.compile(r"'([^']+)'")

CACHE_VERSION = 1


# --------------------------------------------------------------------------- #
# Error-text parsing
# 报错文本解析
# --------------------------------------------------------------------------- #
def extract_supported_efforts(error_text: str) -> List[str]:
    """
    Pull the accepted effort levels out of an upstream validation error.

    The upstream error text looks like (pydantic/vLLM style, possibly embedded
    and escaped inside a JSON "detail" string):

        1 validation error:
          {'type': 'literal_error', 'loc': ('body', 'reasoning_effort'),
           'msg': "Input should be 'none', 'low', 'medium' or 'high'",
           'input': '__probe__', ...}

    Two guards make this robust rather than just greedy:
    1. The text must mention reasoning_effort -- otherwise an "Input should be"
       clause belonging to a *different* field would be misparsed;
    2. At least one extracted value must be a known effort level -- otherwise a
       same-shape error about an unrelated enum slips through.

    Returns [] when nothing could be extracted.

    从上游校验错误中提取可接受的挡位列表。

    上游错误文本形如（pydantic/vLLM 风格，可能被转义嵌在 JSON detail 字符串里）：

        1 validation error:
          {'type': 'literal_error', 'loc': ('body', 'reasoning_effort'),
           'msg': "Input should be 'none', 'low', 'medium' or 'high'",
           'input': '__probe__', ...}

    两道防线让它比单纯贪心匹配更稳：
    1. 文本必须提到 reasoning_effort —— 否则别的字段的 "Input should be"
       子句会被误解析；
    2. 提取值中至少一个是已知挡位 —— 否则同构的无关枚举错误会漏进来。

    提取不到时返回 []。
    """
    if not error_text or "reasoning_effort" not in error_text:
        return []
    for match in _INPUT_SHOULD_RE.finditer(error_text):
        values = _QUOTED_RE.findall(match.group(1))
        if values and any(v in _KNOWN_EFFORTS for v in values):
            return values
    return []


def sort_efforts(efforts: List[str]) -> List[str]:
    """
    Sort effort levels into canonical order (none -> max); unknown values keep
    their original relative order at the end.

    把挡位按规范顺序（none -> max）排序；未知值按原相对顺序排在末尾。
    """
    known = [e for e in EFFORT_ORDER if e in efforts]
    seen = set(known)
    unknown = [e for e in efforts if e not in seen]
    return known + unknown


def derive_default_effort(efforts: List[str]) -> str:
    """
    Heuristic default: "medium" when supported, otherwise the median of the
    canonical ordering. The validation error carries no default information,
    so this is the best honest guess.

    启发式默认值：支持 "medium" 就用它，否则取规范顺序的中位数。校验错误里
    不包含默认值信息，这是最诚实的一个推断。
    """
    if "medium" in efforts:
        return "medium"
    ordered = [e for e in EFFORT_ORDER if e in efforts]
    if not ordered:
        return efforts[0]
    return ordered[len(ordered) // 2]


def build_reasoning_info(efforts: List[str]) -> Optional[Dict[str, Any]]:
    """
    Assemble the per-model "reasoning" object served on /v1/models.

    Field semantics (derived, since the probe only reveals the accepted set):
      - supported_efforts: exactly what the upstream validation accepts;
      - default_effort:    heuristic (medium / median), see derive_default_effort;
      - default_enabled:   true -- the field is accepted, so reasoning is on by
                           default as far as the upstream is concerned;
      - mandatory:         true when "none" is absent, i.e. thinking cannot be
                           turned off at all.

    Returns None when efforts is empty (nothing presentable).

    组装 /v1/models 上每个模型的 "reasoning" 对象。

    字段语义（均为推导值，探测只能揭示可接受集合）：
      - supported_efforts：上游校验接受的确切集合；
      - default_effort：启发式（medium / 中位数），见 derive_default_effort；
      - default_enabled：true —— 该字段被接受，就上游而言思考默认开启；
      - mandatory：缺少 "none" 时为 true，即完全无法关闭思考。

    efforts 为空（没有可呈现的信息）时返回 None。
    """
    if not efforts:
        return None
    return {
        "supported_efforts": sort_efforts(efforts),
        "default_effort": derive_default_effort(efforts),
        "default_enabled": True,
        "mandatory": "none" not in efforts,
    }


# --------------------------------------------------------------------------- #
# Persistent cache
# 持久化缓存
# --------------------------------------------------------------------------- #
class ReasoningCache:
    """
    model_id -> {supported_efforts, probed_at} persisted as JSON.

    An entry with an empty supported_efforts list is meaningful: it records
    "probed, but the upstream accepted the sentinel without validating" so the
    model is not re-probed on every startup. Models missing from the cache are
    the ones still to be probed.

    model_id -> {supported_efforts, probed_at}，以 JSON 持久化。

    supported_efforts 为空的条目也是有意义的：它记录"已探测，但上游未校验
    哨兵值"，避免每次启动都重探该模型。缓存中缺失的模型才是待探测的。
    """

    def __init__(self, path: Path):
        self._path = path
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        """
        Read the cache file (idempotent). Missing or corrupt files simply start
        an empty cache -- a re-probe is annoying, not fatal.

        读取缓存文件（幂等）。文件缺失或损坏都只是从空缓存开始——重探一遍
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
            logger.warning("reasoning cache unreadable (%s); starting empty", exc)
            return
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, dict):
            return
        for model_id, entry in models.items():
            if isinstance(entry, dict) and isinstance(entry.get("supported_efforts"), list):
                self._entries[str(model_id)] = {
                    "supported_efforts": [str(e) for e in entry["supported_efforts"]],
                    "probed_at": float(entry.get("probed_at") or 0.0),
                }

    def save(self) -> None:
        payload = {
            "version": CACHE_VERSION,
            "models": self._entries,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        The presentable "reasoning" object for a model, or None when there is
        nothing to show (never probed / probed but unprobeable).

        模型可呈现的 "reasoning" 对象；没有可呈现信息（未探测/探测但拿不到）
        时返回 None。
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return None
        return build_reasoning_info(entry.get("supported_efforts") or [])

    def update(self, model_id: str, efforts: List[str]) -> None:
        self._entries[model_id] = {
            "supported_efforts": list(efforts),
            "probed_at": time.time(),
        }

    def sync_with_models(self, model_ids: List[str], *, force: bool = False) -> List[str]:
        """
        Reconcile the cache with the current model list and return the model
        ids that still need probing.

        Entries for models that no longer exist upstream are dropped (a
        disappearing model frees its cache slot; a re-added model is probed
        again because its entry was removed). With force=True every current
        model is returned for a full re-probe.

        将缓存与当前模型列表对齐，返回仍需探测的模型 id 列表。

        上游已不存在的模型条目会被清除（模型消失即释放缓存槽位；重新上架的
        模型因条目已删会再次探测）。force=True 时返回全部当前模型做完整重探。
        """
        current = set(model_ids)
        stale = [m for m in self._entries if m not in current]
        for model_id in stale:
            del self._entries[model_id]
        if force:
            return list(model_ids)
        return [m for m in model_ids if m not in self._entries]

    def __len__(self) -> int:
        return len(self._entries)
