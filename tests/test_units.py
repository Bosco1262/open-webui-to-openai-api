"""
Pure-logic unit tests: no Playwright, no network required.

Run from the project root:
    python tests/test_units.py

纯逻辑单元测试，不依赖 Playwright，也不需要联网。

运行方式（项目根目录）：
    python tests/test_units.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app as proxy  # noqa: E402
import config as config_module  # noqa: E402
import lang as lang_module  # noqa: E402
import model_probe as mprobe  # noqa: E402
import session_store as store  # noqa: E402

PASSED: list = []
FAILED: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [ok]   {name}")
    else:
        FAILED.append(f"{name}{(' -> ' + detail) if detail else ''}")
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
def test_session_roundtrip() -> None:
    print("\n--- Session serialization / Session 序列化 ---")
    session = store.Session(
        authorization="Bearer abc.def.ghi",
        cookie="a=1; b=2",
        user_agent="UA/1.0",
        captured_at=time.time(),
        base_url="http://x",
    )
    restored = store.Session.from_dict(session.to_dict())
    check("往返不丢字段", restored == session, f"{restored}")

    legacy = store.Session.from_dict(
        {"Authorization": "Bearer zzz", "Cookie": "c=3", "User-Agent": "old-agent"}
    )
    check("兼容旧版键名", legacy.authorization == "Bearer zzz" and legacy.cookie == "c=3", str(legacy))
    check("旧版数据缺少 captured_at 不报错", legacy.captured_at == 0.0, str(legacy.captured_at))

    mixed = store.Session.from_dict({"AUTHORIZATION": "Bearer mix", "cookie": "k=v"})
    check("键名大小写不敏感", mixed.authorization == "Bearer mix", str(mixed))

    empty = store.Session.from_dict({})
    check("空凭证判定为不可用", not empty.is_usable())

    headers = session.to_headers()
    check(
        "生成的上游请求头完整",
        headers["Authorization"] == "Bearer abc.def.ghi"
        and headers["Cookie"] == "a=1; b=2"
        and headers["User-Agent"] == "UA/1.0",
        str(headers),
    )
    check("日志摘要已脱敏", "abc.def.ghi" not in session.describe(), session.describe())


def test_session_file() -> None:
    print("\n--- Session file read/write / 凭证文件读写 ---")
    tmp = Path(tempfile.mkdtemp())
    settings = config_module.load_settings()
    settings = type(settings)(**{**settings.__dict__, "session_file": tmp / "session.json"})

    check("初始状态：文件不存在", not store.session_exists(settings))
    try:
        store.load_session(settings)
        check("缺少文件时抛 SessionMissing", False)
    except store.SessionMissing:
        check("缺少文件时抛 SessionMissing", True)

    (tmp / "session.json").write_text("{ not json", encoding="utf-8")
    try:
        store.load_session(settings)
        check("坏 JSON 抛 SessionInvalid", False)
    except store.SessionInvalid:
        check("坏 JSON 抛 SessionInvalid", True)

    (tmp / "session.json").write_text('{"authorization": "Bearer ok"}', encoding="utf-8")
    loaded = store.load_session(settings)
    check("正常读取", loaded.authorization == "Bearer ok", str(loaded))

    (tmp / "session.json").write_text('{"cookie": ""}', encoding="utf-8")
    try:
        store.load_session(settings)
        check("凭证为空抛 SessionInvalid", False)
    except store.SessionInvalid:
        check("凭证为空抛 SessionInvalid", True)


def test_login_signal() -> None:
    print("\n--- Login signal detection / 登录信号判定 ---")
    base = "https://webui.example.com"
    api = f"{base}/api"

    ok_bearer = {"authorization": "Bearer eyJhbGciOi...", "cookie": ""}
    check(
        "Bearer Token 命中 /api/models",
        store.is_login_signal(f"{base}/api/models", ok_bearer, api),
    )
    check(
        "请求头大小写不敏感",
        store.is_login_signal(f"{base}/api/models", {"Authorization": "Bearer x.y"}, api),
    )
    check(
        "非上游域名不判定",
        not store.is_login_signal("https://evil.example.com/api/models", ok_bearer, api),
    )
    check(
        "空 Bearer 不算登录",
        not store.is_login_signal(f"{base}/api/models", {"authorization": "Bearer "}, api),
    )
    check(
        "匿名 Cookie 访问 /api/config 不算登录",
        not store.is_login_signal(f"{base}/api/config", {"cookie": "theme=dark"}, api),
    )
    check(
        "匿名 Cookie 但命中需登录接口算登录",
        store.is_login_signal(f"{base}/api/v1/models", {"cookie": "theme=dark"}, api),
    )
    check(
        "完全没有身份信息不算登录",
        not store.is_login_signal(f"{base}/api/models", {"user-agent": "UA"}, api),
    )


def test_model_normalization() -> None:
    print("\n--- Model list normalization / 模型列表规范化 ---")
    check("裸字符串", proxy.normalize_model("gpt-4o") == {
        "id": "gpt-4o", "object": "model", "created": 0, "owned_by": "openai"
    })
    full = proxy.normalize_model(
        {
            "id": "llama3:latest",
            "name": "llama3",
            "object": "model",
            "created": 1700000000,
            "owned_by": "ollama",
            "info": {"meta": {}},
            "params": {},
        }
    )
    check("标准字段保留", full["id"] == "llama3:latest" and full["owned_by"] == "ollama", str(full))
    check("上游的显示名被透出", full.get("name") == "llama3", str(full))
    check(
        "私有字段被剔除",
        set(full.keys()) == {"id", "name", "object", "created", "owned_by"},
        str(full),
    )

    legacy = proxy.normalize_model({"id": "old-model"})
    check(
        "补齐缺失字段",
        legacy == {"id": "old-model", "object": "model", "created": 0, "owned_by": "openai"},
        str(legacy),
    )
    check("非法 created 归零", proxy.normalize_model({"id": "x", "created": "abc"})["created"] == 0)
    check(
        "缺 id 时回退使用 name",
        proxy.normalize_model({"name": "only-name"})["id"] == "only-name",
    )
    check("非对象类型返回 None", proxy.normalize_model(12345) is None)

    shared_capabilities = {"vision": True, "web_search": True}
    extended = proxy.normalize_model(
        {
            "id": "GLM-5.2-NVFP4",
            "created": 1788788099,
            "owned_by": "openai",
            "max_model_len": 131072,
            "info": {
                "user_id": "secret-user",
                "created_at": 1779326071,
                "meta": {
                    "description": "A test model",
                    "capabilities": dict(shared_capabilities, usage=False),
                },
                "access_grants": [{"principal_id": "*"}],
            },
            "urlIdx": 3,
            "permission": [],
            "openai": {"owned_by": "vllm", "max_model_len": 999},
        },
        shared_capabilities,
    )
    check("created 优先取 info.created_at", extended.get("created") == 1779326071, str(extended))
    check("owned_by 优先取内层引擎归属", extended.get("owned_by") == "vllm", str(extended))
    check(
        "通用模板上下文字段 + 兼容别名",
        extended.get("max_context_length") == 131072
        and extended.get("context_length") == 131072
        and extended.get("max_model_len") == 131072,
        str(extended),
    )
    check("quantization 从模型名解析", extended.get("quantization") == "NVFP4", str(extended))
    check("description 透出", extended.get("description") == "A test model", str(extended))
    check(
        "偏离实例模板的能力放进 x_open_webui，而不是 capabilities",
        extended.get("capabilities") is None
        and extended.get("x_open_webui") == {"capabilities": {"usage": False}},
        str(extended),
    )
    matched = proxy.normalize_model(
        {
            "id": "in-template",
            "info": {"meta": {"capabilities": dict(shared_capabilities)}},
        },
        shared_capabilities,
    )
    check(
        "与实例模板一致时不重复输出",
        "x_open_webui" not in matched and "capabilities" not in matched,
        str(matched),
    )
    no_template = proxy.normalize_model(
        {"id": "no-template", "info": {"meta": {"capabilities": {"vision": True}}}}, None
    )
    check(
        "模板不可用时整份能力作为偏离保留",
        no_template.get("x_open_webui") == {"capabilities": {"vision": True}},
        str(no_template),
    )
    check(
        "无量化标识时不输出 quantization",
        proxy.normalize_model({"id": "plain-model", "max_model_len": 100}).get("quantization") is None,
    )
    check(
        "私有字段不透出",
        "user_id" not in extended
        and "access_grants" not in extended
        and "urlIdx" not in extended
        and "permission" not in extended,
        str(extended),
    )


def test_model_fingerprint() -> None:
    print("\n--- Engine fingerprint / 引擎指纹 ---")
    base = {
        "id": "m",
        "max_model_len": 8192,
        "openai": {"root": "org/model", "owned_by": "vllm"},
        "info": {"base_model_id": None, "updated_at": 100},
    }
    fingerprint = proxy._model_fingerprint(base, "m")
    check("同一模型指纹稳定", proxy._model_fingerprint(dict(base), "m") == fingerprint)

    # The top-level `created` is the vLLM response build time: it changes on every
    # fetch and must not invalidate the cache.
    #
    # 顶层 created 是 vLLM 的响应构建时间：每次拉取都变，绝不能让它作废缓存。
    check(
        "顶层 created 不参与指纹",
        proxy._model_fingerprint(dict(base, created=1), "m") == fingerprint
        and proxy._model_fingerprint(dict(base, created=2), "m") == fingerprint,
    )
    check(
        "引擎换路径/上下文/配置时指纹变化",
        proxy._model_fingerprint({**base, "openai": {"root": "other/model"}}, "m") != fingerprint
        and proxy._model_fingerprint({**base, "max_model_len": 4096}, "m") != fingerprint
        and proxy._model_fingerprint({**base, "info": {"updated_at": 200}}, "m") != fingerprint,
    )


def test_shared_default_capabilities() -> None:
    print("\n--- Instance-level default capability template / 实例级默认能力模板 ---")
    template = {"vision": True, "web_search": True}
    same = [{"info": {"meta": {"capabilities": dict(template)}}} for _ in range(3)]
    check(
        "一致时整体归并为实例级模板",
        proxy._shared_default_capabilities(same) == template,
        str(proxy._shared_default_capabilities(same)),
    )
    differing = same + [{"info": {"meta": {"capabilities": {"vision": False, "web_search": True}}}}]
    check(
        "不一致的键被排除，其余仍归并",
        proxy._shared_default_capabilities(differing) == {"web_search": True},
        str(proxy._shared_default_capabilities(differing)),
    )
    partially_reported = same + [
        {"info": {"meta": {"capabilities": {"vision": True, "web_search": True, "usage": True}}}}
    ]
    check(
        "只有部分模型上报的键不算模板",
        proxy._shared_default_capabilities(partially_reported) == template,
        str(proxy._shared_default_capabilities(partially_reported)),
    )
    check(
        "没有模型上报时返回 None",
        proxy._shared_default_capabilities([{"id": "no-info"}, {"id": "also-none"}]) is None,
    )


def test_model_list_extraction() -> None:
    print("\n--- Model list shape compatibility / 模型列表结构兼容 ---")
    check("data 包裹", len(proxy.extract_model_list({"data": [{"id": "a"}, {"id": "b"}]})) == 2)
    check("items 包裹", len(proxy.extract_model_list({"items": [{"id": "a"}]})) == 1)
    check("裸列表", len(proxy.extract_model_list([{"id": "a"}])) == 1)
    check("未知结构返回空", proxy.extract_model_list({"foo": "bar"}) == [])
    check("字符串返回空", proxy.extract_model_list("nope") == [])


def test_config() -> None:
    print("\n--- Config parsing / 配置解析 ---")
    env_backup = dict(os.environ)
    try:
        os.environ["OPEN_WEBUI_BASE_URL"] = "https://webui.example.com/"
        os.environ["PROXY_PORT"] = "not-a-number"
        os.environ["MODEL_ALIASES"] = '{"gpt-4o": "gpt-4o-mini"}'
        os.environ["UPSTREAM_API_STYLE"] = "weird"
        os.environ["SESSION_FILE"] = "~/creds.json"
        settings = config_module.load_settings()
        check("去掉结尾斜杠", settings.open_webui_base_url == "https://webui.example.com")
        check("非法整数回退默认值", settings.proxy_port == 8000, str(settings.proxy_port))
        check("解析模型别名", settings.resolve_model("gpt-4o") == "gpt-4o-mini")
        check("未配置的模型原样返回", settings.resolve_model("llama3") == "llama3")
        check("非法风格回退 auto", settings.upstream_api_style == "auto")
        check("展开 ~ 路径", "~" not in str(settings.session_file), str(settings.session_file))

        check("auto 的候选前缀", settings.prefix_candidates() == ["/api/v1", "/api"])
        v1 = type(settings)(**{**settings.__dict__, "upstream_api_style": "v1"})
        check("v1 的候选前缀", v1.prefix_candidates() == ["/api/v1"])
        legacy = type(settings)(**{**settings.__dict__, "upstream_api_style": "legacy"})
        check("legacy 的候选前缀", legacy.prefix_candidates() == ["/api"])

        check(
            "拼接上游 URL",
            settings.upstream_url("/api/v1", "chat/completions")
            == "https://webui.example.com/api/v1/chat/completions",
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_probe_settings() -> None:
    print("\n--- Probe settings / 探测配置 ---")
    env_backup = dict(os.environ)
    try:
        for name in (
            "MODEL_PROBE_CACHE_FILE",
            "MODEL_PROBE_CONCURRENCY",
            "MODEL_PROBE_TIMEOUT",
            "MODEL_PROBE_WAIT",
            "EXPOSE_INSTANCE_META",
        ):
            os.environ.pop(name, None)

        settings = config_module.load_settings()
        check(
            "默认缓存文件名与探测参数",
            settings.model_probe_cache_file.name == "model_probe_cache.json"
            and settings.model_probe_concurrency == 4
            and settings.model_probe_timeout == 30.0
            and settings.model_probe_wait == 5.0,
            f"{settings.model_probe_cache_file} {settings.model_probe_concurrency} "
            f"{settings.model_probe_timeout} {settings.model_probe_wait}",
        )

        os.environ["MODEL_PROBE_CACHE_FILE"] = "custom-cache.json"
        os.environ["MODEL_PROBE_CONCURRENCY"] = "2"
        os.environ["MODEL_PROBE_WAIT"] = "0"
        settings = config_module.load_settings()
        check(
            "环境变量生效",
            settings.model_probe_cache_file.name == "custom-cache.json"
            and settings.model_probe_concurrency == 2
            and settings.model_probe_wait == 0.0,
            f"{settings.model_probe_cache_file} {settings.model_probe_concurrency} {settings.model_probe_wait}",
        )

        check("实例元信息默认开启", settings.expose_instance_meta is True)
        os.environ["EXPOSE_INSTANCE_META"] = "false"
        check("实例元信息可关闭", config_module.load_settings().expose_instance_meta is False)

        # The old REASONING_* names are gone: setting one must change nothing.
        # 旧的 REASONING_* 名字已彻底移除：设置它们不应产生任何影响。
        os.environ.pop("MODEL_PROBE_CACHE_FILE", None)
        os.environ.pop("MODEL_PROBE_WAIT", None)
        os.environ["REASONING_CACHE_FILE"] = "should-be-ignored.json"
        os.environ["REASONING_PROBE_WAIT"] = "99"
        settings = config_module.load_settings()
        check(
            "旧的 REASONING_* 变量不再被读取",
            settings.model_probe_cache_file.name == "model_probe_cache.json"
            and settings.model_probe_wait == 5.0,
            f"{settings.model_probe_cache_file} {settings.model_probe_wait}",
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_error_shape() -> None:
    print("\n--- Error response shape / 错误响应结构 ---")
    import json

    resp = proxy.openai_error("bad request", 400, code="invalid_api_key")
    body = resp.body.decode("utf-8")
    parsed = json.loads(body)
    check("状态码正确", resp.status_code == 400, str(resp.status_code))
    check(
        "OpenAI 风格错误体",
        parsed.get("error", {}).get("code") == "invalid_api_key"
        and parsed.get("error", {}).get("message") == "bad request",
        body,
    )
    check("错误体不含 detail", "detail" not in parsed, body)


def test_credentials_are_valid() -> None:
    print("\n--- Upstream validation of captured credentials / 抓取凭证的上游校验 ---")
    import asyncio

    from mock_openwebui import VALID_TOKEN, MockOpenWebUI

    server = MockOpenWebUI()
    server.start()
    try:
        settings = config_module.load_settings()
        settings = type(settings)(**{**settings.__dict__, "open_webui_base_url": server.base_url})

        ok = store.Session(authorization=f"Bearer {VALID_TOKEN}")
        check(
            "有效凭证通过上游校验",
            asyncio.run(store._credentials_are_valid(settings, ok)),
        )

        expired = store.Session(authorization="Bearer stale-expired-token")
        check(
            "过期/无效 Token 被拒绝",
            not asyncio.run(store._credentials_are_valid(settings, expired)),
        )

        empty = store.Session()
        check(
            "空凭证直接判无效",
            not asyncio.run(store._credentials_are_valid(settings, empty)),
        )
    finally:
        server.stop()


def test_language_detection() -> None:
    print("\n--- Language detection and priority / 语言检测与优先级 ---")
    env_backup = dict(os.environ)
    try:
        # CLI flag wins over everything
        # CLI 参数优先于一切
        check("--lang zh 强制中文", lang_module.resolve_language("zh") == "zh")
        check("--lang en 强制英文", lang_module.resolve_language("en") == "en")

        # System detection via POSIX-style env vars
        # 系统检测：POSIX 风格环境变量
        for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
            os.environ.pop(var, None)
        os.environ["LANG"] = "zh_CN.UTF-8"
        check("检测 zh 环境变量", lang_module.detect_system_language() == "zh")
        check("auto 跟随系统语言(zh)", lang_module.resolve_language("auto") == "zh")

        os.environ["LANG"] = "en_US.UTF-8"
        check("检测 en 环境变量", lang_module.detect_system_language() == "en")
        check("auto 跟随系统语言(en)", lang_module.resolve_language("auto") == "en")

        # Any non-Chinese language falls back to English
        # 其它语言一律回退英文
        os.environ["LANG"] = "ja_JP.UTF-8"
        check("非中文语言回退英文", lang_module.detect_system_language() == "en")

        # Message lookup follows the configured language
        # 消息查表跟随当前语言
        lang_module.configure("zh")
        check("zh 输出中文", "启动失败" in lang_module.t("startup_failed", exc="boom"))
        lang_module.configure("en")
        check("en 输出英文", "Startup failed" in lang_module.t("startup_failed", exc="boom"))
        lang_module.configure("fr")
        check("非法语言回退英文", "Startup failed" in lang_module.t("startup_failed", exc="boom"))
        lang_module.configure("de")
        check("未知 key 回退原文", lang_module.t("no_such_key_xyz") == "no_such_key_xyz")
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
        # Restore the system-detected language so other tests are unaffected
        # 恢复系统检测到的语言，避免影响其它用例
        lang_module.configure(lang_module.detect_system_language())


def _literal_error(efforts: list) -> str:
    """
    Rebuild the vLLM-style outer-schema error captured from a real upstream
    (as JSON-wrapped by Open WebUI), for a given accepted-efforts list.

    按给定挡位集合重建从真实上游捕获的 vLLM 风格外层 schema 报错
    （含 Open WebUI 的 JSON 包裹层）。
    """
    import json

    quoted = ", ".join(f"'{effort}'" for effort in efforts[:-1]) + f" or '{efforts[-1]}'"
    detail = (
        "1 validation error:\n"
        "  {'type': 'literal_error', 'loc': ('body', 'reasoning_effort'), "
        f"'msg': \"Input should be {quoted}\", "
        "'input': '__probe__', "
        f"'ctx': {{'expected': \"{quoted}\"}}}}"
    )
    return json.dumps({"detail": detail})


def test_effort_candidate_parsing() -> None:
    print("\n--- Effort candidate parsing (all real phrasings) / 挡位候选解析（全部真实措辞）---")
    full = _literal_error(["none", "minimal", "low", "medium", "high", "xhigh", "max"])
    check(
        "外层 schema：7 挡位",
        mprobe.extract_effort_candidates(full)
        == ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        full[:200],
    )
    partial = _literal_error(["none", "low", "medium", "high"])
    check(
        "外层 schema：4 挡位",
        mprobe.extract_effort_candidates(partial) == ["none", "low", "medium", "high"],
        partial[:200],
    )

    import json

    harmony = json.dumps(
        {
            "detail": "reasoning_effort='max' is not supported by Harmony. "
            "Supported values are: high, medium, low."
        }
    )
    check(
        "第二层 Harmony 措辞（旧实现完全读不懂）",
        mprobe.extract_effort_candidates(harmony) == ["low", "medium", "high"],
        harmony,
    )
    harmony_none = json.dumps({"detail": "Harmony does not support reasoning_effort='none'"})
    check(
        "第二层只否定一个值时不下结论",
        mprobe.extract_effort_candidates(harmony_none) == [],
        harmony_none,
    )
    qwen = json.dumps(
        {"detail": "Unexpected reasoning effort max. Supported types are xhigh (default), medium, and low."}
    )
    check(
        "第二层 Qwen 措辞（空格而非下划线）",
        mprobe.extract_effort_candidates(qwen) == ["low", "medium", "xhigh"],
        qwen,
    )
    check("从 Qwen 措辞里提取默认挡位", mprobe.extract_default_effort(qwen) == "xhigh", qwen)
    check(
        "没有 (default) 标注时不猜默认值",
        mprobe.extract_default_effort(harmony) is None,
        harmony,
    )
    check(
        "非 reasoning_effort 的报错不解析",
        mprobe.extract_effort_candidates(
            "1 validation error:\n  {'type': 'literal_error', 'loc': ('body', 'stop'), "
            "'msg': \"Input should be 'stop' or 'length'\", 'input': 'x'}"
        )
        == [],
    )
    check(
        "同构但值非挡位的报错不解析",
        mprobe.extract_effort_candidates(
            "loc: ('body', 'reasoning_effort'), msg: \"Input should be 'left' or 'right'\""
        )
        == [],
    )
    check("空文本返回空", mprobe.extract_effort_candidates("") == [])
    check("普通 500 文本返回空", mprobe.extract_effort_candidates("Internal Server Error") == [])
    check(
        "线上 400 是否与挡位有关",
        mprobe.looks_like_effort_error(qwen)
        and mprobe.looks_like_effort_error(harmony_none)
        and not mprobe.looks_like_effort_error("Model is not available"),
    )


def test_parameter_attribution() -> None:
    print("\n--- Request-parameter attribution / 请求参数归因 ---")
    check(
        "pydantic loc 精确归因",
        mprobe.parameter_of_error(
            "1 validation error:\n  {'type': 'literal_error', 'loc': ('body', 'reasoning_effort')}"
        )
        == "reasoning_effort",
    )
    check(
        "无 loc 时按关键词归因（vLLM 工具报错）",
        mprobe.parameter_of_error(
            '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
        )
        in ("tools", "tool_choice"),
    )
    check(
        "无法归因时返回 None",
        mprobe.parameter_of_error("Internal Server Error") is None,
    )


def test_probe_payloads() -> None:
    print("\n--- Probe payloads / 探测载荷 ---")
    effort = mprobe.effort_payload("m", "high")
    check(
        "挡位载荷最小且只带一个值",
        effort["model"] == "m"
        and effort["reasoning_effort"] == "high"
        and effort["max_tokens"] == 1
        and effort["stream"] is False,
        str(effort),
    )
    check("基线载荷不带 reasoning_effort", "reasoning_effort" not in mprobe.baseline_payload("m"))
    merged = mprobe.parameter_payload("m", ["tools", "tool_choice", "response_format"])
    check(
        "参数合并请求同时携带三项",
        merged.get("tools") and merged.get("tool_choice") == "auto"
        and (merged.get("response_format") or {}).get("type") == "json_schema",
        str(merged)[:200],
    )
    vision = mprobe.vision_payload("m")
    check(
        "视觉载荷是图片内容块",
        any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in vision["messages"][0]["content"]
        ),
        str(vision)[:200],
    )

    import json

    check(
        "从响应体判断思考内容：有",
        mprobe.response_has_reasoning(
            json.dumps({"choices": [{"message": {"content": None, "reasoning": "We"}}]})
        )
        is True,
    )
    check(
        "从响应体判断思考内容：无",
        mprobe.response_has_reasoning(
            json.dumps({"choices": [{"message": {"content": "P", "reasoning": None}}]})
        )
        is False,
    )
    check(
        "从响应体判断思考内容：看不出来时为 None",
        mprobe.response_has_reasoning(
            json.dumps({"choices": [{"message": {"content": None, "reasoning": None}}]})
        )
        is None
        and mprobe.response_has_reasoning("<html>") is None,
    )


def test_reasoning_info_derivation() -> None:
    print("\n--- Reasoning info derivation / 思考挡位信息推导 ---")
    info = mprobe.build_reasoning_info(
        ["high", "none", "medium", "low", "minimal", "xhigh", "max"], "xhigh", True
    )
    check(
        "supported_efforts 按规范顺序排序",
        info["supported_efforts"] == ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        str(info),
    )
    check("默认挡位来自引擎声明", info["default_effort"] == "xhigh", str(info))
    check("含 none 时非强制", info["mandatory"] is False, str(info))
    check("default_enabled 如实透出", info["default_enabled"] is True, str(info))

    unverified = mprobe.build_reasoning_info(["low", "medium", "high"])
    check(
        "引擎未声明默认值时不输出该字段",
        "default_effort" not in unverified and "default_enabled" not in unverified,
        str(unverified),
    )
    check("无 none 时判定为强制思考", unverified["mandatory"] is True, str(unverified))
    check("空挡位返回 None", mprobe.build_reasoning_info([]) is None)
    check(
        "未知挡位排在末尾且透传",
        mprobe.build_reasoning_info(["turbo", "low", "none"])["supported_efforts"]
        == ["none", "low", "turbo"],
    )
    check("模态由视觉结论推导", mprobe.build_architecture(False) == {
        "modality": "text->text",
        "input_modalities": ["text"],
        "output_modalities": ["text"],
    })
    check("视觉未知时不输出模态", mprobe.build_architecture(None) is None)
    check(
        "思考能力：接受非关闭挡位即具备",
        mprobe.derive_reasoning_capability(["none", "low"], None) is True
        and mprobe.derive_reasoning_capability(["none"], None) is False,
    )
    check(
        "思考能力：挡位未知时看默认行为",
        mprobe.derive_reasoning_capability(None, True) is True
        and mprobe.derive_reasoning_capability(None, None) is None,
    )


def test_probe_cache_store() -> None:
    print("\n--- Probe cache (v2) persistence / 探测缓存（v2）持久化 ---")
    tmp = Path(tempfile.mkdtemp())
    cache_file = tmp / "model_probe_cache.json"
    cache = mprobe.ModelProbeCache(cache_file)

    check("初始无条目", len(cache) == 0)
    missing = cache.sync_with_models([("a", "fp-a"), ("b", "fp-b")])
    check("未缓存的模型需要探测", missing == ["a", "b"], str(missing))

    cache.record_result(
        "a",
        mprobe.ModelProbe(
            fingerprint="fp-a",
            probed_at=1.0,
            status=mprobe.STATUS_OK,
            supported_efforts=["none", "low"],
            efforts_verified=True,
            default_effort="low",
            default_enabled=True,
            capabilities={"vision": True, "function_calling": False, "reasoning": True},
            supported_parameters=["temperature", "tools"],
        ),
    )
    cache.record_result(
        "b", mprobe.ModelProbe(fingerprint="fp-b", status=mprobe.STATUS_UNPROBEABLE)
    )
    cache.save()
    check("缓存文件已生成", cache_file.exists())

    fresh = mprobe.ModelProbeCache(cache_file)
    fresh.load()
    check("重新加载后条目数一致", len(fresh) == 2)
    presented = fresh.present("a") or {}
    check(
        "重新加载后挡位完整",
        presented.get("reasoning", {}).get("supported_efforts") == ["none", "low"],
        str(presented),
    )
    check(
        "能力与参数一并持久化",
        presented.get("capabilities") == {"vision": True, "function_calling": False, "reasoning": True}
        and presented.get("supported_parameters") == ["temperature", "tools"],
        str(presented),
    )
    check(
        "模态由能力推导",
        presented.get("architecture", {}).get("modality") == "text+image->text",
        str(presented),
    )
    check(
        "不可探测的模型仍能给出能力，但不给 reasoning",
        (fresh.present("b") or {}).get("reasoning") is None,
        str(fresh.present("b")),
    )
    check("未探测模型返回 None", fresh.present("c") is None)

    check("结论性条目在指纹不变时不再探测", not fresh.needs_probe("a", "fp-a"))
    check("引擎指纹变化即需重探", fresh.needs_probe("a", "fp-a2"))
    check("不可探测条目不会被反复重探", not fresh.needs_probe("b", "fp-b"))
    check("未知模型需要探测", fresh.needs_probe("c", "fp-c"))

    # Negative cache + backoff: a failed probe must not be retried immediately.
    # 负缓存 + 退避：失败的探测不得立刻重试。
    fresh.record_failure("c", "fp-c", "boom")
    check("失败后进入退避", not fresh.needs_probe("c", "fp-c"))
    check(
        "退避到期后重试",
        fresh.needs_probe("c", "fp-c", now=time.time() + mprobe.BACKOFF_MAX_SECONDS + 1),
    )
    check(
        "退避随失败次数增长且有上限",
        mprobe.backoff_seconds(1) < mprobe.backoff_seconds(3)
        and mprobe.backoff_seconds(50) == mprobe.BACKOFF_MAX_SECONDS,
    )

    # A failed re-probe must not throw away facts established earlier.
    # 失败的重探不得丢掉此前已确立的事实。
    fresh.record_failure("a", "fp-a", "boom")
    check(
        "重探失败时保留已有结论",
        (fresh.present("a") or {}).get("reasoning", {}).get("supported_efforts") == ["none", "low"],
        str(fresh.present("a")),
    )

    # A live 400 that names one level disproves it.
    # 线上 400 点名某个挡位即证伪它。
    check("证伪挡位返回 True", fresh.invalidate_effort("a", "low"))
    check("被证伪的挡位消失且可立即重探", "low" not in (fresh.present("a") or {})["reasoning"]["supported_efforts"]
          and fresh.needs_probe("a", "fp-a"))
    check("未被声明的挡位无需处理", not fresh.invalidate_effort("a", "max"))

    missing = fresh.sync_with_models([("a", "fp-a")])
    check("模型消失后条目被清除", len(fresh) == 1 and missing == ["a"], str(missing))
    check("force 时全部模型重探", fresh.sync_with_models([("a", "fp-a")], force=True) == ["a"])

    # Corrupt and version-1 files must not crash, just start empty.
    # 损坏与旧版本文件不得崩溃，只是从空缓存开始。
    cache_file.write_text("{ not json", encoding="utf-8")
    broken = mprobe.ModelProbeCache(cache_file)
    broken.load()
    check("损坏的缓存文件不致崩溃", len(broken) == 0)

    cache_file.write_text(
        json.dumps({"version": 1, "models": {"a": {"supported_efforts": ["none"]}}}),
        encoding="utf-8",
    )
    old = mprobe.ModelProbeCache(cache_file)
    old.load()
    check("旧版本缓存被忽略并重探", len(old) == 0 and old.sync_with_models([("a", "fp")]) == ["a"])


if __name__ == "__main__":
    test_session_roundtrip()
    test_session_file()
    test_login_signal()
    test_model_normalization()
    test_model_fingerprint()
    test_shared_default_capabilities()
    test_model_list_extraction()
    test_config()
    test_probe_settings()
    test_error_shape()
    test_credentials_are_valid()
    test_language_detection()
    test_effort_candidate_parsing()
    test_parameter_attribution()
    test_probe_payloads()
    test_reasoning_info_derivation()
    test_probe_cache_store()

    total = len(PASSED) + len(FAILED)
    print("\n" + "=" * 60)
    print(f"Passed {len(PASSED)}/{total} / 通过 {len(PASSED)}/{total}")
    if FAILED:
        print("Failed items / 失败项：")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    print("All passed / 全部通过 ✅")
