"""
Pure-logic unit tests: no Playwright, no network required.

Run from the project root:
    python tests/test_units.py

纯逻辑单元测试，不依赖 Playwright，也不需要联网。

运行方式（项目根目录）：
    python tests/test_units.py
"""

from __future__ import annotations

import dataclasses
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
import probe_runner as runner  # noqa: E402  (owns the runtime singletons post-R5)
import session_store as store  # noqa: E402

# A Windows console whose code page cannot represent every character the script prints
# (cp936 here) would otherwise abort the run with UnicodeEncodeError *after* the last
# check, reporting a failure that never happened.
#
# 在无法表示全部输出字符的 Windows 控制台（此处 cp936）上，若不做处理，脚本会在最后一个
# 检查之后抛 UnicodeEncodeError 中止，报告一个并不存在的失败。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

PASSED: list = []
FAILED: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """
    Under pytest (R1) a failed check raises immediately so pytest reports the exact
    case; run directly, failures are collected and summarized at the end as before.
    """
    if condition:
        PASSED.append(name)
        print(f"  [ok]   {name}")
        return
    message = f"{name}{(' -> ' + detail) if detail else ''}"
    FAILED.append(message)
    print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))
    if "pytest" in sys.modules:
        raise AssertionError(message)


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
    settings = dataclasses.replace(settings, session_file=tmp / "session.json")

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

    # load_session() caches the parsed file (it runs on every request); a replaced file
    # must still be visible.
    # load_session() 会缓存解析结果（它每个请求都会跑）；文件被替换后必须仍能被看到。
    (tmp / "session.json").write_text('{"authorization": "Bearer first"}', encoding="utf-8")
    check("首次读取成功", store.load_session(settings).authorization == "Bearer first")
    check("命中缓存仍返回同一凭证", store.load_session(settings).authorization == "Bearer first")
    (tmp / "session.json").write_text(
        '{"authorization": "Bearer second-longer"}', encoding="utf-8"
    )
    check(
        "文件被替换后缓存失效",
        store.load_session(settings).authorization == "Bearer second-longer",
        "mtime/size 变化必须让缓存失效",
    )
    # An invalid file must not be cached as a failure either: once it is fixed, the valid
    # credentials must be visible immediately.
    # 无效文件同样不得把"失败"缓存下来：修好之后必须能立刻读到有效凭证。
    (tmp / "session.json").write_text('{"cookie": ""}', encoding="utf-8")
    try:
        store.load_session(settings)
        cached_failure = True
    except store.SessionInvalid:
        cached_failure = False
    (tmp / "session.json").write_text('{"cookie": "a=1; b=2"}', encoding="utf-8")
    check(
        "失败的读取不会被缓存（修好后立即可用）",
        not cached_failure and store.load_session(settings).cookie == "a=1; b=2",
        "SessionInvalid 之后必须能立刻读到修复后的内容",
    )


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
        "偏离实例模板的能力放进 x_open_webui_deviations（与信封级 x_open_webui 区分，R9）",
        extended.get("capabilities") is None
        and extended.get("x_open_webui") is None
        and extended.get("x_open_webui_deviations") == {"capabilities": {"usage": False}},
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
        "x_open_webui_deviations" not in matched and "capabilities" not in matched,
        str(matched),
    )
    no_template = proxy.normalize_model(
        {"id": "no-template", "info": {"meta": {"capabilities": {"vision": True}}}}, None
    )
    check(
        "模板不可用时整份能力作为偏离保留",
        no_template.get("x_open_webui_deviations") == {"capabilities": {"vision": True}},
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
        check("model 为 null 时原样返回", settings.resolve_model(None) is None)
        check(
            "model 非字符串原样返回（别名映射不抛 TypeError）",
            settings.resolve_model({"a": 1}) == {"a": 1}
            and settings.resolve_model(["a"]) == ["a"],
            "dict/list 必须不被哈希、原样返回",
        )
        check("非法风格回退 auto", settings.upstream_api_style == "auto")
        check("展开 ~ 路径", "~" not in str(settings.session_file), str(settings.session_file))

        check("auto 的候选前缀", settings.prefix_candidates() == ["/api/v1", "/api"])
        v1 = dataclasses.replace(settings, upstream_api_style="v1")
        check("v1 的候选前缀", v1.prefix_candidates() == ["/api/v1"])
        legacy = dataclasses.replace(settings, upstream_api_style="legacy")
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

    import mock_openwebui as mock_module
    from mock_openwebui import VALID_TOKEN, MockOpenWebUI

    server = MockOpenWebUI()
    server.start()
    try:
        settings = config_module.load_settings()
        settings = dataclasses.replace(
            settings,
            open_webui_base_url=server.base_url,
            # Both candidate prefixes must be probed, otherwise the SPA case below
            # would depend on the ambient .env
            # 两个候选前缀都必须被探测，否则下面的 SPA 用例会受本机 .env 影响
            upstream_api_style="auto",
        )

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

        # Open WebUI's SPA answers an unknown path with 200 + HTML. That must not be read
        # as "logged in", and it must not stop the other candidate from being tried.
        #
        # Open WebUI 的 SPA 会用 200 + HTML 回答未知路径：这既不能读作"已登录"，
        # 也不应阻止继续尝试另一个候选前缀。
        mock_module.SPA_PATHS.add("/api/v1/models")
        try:
            check(
                "SPA 的 200 + HTML 不被当作鉴权成功",
                not asyncio.run(store._credentials_are_valid(settings, expired)),
            )
            check(
                "SPA 挡住首选候选后仍能在另一候选上确认有效凭证",
                asyncio.run(store._credentials_are_valid(settings, ok)),
            )
        finally:
            mock_module.SPA_PATHS.discard("/api/v1/models")
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

    # save() must load first: the probe-heal path can save before anything loaded the
    # cache, and that must not wipe the entries already on disk.
    #
    # save() 必须先 load：自愈路径可能在尚未加载缓存时就 save，这绝不能抹掉磁盘上的条目。
    cache_file.write_text(
        json.dumps(
            {
                "version": mprobe.CACHE_VERSION,
                "models": {
                    "kept": {
                        "fingerprint": "fp-kept",
                        "status": mprobe.STATUS_OK,
                        "supported_efforts": ["none"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    never_loaded = mprobe.ModelProbeCache(cache_file)
    never_loaded.save()
    survived = mprobe.ModelProbeCache(cache_file)
    survived.load()
    check(
        "未 load 直接 save 不会清空磁盘缓存",
        len(survived) == 1
        and (survived.present("kept") or {}).get("reasoning", {}).get("supported_efforts") == ["none"],
        f"{len(survived)} 条：{cache_file.read_text(encoding='utf-8')[:160]}",
    )
    check(
        "save 之后不残留临时文件",
        not list(cache_file.parent.glob("*.tmp")),
        str([path.name for path in cache_file.parent.iterdir()]),
    )

    dedup = mprobe.ModelProbeCache(tmp / "dedup.json")
    check(
        "重复的模型 id 只返回一次（不并发重复探测）",
        dedup.sync_with_models([("dup", "fp-a"), ("dup", "fp-b"), ("dup", "fp-a")]) == ["dup"],
        str(dedup.sync_with_models([("dup", "fp-a")])),
    )


def test_probe_parameter_loop() -> None:
    print("\n--- Probe parameter loop / 探测参数循环 ---")
    import asyncio

    class _Response:
        """The minimal httpx.Response surface _probe_model touches."""

        def __init__(self, status: int, text: str):
            self.status_code = status
            self.text = text
            self.is_closed = True

        async def aclose(self) -> None:
            return None

    # An upstream that keeps blaming `tools` even after the proxy removed it. Without the
    # guard this makes the loop re-send the identical request until its round budget runs
    # out, and then report `ok` with an incomplete parameter list.
    #
    # 一个在 tools 被剔除后仍把 400 归咎于 tools 的上游：没有防护的话，循环会把同一发请求
    # 重发到轮次耗尽，然后再以 ok 的状态报出一份不完整的参数列表。
    tools_error = json.dumps(
        {
            "detail": '"auto" tool choice requires --enable-auto-tool-choice and '
            "--tool-call-parser to be set"
        }
    )
    sentinel_error = _literal_error(["none", "low"])
    attempts = {"parameters": 0}
    probe_headers = {"seen": None}

    class _Upstream:
        async def post(self, session, subpath, payload, stream=False, extra_headers=None):
            probe_headers["seen"] = extra_headers
            effort = payload.get("reasoning_effort")
            if effort == mprobe.PROBE_SENTINEL:
                return _Response(400, sentinel_error)
            if effort is not None:
                return _Response(200, "{}")
            if any(key in payload for key in mprobe.PROBED_PARAMETERS):
                attempts["parameters"] += 1
                return _Response(400, tools_error)
            if isinstance((payload.get("messages") or [{}])[0].get("content"), list):
                return _Response(
                    200, json.dumps({"choices": [{"message": {"content": "seen"}}]})
                )
            return _Response(200, json.dumps({"choices": [{"message": {"content": "pong"}}]}))

    original = runner.upstream
    runner.upstream = _Upstream()
    try:
        probe = asyncio.run(proxy._probe_model(None, "m", "fp"))
    finally:
        runner.upstream = original

    check(
        "归因落在已剔除的参数上时立即停止（不空转到轮次耗尽）",
        attempts["parameters"] == 2,
        f"参数探测请求数={attempts['parameters']}",
    )
    check(
        "未得出结论的参数按未解决处理（状态 partial）",
        probe.status == mprobe.STATUS_PARTIAL
        and "tools" not in probe.supported_parameters,
        f"{probe.status} {probe.supported_parameters}",
    )
    check(
        "D11: 探测请求带自报身份的头",
        probe_headers["seen"] == runner.PROBE_REQUEST_HEADERS,
        str(probe_headers["seen"]),
    )


def test_review_fixes() -> None:
    print("\n--- Regression checks for review fixes / 审查修复的回归检查 ---")
    import asyncio

    # --- login-signal URL boundary: sibling namespaces must not match ---
    base = "https://webui.example.com"
    api = f"{base}/api"
    bearer = {"authorization": "Bearer x.y"}
    check(
        "同级路径 /api-evil、/apiary 不构成登录信号",
        not store.is_login_signal(f"{base}/api-evil/models", bearer, api)
        and not store.is_login_signal(f"{base}/apiary/models", bearer, api),
    )
    check(
        "精确前缀与前缀+斜杠仍命中",
        store.is_login_signal(api, bearer, api)
        and store.is_login_signal(f"{base}/api/models", bearer, api),
    )

    # --- --lang pre-scan recognizes the equals form ---
    lang_module.configure("en")
    proxy._preconfigure_language(["--lang=zh", "--check"])
    check("--lang=zh 等号形式生效", lang_module.current() == "zh", lang_module.current())
    proxy._preconfigure_language(["--lang", "en"])
    check("--lang en 空格形式仍生效", lang_module.current() == "en", lang_module.current())
    proxy._preconfigure_language(["--port", "9000"])
    check("没有 --lang 时不改变语言", lang_module.current() == "en", lang_module.current())
    lang_module.configure(lang_module.detect_system_language())

    # --- _get_int minimum bound ---
    os.environ["_PROXY_TEST_INT"] = "-1"
    check(
        "低于下界的整数回退默认值",
        config_module._get_int("_PROXY_TEST_INT", 8000, minimum=1) == 8000,
    )
    os.environ["_PROXY_TEST_INT"] = "9"
    check(
        "达到下界的整数被接受",
        config_module._get_int("_PROXY_TEST_INT", 8000, minimum=1) == 9,
    )
    os.environ["_PROXY_TEST_INT"] = "-1"
    check(
        "未设下限时负数不回退",
        config_module._get_int("_PROXY_TEST_INT", 0) == -1,
    )
    os.environ.pop("_PROXY_TEST_INT", None)

    # --- refresh-state announcement handshake ---
    async def _announcement_flow() -> list:
        state = proxy._RefreshState()
        results = []
        state.announce({"a"})
        results.append((set(state.pending), state.announcement().is_set()))
        state.reset_announcement()
        results.append(state.announcement().is_set())
        state.announce(set())
        results.append((set(state.pending), state.announcement().is_set()))
        return results

    results = asyncio.run(_announcement_flow())
    check(
        "公布事件语义：公布置位/复位清零/空集也公布",
        results[0] == ({"a"}, True) and results[1] is False and results[2] == (set(), True),
        str(results),
    )

    # --- stale-prefix self-heal on 404 (upstream moved /api/v1 -> /api) ---
    class _Resp:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self.text = json.dumps(payload)
            self._payload = payload
            self.is_closed = True

        async def aclose(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class _MovingUpstream:
        def __init__(self):
            self.prefix = "/api/v1"
            self.get_models_calls = 0
            self.detect_calls = 0

        async def get_models(self, session):
            self.get_models_calls += 1
            if self.prefix == "/api/v1":
                return _Resp(404, {"detail": "Not Found"})
            return _Resp(200, {"data": [{"id": "m"}]})

        async def detect_prefix(self, session, refresh=False):
            self.detect_calls += 1
            self.prefix = "/api"
            return self.prefix, 200

    fake = _MovingUpstream()
    original_upstream = runner.upstream
    runner.upstream = fake
    try:
        raw = asyncio.run(proxy._fetch_raw_models(None))
    finally:
        runner.upstream = original_upstream
    check(
        "404 后重探前缀并重试到新前缀",
        fake.get_models_calls == 2 and fake.detect_calls == 1 and raw == [{"id": "m"}],
        f"get={fake.get_models_calls} detect={fake.detect_calls} raw={raw}",
    )

    class _StuckUpstream:
        def __init__(self):
            self.prefix = "/api/v1"
            self.get_models_calls = 0
            self.detect_calls = 0

        async def get_models(self, session):
            self.get_models_calls += 1
            return _Resp(404, {"detail": "Not Found"})

        async def detect_prefix(self, session, refresh=False):
            self.detect_calls += 1
            return self.prefix, 404

    fake = _StuckUpstream()
    runner.upstream = fake
    raised = False
    try:
        try:
            asyncio.run(proxy._fetch_raw_models(None))
        except proxy.HttpError as exc:
            raised = exc.status_code == 502
    finally:
        runner.upstream = original_upstream
    check(
        "重探后前缀未变时不重试并按原错误上报",
        raised and fake.get_models_calls == 1 and fake.detect_calls == 1,
        f"raised={raised} get={fake.get_models_calls} detect={fake.detect_calls}",
    )

    class _BrokenUpstream:
        def __init__(self):
            self.prefix = "/api/v1"
            self.get_models_calls = 0
            self.detect_calls = 0

        async def get_models(self, session):
            self.get_models_calls += 1
            return _Resp(500, {"detail": "boom"})

        async def detect_prefix(self, session, refresh=False):
            self.detect_calls += 1
            return self.prefix, 500

    fake = _BrokenUpstream()
    runner.upstream = fake
    raised = False
    try:
        try:
            asyncio.run(proxy._fetch_raw_models(None))
        except proxy.HttpError as exc:
            raised = exc.status_code == 502
    finally:
        runner.upstream = original_upstream
    check(
        "5xx 不触发前缀重探",
        raised and fake.get_models_calls == 1 and fake.detect_calls == 0,
        f"raised={raised} get={fake.get_models_calls} detect={fake.detect_calls}",
    )

    # --- fire-and-forget background tasks keep a strong reference ---
    async def _heal_flow() -> tuple:
        async def noop_refresh(*args, **kwargs):
            return True

        class _NoopCache:
            def load(self):
                return None

            def entry(self, model_id):
                return None

            def invalidate_effort(self, model_id, effort):
                return False

            def save(self):
                return None

        original_refresh = runner._refresh_model_probe
        original_cache = runner.model_probe
        runner._refresh_model_probe = noop_refresh
        runner.model_probe = _NoopCache()
        try:
            proxy._trigger_probe_heal("heal-check-model", "high")
            held = len(proxy._background_tasks)
            for task in list(proxy._background_tasks):
                await task
            return held, len(proxy._background_tasks)
        finally:
            runner._refresh_model_probe = original_refresh
            runner.model_probe = original_cache

    held, leftover = asyncio.run(_heal_flow())
    check(
        "自愈任务持有强引用且完成后自动移除",
        held == 1 and leftover == 0 and "heal-check-model" not in proxy._healing,
        f"held={held} leftover={leftover}",
    )


def test_review_fixes_round2() -> None:
    print("\n--- Regression checks round 2 (decisioned fixes) / 第二轮回归（已决策修复）---")
    import asyncio

    # --- B14: the heal path must work even when the cache was never loaded ---
    # --- B14：缓存从未加载时 heal 也必须能剔除被证伪的挡位 ---
    heal_dir = Path(tempfile.mkdtemp())
    heal_file = heal_dir / "model_probe_cache.json"
    heal_file.write_text(
        json.dumps(
            {
                "version": mprobe.CACHE_VERSION,
                "models": {
                    "heal-late": {
                        "fingerprint": "fp-late",
                        "status": mprobe.STATUS_OK,
                        "supported_efforts": ["none", "low", "high"],
                        "efforts_verified": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    late_cache = mprobe.ModelProbeCache(heal_file)  # 故意不 load

    async def _heal_before_load() -> list:
        async def noop_refresh(*args, **kwargs):
            return True

        original_refresh = runner._refresh_model_probe
        original_cache = runner.model_probe
        runner.model_probe = late_cache
        runner._refresh_model_probe = noop_refresh
        try:
            proxy._trigger_probe_heal("heal-late", "high")
            for task in list(proxy._background_tasks):
                await task
        finally:
            runner.model_probe = original_cache
            runner._refresh_model_probe = original_refresh
        late_cache.load()
        return (late_cache.present("heal-late") or {}).get("reasoning", {}).get("supported_efforts") or []

    healed = asyncio.run(_heal_before_load())
    check(
        "B14: 未加载缓存时 heal 仍剔除被证伪挡位",
        healed == ["none", "low"],
        str(healed),
    )
    check(
        "B14: 证伪集合被持久化（防止更早的探测结果复活该挡位）",
        late_cache.entry("heal-late").invalidated_efforts == ["high"]
        and late_cache.entry("heal-late").last_error != "",
        str(late_cache.entry("heal-late")),
    )

    # --- B10: an earlier-completed probe result must not resurrect a disproved level ---
    # --- B10：更早完成的探测结果不得让被证伪挡位复活 ---
    b10_cache = mprobe.ModelProbeCache(Path(tempfile.mkdtemp()) / "b10.json")
    b10_cache.record_result(
        "m",
        mprobe.ModelProbe(
            fingerprint="fp",
            status=mprobe.STATUS_OK,
            supported_efforts=["none", "low", "high"],
            efforts_verified=True,
        ),
    )
    check("B10 前置：线上 400 证伪 high", b10_cache.invalidate_effort("m", "high"))
    b10_cache.record_result(
        "m",
        mprobe.ModelProbe(
            fingerprint="fp",
            status=mprobe.STATUS_OK,
            # An earlier probe that finished before the upstream started rejecting
            # "high": the naive whole-entry overwrite would bring it back.
            #
            # 在上游开始拒绝 high 之前完成的探测：天真的整体覆盖会让它复活。
            supported_efforts=["none", "low", "high"],
            efforts_verified=True,
        ),
    )
    presented_b10 = b10_cache.present("m") or {}
    check(
        "B10: record_result 不让被证伪挡位复活",
        "high" not in (presented_b10.get("reasoning", {}).get("supported_efforts") or [])
        and b10_cache.entry("m").invalidated_efforts == ["high"]
        and not b10_cache.entry("m").efforts_verified,
        str(b10_cache.entry("m")),
    )
    check(
        "B10: 引擎指纹变化时证伪集合清空",
        (
            b10_cache.record_result(
                "m",
                mprobe.ModelProbe(
                    fingerprint="fp-new",
                    status=mprobe.STATUS_OK,
                    supported_efforts=["none", "high"],
                ),
            ).invalidated_efforts
            == []
        ),
        str(b10_cache.entry("m").invalidated_efforts),
    )

    # --- D13: unauthenticated + non-loopback listen is refused ---
    # --- D13：无鉴权 + 非回环监听被拒绝 ---
    base_settings = config_module.load_settings()
    insecure = dataclasses.replace(base_settings, proxy_api_key="", proxy_host="0.0.0.0")
    refused = False
    try:
        proxy._enforce_listen_safety(insecure)
    except SystemExit:
        refused = True
    check("D13: 空 Key + 0.0.0.0 拒绝启动", refused)
    check(
        "D13: ALLOW_INSECURE 可覆盖",
        (
            proxy._enforce_listen_safety(
                dataclasses.replace(insecure, allow_insecure=True)
            )
            or True
        ),
    )
    check(
        "D13: 回环地址不拒绝",
        (
            proxy._enforce_listen_safety(
                dataclasses.replace(base_settings, proxy_api_key="", proxy_host="127.0.0.1")
            )
            or True
        ),
    )
    check(
        "D13: 有 Key 的非回环不拒绝",
        (
            proxy._enforce_listen_safety(
                dataclasses.replace(base_settings, proxy_api_key="sk-x", proxy_host="0.0.0.0")
            )
            or True
        ),
    )

    # --- B4: probe_prefix fallback reports the most informative status ---
    # --- B4：probe_prefix 兜底上报最有信息量的状态 ---
    from upstream import _most_informative_status

    check(
        "B4: 首候选 5xx + 次候选 404 上报 5xx 而非 404",
        _most_informative_status([500, 404]) == 500
        and _most_informative_status([404, 401]) == 401
        and _most_informative_status([404, 200]) == 200
        and _most_informative_status([404]) == 404,
    )

    # --- R4: quantization regex requires an underscore segment after Q<n> ---
    # --- R4：量化正则要求 Q<n> 后至少带一个下划线段 ---
    check(
        "R4: 独立 Q4 不再误标，Q4_K_M 仍命中",
        proxy.normalize_model({"id": "q4-summary"}) .get("quantization") is None
        and proxy.normalize_model({"id": "model-Q4_K_M"}).get("quantization") == "Q4_K_M",
    )


def test_created_timestamp_and_summaries() -> None:
    print("\n--- created parsing chain & shared summaries / created 解析链与公共摘要 ---")

    # --- #7 created timestamp candidate chain ---
    check(
        "created：epoch 整数直接解析",
        proxy.normalize_model({"id": "x", "created": 1700000000})["created"] == 1700000000,
    )
    check(
        "created：数字字符串仍可解析",
        proxy.normalize_model({"id": "x", "created": "1700000000"})["created"] == 1700000000,
    )
    check(
        "created：ISO 字符串解析为 epoch（无时区按 UTC）",
        proxy.normalize_model({"id": "x", "info": {"created_at": "2024-01-01T00:00:00"}})["created"]
        == 1704067200,
    )
    check(
        "created：带时区的 ISO 字符串解析为 epoch",
        proxy.normalize_model(
            {"id": "x", "info": {"created_at": "2024-01-01T00:00:00+00:00"}}
        )["created"]
        == 1704067200,
    )
    check(
        "created：info.created_at 解析失败时回退 raw.created",
        proxy.normalize_model(
            {"id": "x", "created": 1700000000, "info": {"created_at": "not-a-date"}}
        )["created"]
        == 1700000000,
    )
    check(
        "created：info.created_at 为 None 时回退 raw.created",
        proxy.normalize_model({"id": "x", "created": 1700000000, "info": {}})["created"]
        == 1700000000,
    )
    check(
        "created：全链无法解析才归零",
        proxy.normalize_model({"id": "x", "created": "abc", "info": {"created_at": ""}})["created"]
        == 0,
    )
    check(
        "created：布尔值不算时间戳",
        proxy.normalize_model({"id": "x", "created": True})["created"] == 0,
    )

    # --- #17 single definition of normalization + fingerprint ---
    raw_models = [
        {"id": "a", "created": 100, "openai": {"root": "org/a"}, "info": {}},
        {"id": "b"},
        "string-model",
        {"no-id-here": True},
    ]
    models, summaries = proxy._model_summaries(raw_models)
    check(
        "公共摘要：models 与 summaries 一一对应且跳过无效条目",
        [m["id"] for m in models] == ["a", "b", "string-model"]
        and [s[0] for s in summaries] == ["a", "b", "string-model"]
        and len(models) == len(summaries),
        str(summaries),
    )
    check(
        "公共摘要：dict 条目指纹与 _model_fingerprint(raw, id) 一致",
        summaries[0][1] == proxy._model_fingerprint(raw_models[0], "a")
        and summaries[1][1] == proxy._model_fingerprint(raw_models[1], "b"),
        str(summaries),
    )
    check(
        "公共摘要：裸字符串条目无引擎身份，指纹为空",
        summaries[2] == ("string-model", ""),
        str(summaries),
    )
    check(
        "公共摘要：shared_capabilities 传入时不改变摘要口径",
        proxy._model_summaries(raw_models, {"vision": True})[1] == summaries,
    )
    check(
        "公共摘要：空列表产出空结果",
        proxy._model_summaries([]) == ([], []),
    )


def test_language_keys() -> None:
    print("\n--- Language keys: every used key is defined (R3) / 语言 key：用到的都已定义（R3）---")
    import re as re_module

    pattern = re_module.compile(r"""lang\.t\(\s*["']([^"']+)["']""")
    used: dict = {}
    for path in sorted(REPO_ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            used.setdefault(match.group(1), set()).add(path.name)

    defined = set(lang_module._MESSAGE_TEXTS)
    check(
        "R3: 扫描到了 lang.t 字面量 key",
        len(used) > 50,
        f"used={len(used)} defined={len(defined)}",
    )
    missing = sorted(key for key in used if key not in defined)
    check(
        "R3: 源码里用到的 lang key 全部已定义",
        not missing,
        ", ".join(
            f"{key} ({'/'.join(sorted(used[key]))})" for key in missing
        ),
    )
    unused = sorted(defined - set(used))
    check(
        "R3: 没有定义后从未被使用的 lang key",
        not unused,
        ", ".join(unused),
    )


def test_review_fixes_round3() -> None:
    print("\n--- Regression checks round 3 (autonomous fixes) / 第三轮回归（自主修复项）---")
    import asyncio

    import atomic_json
    import httpx
    import upstream as upstream_module
    from mock_openwebui import VALID_COOKIE, _Handler

    base_settings = dataclasses.replace(
        config_module.settings,
        open_webui_base_url="http://upstream.test",
        upstream_api_style="auto",
    )

    # --- R6/D10: one atomic-write implementation, complete file, no temp residue ---
    # --- R6/D10：原子写只有一份实现，写入完整，且不留临时文件 ---
    target_dir = Path(tempfile.mkdtemp())
    target = target_dir / "state.json"
    atomic_json.atomic_write_json(target, {"a": 1})
    check(
        "R6/D10: 凭证与探测缓存共用同一份原子写实现",
        store.atomic_write_json is atomic_json.atomic_write_json
        and mprobe.atomic_write_json is atomic_json.atomic_write_json,
    )
    check(
        "R6/D10: 写入内容完整且不留 .tmp 残留",
        json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
        and not (target_dir / "state.json.tmp").exists(),
        sorted(p.name for p in target_dir.iterdir()),
    )

    # --- R12: the redacted summary must not carry a long secret prefix ---
    # --- R12：脱敏摘要不得携带过长的密钥前缀 ---
    described = store.Session(authorization="Bearer abcdefghijklmnop").describe()
    check(
        "R12: 脱敏摘要只暴露前 8 个字符",
        "Bearer a" in described and "abcdefghij" not in described,
        described,
    )

    # --- D7: credentials captured for another upstream are refused, empty is allowed ---
    # --- D7：为别的上游抓取的凭证被拒绝，空值放行 ---
    same_dir = Path(tempfile.mkdtemp())
    cases = [
        ("https://webui.example.com", "https://webui.example.com/", False),
        ("https://WEBUI.example.com", "https://webui.example.com", False),
        ("https://other.example.com", "https://webui.example.com", True),
        ("", "https://webui.example.com", False),
    ]
    results = []
    for index, (stored, configured, expect_invalid) in enumerate(cases):
        session_file = same_dir / f"session-{index}.json"
        session_file.write_text(
            json.dumps({"authorization": "Bearer x", "base_url": stored}),
            encoding="utf-8",
        )
        cfg = dataclasses.replace(
            config_module.settings,
            session_file=session_file,
            open_webui_base_url=configured,
        )
        try:
            store.load_session(cfg)
            invalid = False
        except store.SessionInvalid:
            invalid = True
        results.append(invalid == expect_invalid)
    check(
        "D7: base_url 不一致拒绝、空值与尾斜杠/大小写差异放行",
        all(results),
        str(results),
    )
    mismatch_file = same_dir / "session-mismatch.json"
    mismatch_file.write_text(
        json.dumps({"authorization": "Bearer x", "base_url": "https://other.example.com"}),
        encoding="utf-8",
    )
    message = ""
    try:
        store.load_session(
            dataclasses.replace(
                config_module.settings,
                session_file=mismatch_file,
                open_webui_base_url="https://webui.example.com",
            )
        )
    except store.SessionInvalid as exc:
        message = str(exc)
    check(
        "D7: 错误信息同时指出记录地址与当前配置",
        "other.example.com" in message and "webui.example.com" in message,
        message,
    )

    # --- D9: a Playwright failure reaches the caller as SessionError, with the fix ----
    # --- D9：Playwright 失败以 SessionError 抵达调用方，并带上解决办法 ---
    class _FailingChromium:
        async def launch(self, headless: bool = False):
            raise RuntimeError("Executable doesn't exist at ...")

    class _FakePlaywright:
        chromium = _FailingChromium()

    launch_message = ""
    try:
        asyncio.run(store._launch_browser(_FakePlaywright(), headless=True))
    except store.SessionError as exc:
        launch_message = str(exc)
    except Exception as exc:  # noqa: BLE001 - the point of the check is the type
        launch_message = f"{type(exc).__name__}: {exc}"
    check(
        "D9: 浏览器启动失败转 SessionError 并附安装/无头提示",
        "playwright install chromium" in launch_message
        and "LOGIN_HEADLESS" in launch_message,
        launch_message[:200],
    )

    class _FailingBrowser:
        async def new_context(self, **kwargs: object):
            raise RuntimeError("context refused")

    context_message = ""
    try:
        asyncio.run(store._open_login_page(_FailingBrowser(), base_settings))
    except store.SessionError as exc:
        context_message = str(exc)
    except Exception as exc:  # noqa: BLE001
        context_message = f"{type(exc).__name__}: {exc}"
    check(
        "D9: 上下文创建失败同样转 SessionError",
        context_message.startswith("无法创建浏览器上下文")
        or "Cannot open a browser context" in context_message,
        context_message[:200],
    )

    # --- R10: the mock upstream accepts a cookie-only session ---
    # --- R10：模拟上游接受仅有 Cookie 的会话 ---
    check(
        "R10: mock 上游接受 Cookie 形态的凭证",
        _Handler._authorized({"Cookie": f"theme=dark; {VALID_COOKIE}"})
        and _Handler._authorized({"Authorization": "Bearer mock-jwt-token"})
        and not _Handler._authorized({"Cookie": "theme=dark"})
        and not _Handler._authorized({}),
    )

    # --- R14: the smoke probe poll timeout is configurable ---
    # --- R14：冒烟测试的探测轮询上限可配置 ---
    import test_smoke as smoke_module

    os.environ["SMOKE_PROBE_TIMEOUT"] = "12.5"
    try:
        overridden = smoke_module._env_seconds("SMOKE_PROBE_TIMEOUT", 60.0)
    finally:
        os.environ.pop("SMOKE_PROBE_TIMEOUT", None)
    check(
        "R14: SMOKE_PROBE_TIMEOUT 可覆盖轮询上限，未设置/非法时用默认值",
        overridden == 12.5
        and smoke_module._env_seconds("SMOKE_PROBE_TIMEOUT", 60.0) == 60.0
        and smoke_module._env_seconds("SMOKE_PROBE_TIMEOUT", 60.0) == 60.0
        and smoke_module.PROBE_POLL_TIMEOUT > 0,
        f"override={overridden} default={smoke_module.PROBE_POLL_TIMEOUT}",
    )

    # ------------------------------------------------------------------ #
    # Upstream behaviour that needs an event loop and a fake HTTP client
    # 需要事件循环与假 HTTP 客户端的上游行为
    # ------------------------------------------------------------------ #
    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self.headers = httpx.Headers({"content-type": "application/json"})
            self.is_closed = False

        async def aclose(self) -> None:
            self.is_closed = True

    class _FakeAsyncClient:
        """Records every request and answers with the queued statuses."""

        def __init__(self, statuses: list):
            self.statuses = list(statuses)
            self.requests = []
            self.is_closed = False
            # Real request building, so header encoding behaves exactly as it does in
            # production (that is what raises UnicodeEncodeError in the B12 check).
            #
            # 用真实的请求构造，使请求头编码行为与生产一致（B12 那条检查依赖的正是它抛
            # UnicodeEncodeError）。
            self._builder = httpx.Client()

        def build_request(self, method, url, **kwargs):
            return self._builder.build_request(method, url, **kwargs)

        async def send(self, request, stream: bool = False):
            self.requests.append(request)
            return _FakeResponse(self.statuses.pop(0), {})

        async def aclose(self) -> None:
            self.is_closed = True

    class _CountingJson:
        """Counts json.dumps calls so "serialized once" (R8) is observable."""

        def __init__(self, real, counter: dict):
            self._real = real
            self._counter = counter

        def dumps(self, *args, **kwargs):
            self._counter["dumps"] += 1
            return self._real.dumps(*args, **kwargs)

    session = store.Session(authorization="Bearer abc", user_agent="ua")

    # --- R8 + D12: one serialization per call, and the prefix flips after 3 fallbacks ---
    # --- R8 + D12：每次调用只序列化一次，连续 3 次回退后翻转前缀 ---
    counter = {"dumps": 0}
    client = upstream_module.UpstreamClient(base_settings)
    client.prefix = "/api/v1"
    fake = _FakeAsyncClient([404, 200, 404, 200, 404, 200, 200])
    client._client = fake  # type: ignore[assignment]
    real_json = upstream_module.json
    upstream_module.json = _CountingJson(real_json, counter)  # type: ignore[assignment]
    try:
        for _ in range(3):
            asyncio.run(client.post(session, "chat/completions", {"model": "m"}))
        asyncio.run(client.post(session, "chat/completions", {"model": "m"}))
    finally:
        upstream_module.json = real_json  # type: ignore[assignment]

    check(
        "R8: 请求体每次调用只序列化一次（回退不重复序列化）",
        counter["dumps"] == 4 and len(fake.requests) == 7,
        f"dumps={counter['dumps']} requests={len(fake.requests)}",
    )
    check(
        "R8: 发出的请求体就是 payload 的 JSON 编码",
        fake.requests[0].content == json.dumps({"model": "m"}).encode("utf-8"),
        str(fake.requests[0].content[:80]),
    )
    check(
        "D12: 连续 3 次回退后缓存前缀翻转为实际应答者",
        client.prefix == "/api",
        str(client.prefix),
    )
    check(
        "D12: 翻转后首个候选即命中的前缀（不再白跑一跳）",
        str(fake.requests[6].url).endswith("/api/chat/completions"),
        str(fake.requests[6].url),
    )

    # A 404 on every candidate is evidence for no prefix at all: flipping on it would
    # only add a wasted hop to every later request.
    #
    # 所有候选都 404 时没有任何前缀得到证据：据此翻转只会给之后的每个请求都多加一跳。
    not_found_client = upstream_module.UpstreamClient(base_settings)
    not_found_client.prefix = "/api/v1"
    not_found_client._client = _FakeAsyncClient([404, 404, 404, 404])  # type: ignore[assignment]
    for _ in range(2):
        asyncio.run(not_found_client.post(session, "chat/completions", {"model": "m"}))
    check(
        "D12: 全候选 404 不翻转前缀",
        not_found_client.prefix == "/api/v1",
        str(not_found_client.prefix),
    )

    # --- B12: an unencodable header value is a bad request, not an upstream failure ---
    # --- B12：无法编码的头值属于请求有误，而不是上游故障 ---
    async def _unencodable_header() -> str:
        broken = store.Session(authorization="Bearer \u4e2d\u6587", user_agent="ua")
        broken_client = upstream_module.UpstreamClient(base_settings)
        broken_client.prefix = "/api/v1"
        broken_client._client = _FakeAsyncClient([200])  # type: ignore[assignment]
        try:
            await broken_client.post(broken, "chat/completions", {"model": "m"})
        except upstream_module.UpstreamRequestInvalid:
            return "UpstreamRequestInvalid"
        except BaseException as exc:  # noqa: BLE001 - report what actually escaped
            return type(exc).__name__
        return "no error"

    check(
        "B12: 无法编码的请求头归类为 UpstreamRequestInvalid",
        asyncio.run(_unencodable_header()) == "UpstreamRequestInvalid",
    )
    mapped = proxy._invalid_request_response(
        upstream_module.UpstreamRequestInvalid("boom")
    )
    check(
        "B12: 该错误对外映射为 400 而非 500",
        mapped.status_code == 400
        and json.loads(mapped.body).get("error", {}).get("code") == "invalid_request",
        mapped.body[:160],
    )

    # --- D3: concurrent first probes share a single prefix sweep ---
    # --- D3：并发首次探测只跑一轮候选 ---
    async def _concurrent_probe() -> tuple:
        dedup_client = upstream_module.UpstreamClient(base_settings)
        probes = {"count": 0}

        async def fake_probe_prefix(_session) -> tuple:
            probes["count"] += 1
            await asyncio.sleep(0.05)
            dedup_client.prefix = "/api/v1"
            return "/api/v1", 200

        dedup_client.probe_prefix = fake_probe_prefix  # type: ignore[assignment]
        results = await asyncio.gather(
            *(dedup_client.detect_prefix(None) for _ in range(5))
        )
        return probes["count"], set(results)

    probe_count, prefixes = asyncio.run(_concurrent_probe())
    check(
        "D3: 5 个并发调用只触发一次前缀探测且结果一致",
        probe_count == 1 and prefixes == {"/api/v1"},
        f"probes={probe_count} prefixes={prefixes}",
    )

    # --- D4: an unprobeable model reuses the parameter request as its baseline ---
    # --- D4：不可探测的模型复用参数请求作为 baseline ---
    class _UnprobeableUpstream:
        """
        An upstream that ignores the sentinel (so the model is "unprobeable") and
        accepts everything else, while recording every probe payload it receives.
        """

        def __init__(self) -> None:
            self.requests: list = []

        async def post(self, session, subpath, payload, stream=False, extra_headers=None):
            self.requests.append(payload)
            if isinstance((payload.get("messages") or [{}])[0].get("content"), list):
                body = {"choices": [{"message": {"content": "seen"}}]}
            else:
                body = {"choices": [{"message": {"content": "pong"}}]}
            return _TextResponse(200, json.dumps(body))

    class _TextResponse:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text
            self.is_closed = True

        async def aclose(self) -> None:
            return None

    unprobeable = _UnprobeableUpstream()
    original_upstream = runner.upstream
    runner.upstream = unprobeable  # type: ignore[assignment]
    try:
        probe = asyncio.run(proxy._probe_model(None, "m", "fp"))
    finally:
        runner.upstream = original_upstream
    check(
        "D4: 不可探测模型的探测请求数降为 3（哨兵+参数+视觉）",
        len(unprobeable.requests) == 3,
        f"requests={len(unprobeable.requests)}",
    )
    check(
        "D4: 复用参数请求的响应体判定默认思考行为",
        probe.status == mprobe.STATUS_UNPROBEABLE
        and probe.default_enabled is False
        and probe.capabilities.get("vision") is True,
        f"{probe.status} default_enabled={probe.default_enabled} {probe.capabilities}",
    )


def test_base_url_acceptance_and_http_notice() -> None:
    print("\n--- Upstream address: accepted forms + cleartext notice (U-8) ---")
    env_backup = dict(os.environ)

    def load(url: str):
        os.environ["OPEN_WEBUI_BASE_URL"] = url
        try:
            return config_module.load_settings()
        except RuntimeError:
            return None

    try:
        # The shapes this project is actually deployed in must all be accepted. An
        # earlier revision refused the LAN / plain-http / custom-port ones by copying
        # the Cloudflare Workers port's platform constraint; TODO.md U-8's revised
        # direction is explicit that this repository must not do that.
        #
        # 本项目真实的部署形态必须全部被接受。早期版本照搬了 Cloudflare Worker 移植版的
        # 平台约束、会拒绝局域网 / 明文 http / 自定义端口这几类；修订后的 TODO.md U-8
        # 明确要求本仓库不得那样做。
        accepted = {
            "回环 http": "http://127.0.0.1:8080",
            "localhost": "http://localhost:8080",
            "IPv6 回环": "http://[::1]:8080",
            "局域网 http + 自定义端口": "http://192.168.1.5:3000",
            "私网 IP": "http://10.1.2.3:8080",
            "链路本地 IP": "http://169.254.169.254",
            "公网 https": "https://webui.example.com",
            "https 自定义端口": "https://webui.example.com:8443",
        }
        loaded = {name: load(url) for name, url in accepted.items()}
        check(
            "真实部署形态一律接受（含局域网 http 与自定义端口）",
            all(settings is not None for settings in loaded.values()),
            ", ".join(
                f"{name}={url}"
                for name, url in accepted.items()
                if loaded[name] is None
            ),
        )

        check("缺少 scheme 仍然拒绝", load("localhost:8080") is None)
        check("端口非法仍然拒绝", load("http://localhost:not-a-port") is None)

        def warns(url: str) -> bool:
            settings = load(url)
            return bool(settings and settings.upstream_is_plain_http_nonloopback())

        check(
            "非回环明文 http 触发提示",
            warns("http://192.168.1.5:3000") and warns("http://webui.example.com"),
        )
        check(
            "回环地址保持静默（默认配置不产生噪音）",
            not warns("http://localhost:8080")
            and not warns("http://127.0.0.1:8080")
            and not warns("http://[::1]:8080"),
        )
        check(
            "https 地址不提示",
            not warns("https://webui.example.com")
            and not warns("https://webui.example.com:8443"),
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_passthrough_allow_config() -> None:
    print("\n--- Passthrough allowlist (U-1) / 透传白名单（U-1）---")
    env_backup = dict(os.environ)
    try:
        os.environ.pop("PASSTHROUGH_ALLOW", None)
        default = config_module.load_settings()
        check(
            "未配置时默认最小权限白名单",
            default.passthrough_allow == list(config_module.DEFAULT_PASSTHROUGH_ALLOW)
            and not default.passthrough_allow_all,
            str(default.passthrough_allow),
        )
        check(
            "默认放行 responses 及其子路径",
            default.passthrough_permits("responses") and default.passthrough_permits("responses/x"),
        )
        check(
            "默认不放行账号/配置/会话类路径",
            not any(
                default.passthrough_permits(path)
                for path in ("users", "configs", "chats/1", "auths/signin")
            ),
        )

        os.environ["PASSTHROUGH_ALLOW"] = "*"
        wildcard = config_module.load_settings()
        check(
            "PASSTHROUGH_ALLOW=* 恢复全量透传",
            wildcard.passthrough_allow_all
            and wildcard.passthrough_permits("users")
            and wildcard.passthrough_allow == [],
            str(wildcard.passthrough_allow_all),
        )

        os.environ["PASSTHROUGH_ALLOW"] = "responses, images"
        listed = config_module.load_settings()
        check(
            "显式白名单只放行列出的路径",
            listed.passthrough_permits("responses/x")
            and listed.passthrough_permits("images")
            and not listed.passthrough_permits("files"),
            str(listed.passthrough_allow),
        )

        os.environ["PASSTHROUGH_ALLOW"] = ""
        denied = config_module.load_settings()
        check(
            "显式空值 = 一条都不放行",
            not denied.passthrough_allow
            and not denied.passthrough_allow_all
            and not denied.passthrough_permits("responses"),
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def test_named_proxy_keys() -> None:
    print("\n--- Named proxy keys (U-10) / 具名代理 Key（U-10）---")
    env_backup = dict(os.environ)
    original_settings = proxy.settings
    try:
        os.environ["PROXY_API_KEY"] = "sk-legacy"
        os.environ["PROXY_API_KEYS"] = "alice:sk-alice, bob:sk-bob,plain-key"
        settings = config_module.load_settings()
        check(
            "解析 name:key 与裸 Key（自动命名）",
            settings.proxy_api_keys
            == {"alice": "sk-alice", "bob": "sk-bob", "key3": "plain-key"},
            str(settings.proxy_api_keys),
        )
        check("任一来源存在即启用鉴权", settings.authentication_enabled())

        proxy.settings = settings
        check(
            "旧 Key 与全部具名 Key 都通过校验",
            proxy.match_proxy_key("sk-legacy") == "default"
            and proxy.match_proxy_key("sk-alice") == "alice"
            and proxy.match_proxy_key("sk-bob") == "bob"
            and proxy.match_proxy_key("plain-key") == "key3",
        )
        check(
            "未知/空 Key 不通过",
            proxy.match_proxy_key("sk-nope") is None and proxy.match_proxy_key("") is None,
        )

        os.environ["PROXY_API_KEYS"] = "alice:,bob:sk-bob"
        trimmed = config_module.load_settings()
        check(
            "空 Key 值的条目被忽略",
            trimmed.proxy_api_keys == {"bob": "sk-bob"},
            str(trimmed.proxy_api_keys),
        )

        os.environ["PROXY_API_KEY"] = ""
        os.environ["PROXY_API_KEYS"] = "中文:中文键"
        chinese = config_module.load_settings()
        proxy.settings = chinese
        check(
            "非 ASCII 的 Key 可校验（不会退化成 500）",
            proxy.match_proxy_key("中文键") == "中文" and proxy.match_proxy_key("别的") is None,
        )

        os.environ["PROXY_API_KEYS"] = ""
        both_empty = config_module.load_settings()
        check("两个来源都为空时鉴权关闭", not both_empty.authentication_enabled())
    finally:
        proxy.settings = original_settings
        os.environ.clear()
        os.environ.update(env_backup)


def test_request_id() -> None:
    print("\n--- Request correlation id (U-7) / 请求关联 id（U-7）---")
    import logging

    import request_context as ctx

    check(
        "客户端 id 中的安全字符原样保留",
        ctx.sanitize_request_id("abc-123_def.9:00") == "abc-123_def.9:00",
        ctx.sanitize_request_id("abc-123_def.9:00"),
    )
    check(
        "换行等字符被丢弃（无法伪造日志行）",
        ctx.sanitize_request_id("abc\r\n2026-01-01 [ERROR] fake") == "abc2026-01-01ERRORfake",
        ctx.sanitize_request_id("abc\r\n2026-01-01 [ERROR] fake"),
    )
    check(
        "超长 id 截断到上限",
        len(ctx.sanitize_request_id("x" * 500)) == ctx.REQUEST_ID_MAX_LENGTH,
    )
    check(
        "不可用的 id 归为空",
        ctx.sanitize_request_id("   ") == "" and ctx.sanitize_request_id(None) == "",
    )
    generated = ctx.new_request_id("!!!")
    check(
        "清洗后为空则生成新 id",
        generated != "" and ctx.sanitize_request_id(generated) == generated,
        generated,
    )
    check("客户端提供可用值时沿用", ctx.new_request_id("req-1") == "req-1")

    ctx.bind_request_id("req-2")
    check("消息后缀携带当前请求 id", "req-2" in ctx.request_id_suffix(), ctx.request_id_suffix())
    formatter = ctx.RequestIdFormatter("%(request_id)s")
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", None, None)
    check("日志格式化器带上当前 id", formatter.format(record) == "req-2", formatter.format(record))
    ctx.bind_request_id("")
    check("请求上下文之外后缀为空", ctx.request_id_suffix() == "")


def test_probe_health_state() -> None:
    print("\n--- Probe health (U-11) / 探测健康状态（U-11）---")
    health = runner.ProbeHealth()
    check(
        "初始状态为 unknown",
        health.to_dict()["status"] == runner.PROBE_HEALTH_UNKNOWN,
        str(health.to_dict()),
    )

    health.record_auth_rejected("HTTP 401")
    health.record_auth_rejected("HTTP 401")
    check(
        "凭证被拒累计计数并进入 auth_rejected",
        health.to_dict()["status"] == runner.PROBE_HEALTH_AUTH_REJECTED
        and health.to_dict()["consecutive_auth_failures"] == 2,
        str(health.to_dict()),
    )
    health.record_degraded("boom")
    check(
        "暂时性失败不覆盖凭证问题",
        health.to_dict()["status"] == runner.PROBE_HEALTH_AUTH_REJECTED,
        str(health.to_dict()),
    )
    health.record_credentials_ok()
    check(
        "凭证校验通过后清零（保留 degraded 之外的状态为 ok）",
        health.to_dict()["consecutive_auth_failures"] == 0
        and health.to_dict()["status"] == runner.PROBE_HEALTH_OK,
        str(health.to_dict()),
    )
    health.record_degraded("boom")
    check(
        "暂时性失败标记为 degraded",
        health.to_dict()["status"] == runner.PROBE_HEALTH_DEGRADED,
        str(health.to_dict()),
    )
    health.record_round({"probed": 3, "ok": 2, "partial": 1, "unprobeable": 0, "failed": 0})
    view = health.to_dict()
    check(
        "完成一轮后回到 ok 并记录本轮计数",
        view["status"] == runner.PROBE_HEALTH_OK
        and view["last_round"]["ok"] == 2
        and "updated_at" in view,
        str(view),
    )
    health.record_no_session("no session.json")
    check(
        "缺少凭证文件为 no_session",
        health.to_dict()["status"] == runner.PROBE_HEALTH_NO_SESSION,
        str(health.to_dict()),
    )

    warned = runner.ProbeHealth()
    for _ in range(runner.PROBE_AUTH_FAILURE_WARN):
        warned.record_auth_rejected("HTTP 401")
    check(
        "达到阈值时只标记一次告警",
        warned.warned and warned.to_dict()["consecutive_auth_failures"] == 3,
        str(warned.to_dict()),
    )
    warned.record_auth_rejected("HTTP 401")
    check(
        "继续失败不重复告警",
        warned.warned and warned.to_dict()["consecutive_auth_failures"] == 4,
    )
    warned.record_round({"probed": 1, "ok": 1, "partial": 0, "unprobeable": 0, "failed": 0})
    check(
        "恢复后清空告警标记与计数",
        not warned.warned and warned.to_dict()["consecutive_auth_failures"] == 0,
    )


def test_body_cap() -> None:
    print("\n--- Request body cap (U-2) / 请求体上限（U-2）---")
    import asyncio

    from starlette.requests import Request

    def make_request(chunks: list, headers: dict) -> Request:
        raw_headers = [
            (str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
            for key, value in headers.items()
        ]
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": raw_headers,
            "query_string": b"",
        }
        pending = list(chunks)

        async def receive() -> dict:
            if not pending:
                return {"type": "http.disconnect"}
            body = pending.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(pending)}

        return Request(scope, receive)

    original = proxy.settings
    try:
        proxy.settings = dataclasses.replace(original, max_body_bytes=1024)

        def read(chunks: list, headers: dict):
            try:
                return asyncio.run(proxy._read_json_body(make_request(chunks, headers)))
            except proxy.HttpError as exc:
                return exc

        declared = read([b"x" * 2048], {"content-length": "2048"})
        check(
            "声明超限的请求体被直接拒收(413)",
            isinstance(declared, proxy.HttpError)
            and declared.status_code == 413
            and declared.error_fields.get("code") == "body_too_large",
            repr(declared),
        )

        chunked = read(
            [b'{"model": "m", "pad": "' + b"x" * 600, b"y" * 600, b'"}'],
            {},
        )
        check(
            "无 Content-Length 的分块请求体同样被拒（无法绕过）",
            isinstance(chunked, proxy.HttpError)
            and chunked.status_code == 413
            and chunked.error_fields.get("code") == "body_too_large",
            repr(chunked),
        )

        ok = read([b'{"model": "m"}'], {"content-length": "13"})
        check("未超限的请求体正常解析", ok == {"model": "m"}, repr(ok))

        invalid = read([b"not json"], {"content-length": "8"})
        check(
            "非 JSON 仍然 400",
            isinstance(invalid, proxy.HttpError) and invalid.status_code == 400,
            repr(invalid),
        )
    finally:
        proxy.settings = original


def test_upstream_hardening() -> None:
    print("\n--- Upstream hardening: redirects & credential headers (U-3, U-4) ---")
    import asyncio

    import httpx
    import upstream as upstream_module

    class _Resp:
        def __init__(self, status_code: int, headers: dict):
            self.status_code = status_code
            self.headers = httpx.Headers(headers)
            self.text = json.dumps({"detail": "nope"})
            self.is_closed = False

        async def aclose(self) -> None:
            self.is_closed = True

        def json(self) -> dict:
            return json.loads(self.text)

    class _Client:
        """Records every request; answers with the queued responses."""

        def __init__(self, responses: list):
            self.responses = list(responses)
            self.requests: list = []
            self.is_closed = False
            self._builder = httpx.Client()

        def build_request(self, method, url, **kwargs):
            return self._builder.build_request(method, url, **kwargs)

        async def send(self, request, stream: bool = False):
            self.requests.append(request)
            return self.responses.pop(0)

        async def aclose(self) -> None:
            self.is_closed = True

    base_settings = dataclasses.replace(
        config_module.settings,
        open_webui_base_url="http://upstream.test",
        upstream_api_style="auto",
    )
    session = store.Session(authorization="Bearer abc", user_agent="ua")

    redirect_target = "http://attacker.example.com/stolen"
    redirect_client = upstream_module.UpstreamClient(base_settings)
    redirect_client.prefix = "/api/v1"
    fake = _Client(
        [
            _Resp(302, {"content-type": "text/html", "location": redirect_target}),
            _Resp(200, {"content-type": "application/json"}),
        ]
    )
    redirect_client._client = fake  # type: ignore[assignment]
    message = ""
    try:
        asyncio.run(redirect_client.post(session, "chat/completions", {"model": "m"}))
    except upstream_module.UpstreamUnavailable as exc:
        message = str(exc)
    check("3xx 被视为上游故障（UpstreamUnavailable）", bool(message), message[:160])
    check(
        "3xx 之后没有第二次请求（凭证没有被重发）",
        len(fake.requests) == 1,
        f"requests={len(fake.requests)}",
    )
    check(
        "错误文案点名 Location 目标，便于直接修配置",
        redirect_target in message,
        message[:200],
    )

    class _HeadersOnly:
        def __init__(self, headers: dict):
            self.headers = httpx.Headers(headers)

    pairs = upstream_module.UpstreamClient.forward_headers(
        _HeadersOnly(
            {
                "content-type": "application/json",
                "set-cookie": "token=upstream-session; Path=/; HttpOnly",
                "set-cookie2": "legacy=1",
                "www-authenticate": "Basic realm=x",
                "x-custom": "keep-me",
                "date": "Mon, 01 Jan 2026 00:00:00 GMT",
            }
        )
    )
    names = {name for name, _ in pairs}
    check(
        "U-4: 上游的凭证类响应头被剔除",
        "set-cookie" not in names
        and "set-cookie2" not in names
        and "www-authenticate" not in names,
        str(sorted(names)),
    )
    check(
        "U-4: 普通响应头仍然透传",
        "x-custom" in names and "content-type" not in names and "date" not in names,
        str(sorted(names)),
    )


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
    test_probe_parameter_loop()
    test_review_fixes()
    test_review_fixes_round2()
    test_review_fixes_round3()
    test_language_keys()
    test_created_timestamp_and_summaries()
    test_base_url_acceptance_and_http_notice()
    test_passthrough_allow_config()
    test_named_proxy_keys()
    test_request_id()
    test_probe_health_state()
    test_body_cap()
    test_upstream_hardening()

    total = len(PASSED) + len(FAILED)
    print("\n" + "=" * 60)
    print(f"Passed {len(PASSED)}/{total} / 通过 {len(PASSED)}/{total}")
    if FAILED:
        print("Failed items / 失败项：")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    print("All passed / 全部通过")
