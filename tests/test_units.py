"""
Pure-logic unit tests: no Playwright, no network required.

Run from the project root:
    python tests/test_units.py

纯逻辑单元测试，不依赖 Playwright，也不需要联网。

运行方式（项目根目录）：
    python tests/test_units.py
"""

from __future__ import annotations

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
    check("私有字段被剔除", set(full.keys()) == {"id", "object", "created", "owned_by"}, str(full))

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
                    "capabilities": {"vision": True, "usage": False, "bad": "x", "builtin_tools": True},
                },
                "access_grants": [{"principal_id": "*"}],
            },
            "urlIdx": 3,
            "permission": [],
            "openai": {"owned_by": "vllm", "max_model_len": 999},
        }
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
    check(
        "扩展白名单透出（含派生 function_calling）",
        extended.get("description") == "A test model"
        and extended.get("capabilities") == {"vision": True, "usage": False, "builtin_tools": True, "function_calling": True},
        str(extended),
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


if __name__ == "__main__":
    test_session_roundtrip()
    test_session_file()
    test_login_signal()
    test_model_normalization()
    test_model_list_extraction()
    test_config()
    test_error_shape()
    test_credentials_are_valid()
    test_language_detection()

    total = len(PASSED) + len(FAILED)
    print("\n" + "=" * 60)
    print(f"Passed {len(PASSED)}/{total} / 通过 {len(PASSED)}/{total}")
    if FAILED:
        print("Failed items / 失败项：")
        for item in FAILED:
            print(f"  - {item}")
        sys.exit(1)
    print("All passed / 全部通过 ✅")
