"""
End-to-end smoke tests: mock upstream + real proxy startup, verifying OpenAI-compatible behavior.

Run from the project root:
    python tests/test_smoke.py

端到端冒烟测试：mock 上游 + 真实启动本代理，验证 OpenAI 兼容行为。

运行方式（项目根目录）：
    python tests/test_smoke.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_openwebui import (  # noqa: E402
    ABORT_MODEL,
    BROKEN_PATHS,
    ERROR_MODEL,
    GZIP_MODEL,
    MODELS,
    NO_TOOLS_MODEL,
    REASONING_MODEL,
    TEXT_ONLY_MODEL,
    VALID_TOKEN,
)

# A Windows console whose code page cannot represent every character the script prints
# (cp936 here) would otherwise abort the run with UnicodeEncodeError *after* the last
# check, reporting a failure that never happened.
#
# 在无法表示全部输出字符的 Windows 控制台（此处 cp936）上，若不做处理，脚本会在最后一个
# 检查之后抛 UnicodeEncodeError 中止，报告一个并不存在的失败。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

PROXY_KEY = "sk-test-proxy-key"
PASSED: list = []
FAILED: list = []


# --------------------------------------------------------------------------- #
# Assertions & utilities
# 断言与工具
# --------------------------------------------------------------------------- #
def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [ok]   {name}")
    else:
        FAILED.append(f"{name}{(' -> ' + detail) if detail else ''}")
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_ready(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0, trust_env=False)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.2)
    return False


class ProxyProcess:
    # Start the proxy as a subprocess, matching the real deployment path exactly.
    # 以子进程方式启动代理，确保和真实部署路径一致。

    def __init__(
        self,
        upstream_url: str,
        session_file: Optional[Path],
        style: str,
        extra_env: Optional[Dict[str, str]] = None,
    ):
        self.port = free_port()
        # Every per-run artifact lives in this run's own directory: the proxy must
        # never write its session, probe cache or log into the checkout.
        #
        # 每次运行的所有产物都放在本次运行自己的目录里：代理绝不能把凭证、探测缓存
        # 或日志写进代码仓库。
        self.workdir = Path(tempfile.mkdtemp(prefix="owui-proxy-"))
        self.log_file = self.workdir / "proxy.log"
        self.cache_file = self.workdir / "model_probe_cache.json"
        self.base = f"http://127.0.0.1:{self.port}"

        env = os.environ.copy()
        env.update(
            {
                "OPEN_WEBUI_BASE_URL": upstream_url,
                "PROXY_HOST": "127.0.0.1",
                "PROXY_PORT": str(self.port),
                "PROXY_API_KEY": PROXY_KEY,
                "UPSTREAM_API_STYLE": style,
                # Loopback test: must bypass the system proxy
                # 本地回环测试，必须绕过系统代理
                "UPSTREAM_TRUST_ENV": "false",
                "SESSION_FILE": str(session_file) if session_file else str(self.workdir / "missing-session.json"),
                "MODEL_PROBE_CACHE_FILE": str(self.cache_file),
                "LOG_LEVEL": "INFO",
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONUNBUFFERED": "1",
            }
        )
        if extra_env:
            env.update(extra_env)
        self._log = open(self.log_file, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def wait(self) -> bool:
        return wait_ready(f"{self.base}/healthz")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()

    def dump_log(self) -> str:
        try:
            return self.log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            return "<no log / 无日志>"


def headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {PROXY_KEY}"}


def make_session(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "Authorization": f"Bearer {VALID_TOKEN}",
                "Cookie": "",
                "User-Agent": "smoke-test-agent",
                "captured_at": time.time(),
                "base_url": "mock",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# Test cases
# 测试项
# --------------------------------------------------------------------------- #
def run_case(server_url: str, style: str, expected_prefix: str) -> None:
    print(f"\n=== Scenario: UPSTREAM_API_STYLE={style} / 场景：UPSTREAM_API_STYLE={style} ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="owui-session-"))
    session_file = make_session(tmp_dir / "session.json")

    proxy = ProxyProcess(server_url, session_file, style)
    try:
        if not proxy.wait():
            check(f"[{style}] 代理启动", False, proxy.dump_log())
            return
        check(f"[{style}] 代理启动", True)

        client = httpx.Client(base_url=proxy.base, timeout=30.0, trust_env=False)

        # 1. healthz: usable without auth, but the upstream address is only returned with a key
        # 1. healthz：免鉴权可用，但上游地址只在带 Key 时返回
        resp = client.get("/healthz")
        body = resp.json()
        check(f"[{style}] /healthz 无 Key 也可用", resp.status_code == 200, str(body))
        check(
            f"[{style}] 无 Key 时隐藏上游地址",
            "upstream" not in body and "upstream_prefix" not in body,
            str(body),
        )
        check(f"[{style}] 无 Key 时报告需要鉴权", body.get("auth_required") is True, str(body))

        resp = client.get("/healthz", headers=headers())
        body = resp.json()
        check(f"[{style}] 带 Key 的 /healthz 返回 200", resp.status_code == 200, str(body))
        check(
            f"[{style}] 带 Key 时前缀探测为 {expected_prefix}",
            body.get("upstream_prefix") == expected_prefix,
            str(body),
        )
        check(f"[{style}] 带 Key 时报告凭证已就绪", body.get("session_ready") is True, str(body))

        # 1b. / meta endpoint likewise only exposes the upstream address with a key
        # 1b. / 元信息同样"带 Key 才暴露上游地址"
        resp = client.get("/")
        body = resp.json()
        check(f"[{style}] / 无 Key 时隐藏上游地址", "upstream" not in body, str(body))
        resp = client.get("/", headers=headers())
        body = resp.json()
        check(
            f"[{style}] / 带 Key 时返回上游地址",
            body.get("upstream") == server_url and body.get("upstream_prefix") == expected_prefix,
            str(body),
        )

        # 1c. CORS is off by default: no ACAO header when PROXY_CORS_ORIGINS is not configured
        # 1c. 默认不启用 CORS：未配置 PROXY_CORS_ORIGINS 时不应出现 ACAO 头
        resp = client.get("/v1/models", headers={**headers(), "Origin": "http://evil.example.com"})
        check(
            f"[{style}] 默认不启用 CORS",
            "access-control-allow-origin" not in resp.headers,
            str(dict(resp.headers)),
        )

        # 2. Authentication
        # 2. 鉴权
        resp = client.get("/v1/models")
        check(f"[{style}] 缺少 API Key 返回 401", resp.status_code == 401, str(resp.text[:200]))
        check(
            f"[{style}] 错误体为 OpenAI 风格",
            resp.json().get("error", {}).get("code") == "invalid_api_key",
            resp.text[:200],
        )

        # 3. Model list
        # 3. 模型列表
        resp = client.get("/v1/models", headers=headers())
        body = resp.json()
        check(f"[{style}] /v1/models 返回 200", resp.status_code == 200, str(body)[:200])
        check(f"[{style}] 顶层 object=list", body.get("object") == "list", str(body)[:200])
        data = body.get("data") or []
        check(f"[{style}] 模型条目完整", len(data) == len(MODELS["data"]), str(len(data)))
        first = data[0] if data else {}
        check(
            f"[{style}] 模型字段符合 OpenAI 结构",
            set(first.keys()) >= {"id", "object", "created", "owned_by"},
            str(list(first.keys())),
        )
        check(f"[{style}] 已剔除上游私有字段 info", "info" not in first and "params" not in first, str(list(first.keys())))
        legacy = next((m for m in data if m.get("id") == "legacy-model"), {})
        check(
            f"[{style}] 补齐缺失的 object/created/owned_by",
            legacy.get("object") == "model" and legacy.get("owned_by") == "openai" and legacy.get("created") == 0,
            str(legacy),
        )

        # 4. Non-streaming chat
        # 4. 非流式对话
        resp = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": "llama3:latest", "messages": [{"role": "user", "content": "hi"}]},
        )
        body = resp.json()
        check(f"[{style}] 非流式对话返回 200", resp.status_code == 200, str(body)[:200])
        check(
            f"[{style}] 返回 assistant 内容",
            (body.get("choices") or [{}])[0].get("message", {}).get("content") == "hello from mock",
            str(body)[:200],
        )

        # 4b. Reasoning model: thinking fields like reasoning_content must pass through losslessly (non-streaming)
        # 4b. 推理模型：reasoning_content 等思考字段必须无损透传（非流式）
        resp = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={
                "model": REASONING_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        body = resp.json()
        message = (body.get("choices") or [{}])[0].get("message", {})
        check(
            f"[{style}] reasoning_content 无损透传（非流式）",
            message.get("reasoning_content") == "这是思考过程：先分析问题再回答。",
            str(body)[:200],
        )
        check(
            f"[{style}] 请求参数 reasoning_effort 原样到达上游",
            body.get("extra", {}).get("reasoning_effort") == "high",
            str(body)[:200],
        )

        # 4c. Reasoning chunks under streaming are forwarded as-is too
        # 4c. 流式下的 reasoning chunk 同样原样转发
        reasoning_text = ""
        try:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers=headers(),
                json={"model": REASONING_MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
            ) as stream:
                for line in stream.iter_lines():
                    reasoning_text += line + "\n"
        except Exception as exc:
            reasoning_text = f"{type(exc).__name__}: {exc}"
        check(
            f"[{style}] reasoning_content 无损透传（流式）",
            "思考第一段" in reasoning_text and "思考第二段" in reasoning_text,
            reasoning_text[:200],
        )

        # 5. Streaming chat
        # 5. 流式对话
        chunks: list = []
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers(),
            json={"model": "llama3:latest", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as stream:
            check(
                f"[{style}] 流式响应 content-type 为 SSE",
                "text/event-stream" in stream.headers.get("content-type", ""),
                stream.headers.get("content-type", ""),
            )
            check(
                f"[{style}] 设置 X-Accel-Buffering",
                stream.headers.get("x-accel-buffering") == "no",
                str(dict(stream.headers)),
            )
            for line in stream.iter_lines():
                chunks.append(line)
        text = "\n".join(chunks)
        check(f"[{style}] 流式收到多个 data 块", text.count("data: ") >= 3, text[:300])
        check(f"[{style}] 流式内容拼接正确", "Hel" in text and "lo from mock" in text, text[:300])
        check(f"[{style}] 流式以 [DONE] 结束", "[DONE]" in text, text[-200:])

        # 6. Upstream error passthrough
        # 6. 上游错误透传
        resp = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": ERROR_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        body = resp.json()
        check(f"[{style}] 上游错误返回 4xx", resp.status_code == 400, str(body)[:200])
        check(f"[{style}] 上游错误体为 OpenAI 风格", "error" in body, str(body)[:200])

        # 7. Parameter validation
        # 7. 参数校验
        resp = client.post("/v1/chat/completions", headers=headers(), json={"messages": []})
        check(f"[{style}] 缺少 model 返回 400", resp.status_code == 400, resp.text[:200])
        resp = client.post(
            "/v1/chat/completions",
            headers={**headers(), "Content-Type": "application/json"},
            content=b"not json",
        )
        check(f"[{style}] 非法 JSON 返回 400", resp.status_code == 400, resp.text[:200])

        # A non-string model must be answered 400, never 500: the MODEL_ALIASES lookup
        # raises TypeError on an unhashable value.
        #
        # 非字符串的 model 必须得到 400 而不是 500：别名映射对不可哈希的值会抛 TypeError。
        resp = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={"model": {"not": "a string"}, "messages": [{"role": "user", "content": "hi"}]},
        )
        check(
            f"[{style}] model 非字符串返回 400（不是 500）",
            resp.status_code == 400 and resp.json().get("error", {}).get("code") == "invalid_type",
            f"{resp.status_code} {resp.text[:160]}",
        )
        resp = client.post(
            "/v1/embeddings", headers=headers(), json={"model": ["a"], "input": ["hello"]}
        )
        check(
            f"[{style}] embeddings 的 model 非字符串返回 400",
            resp.status_code == 400 and resp.json().get("error", {}).get("code") == "invalid_type",
            f"{resp.status_code} {resp.text[:160]}",
        )

        # 8. Upstream stream cut mid-way: must not surface as 500, and content already
        #    received must be preserved
        #
        # 8. 上游流式中途断开：不能穿透成 500，已收到的内容要保住
        partial = ""
        aborted_cleanly = True
        try:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers=headers(),
                json={"model": ABORT_MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
            ) as stream:
                for raw in stream.iter_raw():
                    partial += raw.decode("utf-8", errors="replace")
        except Exception as exc:
            # any exception here means it leaked through to the client
            # 任何异常都说明异常穿透到了客户端
            aborted_cleanly = False
            partial = f"{type(exc).__name__}: {exc}"
        check(f"[{style}] 上游断流不抛异常给客户端", aborted_cleanly, partial[:200])
        check(f"[{style}] 保住断流前已收到的内容", "partial" in partial, partial[:200])

        # 9. Embeddings
        resp = client.post(
            "/v1/embeddings", headers=headers(), json={"model": "llama3:latest", "input": ["hello"]}
        )
        body = resp.json()
        check(f"[{style}] /v1/embeddings 返回 200", resp.status_code == 200, str(body)[:200])
        check(f"[{style}] embeddings 结构正确", body.get("data", [{}])[0].get("object") == "embedding", str(body)[:200])

        # 9b. Upstream errors from embeddings must also be OpenAI-style
        # 9b. embeddings 上游报错也要是 OpenAI 风格
        resp = client.post(
            "/v1/embeddings", headers=headers(), json={"model": ERROR_MODEL, "input": ["hello"]}
        )
        if resp.status_code >= 400:
            body = resp.json()
            check(f"[{style}] embeddings 错误体为 OpenAI 风格", "error" in body, resp.text[:200])

        # 9. Catch-all passthrough (the mock implements /responses only under /api/v1)
        # 9. 兜底透传（mock 只在 /api/v1 下实现了 /responses）
        resp = client.post("/v1/responses", headers=headers(), json={"model": "llama3:latest"})
        if style == "legacy":
            check(f"[{style}] 未知路由透传返回 404", resp.status_code == 404, resp.text[:200])
        else:
            check(f"[{style}] 未知路由透传成功", resp.status_code == 200 and resp.json().get("id") == "resp_mock", resp.text[:200])

        # 9b. Prefix fallback of the catch-all passthrough: /legacy-only exists only
        #     under the legacy /api prefix
        #   auto   - probes /api/v1, falls back to /api on 404 and hits
        #   legacy - the prefix is /api already, hits directly
        #   v1     - only /api/v1 is a candidate, nowhere to fall back, 404
        #
        # 9b. 兜底透传的前缀回退：/legacy-only 只存在于旧版 /api 前缀下
        #   auto   —— 探测到 /api/v1，404 后回退 /api 命中
        #   legacy —— 前缀本来就是 /api，直接命中
        #   v1     —— 只有 /api/v1 一个候选，无路可退，404
        resp = client.post("/v1/legacy-only", headers=headers(), json={"model": "llama3:latest"})
        if style == "v1":
            check(f"[{style}] 透传无回退候选时返回 404", resp.status_code == 404, resp.text[:200])
        else:
            check(
                f"[{style}] 透传 404 时回退到另一前缀",
                resp.status_code == 200 and resp.json().get("route") == "legacy-only",
                resp.text[:200],
            )

        # 10. Upstream compression: the proxy drops Content-Encoding, so it must forward
        #     the *decoded* body -- httpx advertises gzip by default, and a gzipped body
        #     forwarded raw would be unreadable for the client.
        #
        # 10. 上游压缩：代理会丢掉 Content-Encoding，因此必须转发已解码的响应体——
        #     httpx 默认声明接受 gzip，把压缩字节原样转发会让客户端无法解析。
        gzip_passthrough_ok = False
        try:
            resp = client.post("/v1/gzipped", headers=headers(), json={"model": "llama3:latest"})
            payload = resp.json() or {}
            gzip_passthrough_ok = resp.status_code == 200 and payload.get("route") == "gzipped"
            gzip_passthrough_detail = (
                f"{resp.status_code} upstream_compressed={payload.get('compressed')} "
                f"content-encoding={resp.headers.get('content-encoding')!r} {resp.content[:24]!r}"
            )
        except Exception as exc:
            gzip_passthrough_detail = f"{type(exc).__name__}: {exc}"
        check(
            f"[{style}] 上游 gzip 的透传响应解码后可解析",
            gzip_passthrough_ok,
            gzip_passthrough_detail,
        )

        gzip_chat_ok = False
        try:
            resp = client.post(
                "/v1/chat/completions",
                headers=headers(),
                json={"model": GZIP_MODEL, "messages": [{"role": "user", "content": "hi"}]},
            )
            gzip_chat_ok = (
                resp.status_code == 200
                and (resp.json() or {}).get("choices", [{}])[0]
                .get("message", {})
                .get("content")
                == "hello gzip"
            )
            gzip_chat_detail = f"{resp.status_code} {resp.content[:40]!r}"
        except Exception as exc:
            gzip_chat_detail = f"{type(exc).__name__}: {exc}"
        check(
            f"[{style}] 上游 gzip 的非流式对话解码后可解析",
            gzip_chat_ok,
            gzip_chat_detail,
        )

        gzip_stream_text = ""
        try:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers=headers(),
                json={
                    "model": GZIP_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ) as stream:
                for line in stream.iter_lines():
                    gzip_stream_text += line + "\n"
        except Exception as exc:
            gzip_stream_text = f"{type(exc).__name__}: {exc}"
        check(
            f"[{style}] 上游 gzip 的流式对话解码后可读",
            "[DONE]" in gzip_stream_text
            and '"content":"gzip "' in gzip_stream_text
            and '"content":"stream"' in gzip_stream_text,
            gzip_stream_text[:200],
        )

        client.close()
    finally:
        proxy.stop()


def run_missing_session_case(server_url: str) -> None:
    print("\n=== Scenario: missing session file / 场景：凭证文件缺失 ===")
    proxy = ProxyProcess(server_url, None, "auto")
    try:
        if not proxy.wait():
            check("[no-session] 代理启动", False, proxy.dump_log())
            return
        client = httpx.Client(base_url=proxy.base, timeout=10.0, trust_env=False)
        resp = client.get("/v1/models", headers=headers())
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        check("[no-session] 返回 503", resp.status_code == 503, resp.text[:200])
        check(
            "[no-session] 错误码为 session_missing",
            body.get("error", {}).get("code") == "session_missing",
            resp.text[:200],
        )
        client.close()
    finally:
        proxy.stop()


def run_bad_config_case() -> None:
    print("\n=== Scenario: invalid OPEN_WEBUI_BASE_URL (missing scheme) / 场景：非法 OPEN_WEBUI_BASE_URL（缺 scheme）===")
    env = os.environ.copy()
    env.update(
        {
            "OPEN_WEBUI_BASE_URL": "localhost:8080",
            # missing scheme; load_settings must reject it
            # 缺 scheme，load_settings 应拒绝
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONIOENCODING": "utf-8",
            # load_dotenv does not override existing env vars; setting values explicitly
            # here is enough to shield the local .env influence
            #
            # load_dotenv 不覆盖已有环境变量，这里显式给值即可屏蔽本机 .env 影响
            "UPSTREAM_TRUST_ENV": "false",
        }
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        check("非法 URL 触发快速退出", False, f"import app 超时: {exc}")
        return
    output = (proc.stdout or "") + (proc.stderr or "")
    check(
        "非法 URL 以退出码 2 终止",
        proc.returncode == 2,
        f"returncode={proc.returncode} {output[-300:]}",
    )
    check(
        "非法 URL 报错友好（无 traceback）",
        "Traceback" not in output
        and ("启动失败" in output or "Startup failed" in output),
        output[-300:],
    )


def run_cors_case(server_url: str) -> None:
    print("\n=== Scenario: CORS (PROXY_CORS_ORIGINS set) / 场景：CORS（PROXY_CORS_ORIGINS 配置了具体来源）===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="owui-session-"))
    session_file = make_session(tmp_dir / "session.json")
    proxy = ProxyProcess(
        server_url,
        session_file,
        "v1",
        extra_env={"PROXY_CORS_ORIGINS": "http://example.com"},
    )
    try:
        if not proxy.wait():
            check("[cors] 代理启动", False, proxy.dump_log())
            return
        client = httpx.Client(
            base_url=proxy.base,
            timeout=10.0,
            trust_env=False,
            headers={"Origin": "http://example.com"},
        )

        # A simple request (GET) should carry Access-Control-Allow-Origin
        # 简单请求（GET）应带 Access-Control-Allow-Origin
        resp = client.get("/healthz")
        check(
            "[cors] GET 响应带 Access-Control-Allow-Origin",
            resp.headers.get("access-control-allow-origin") == "http://example.com",
            str(dict(resp.headers)),
        )

        # Preflight (OPTIONS) should be answered by the middleware before routing,
        # and must allow the Authorization header
        #
        # 预检（OPTIONS）应由中间件在路由之前直接应答，且放行 Authorization
        resp = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization, content-type",
            },
        )
        check("[cors] 预检返回 200", resp.status_code == 200, f"{resp.status_code} {resp.text[:200]}")
        check(
            "[cors] 预检放行 Authorization 头",
            "authorization" in resp.headers.get("access-control-allow-headers", "").lower(),
            str(dict(resp.headers)),
        )

        # Non-whitelisted origins must not be allowed
        # 非白名单来源不应放行
        resp = client.get("/healthz", headers={"Origin": "http://evil.example.com"})
        check(
            "[cors] 非白名单来源不带 ACAO",
            "access-control-allow-origin" not in resp.headers,
            str(dict(resp.headers)),
        )

        client.close()
    finally:
        proxy.stop()


def run_prefix_5xx_case(server_url: str) -> None:
    print("\n=== Scenario: primary prefix returns 5xx / 场景：主候选前缀返回 5xx ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="owui-session-"))
    session_file = make_session(tmp_dir / "session.json")
    # Make the first candidate (/api/v1) fail with a 5xx while the legacy /api stays
    # healthy: the proxy must not settle on a prefix it could not confirm, otherwise
    # every request dies on a 502 (the fallback only kicks in on 404).
    #
    # 让首候选（/api/v1）返回 5xx、旧版 /api 保持健康：代理不得停在一个无法确认的前缀上，
    # 否则每个请求都会以 502 失败（回退只对 404 生效）。
    BROKEN_PATHS.add("/api/v1/models")
    proxy = ProxyProcess(server_url, session_file, "auto")
    try:
        if not proxy.wait():
            check("[5xx] 代理启动", False, proxy.dump_log())
            return
        client = httpx.Client(base_url=proxy.base, timeout=20.0, trust_env=False)

        body = client.get("/healthz", headers=headers()).json() or {}
        check(
            "[5xx] 5xx 候选不被当作可用前缀（回退到 /api）",
            body.get("upstream_prefix") == "/api",
            str(body),
        )

        resp = client.get("/v1/models", headers=headers())
        try:
            data = (resp.json() or {}).get("data") or []
            fallback_ok = resp.status_code == 200 and len(data) == len(MODELS["data"])
            fallback_detail = f"{resp.status_code} models={len(data)}"
        except Exception as exc:
            fallback_ok = False
            fallback_detail = f"{type(exc).__name__}: {exc} {resp.text[:120]}"
        check("[5xx] 回退前缀下 /v1/models 依然可用", fallback_ok, fallback_detail)

        client.close()
    finally:
        BROKEN_PATHS.discard("/api/v1/models")
        proxy.stop()


def run_probe_case(server_url: str) -> None:
    print("\n=== Scenario: per-model probe & cache / 场景：逐模型探测与缓存 ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="owui-probe-"))
    session_file = make_session(tmp_dir / "session.json")
    proxy = ProxyProcess(server_url, session_file, "v1")
    # ProxyProcess points the probe cache at this run's own directory, so the smoke
    # tests never touch the checkout's model_probe_cache.json.
    #
    # ProxyProcess 已把探测缓存指向本次运行自己的目录，因此冒烟测试不会碰到仓库里的
    # model_probe_cache.json。
    cache_file = proxy.cache_file
    try:
        if not proxy.wait():
            check("[probe] 代理启动", False, proxy.dump_log())
            return
        client = httpx.Client(base_url=proxy.base, timeout=60.0, trust_env=False)

        # The startup refresh runs in the background; poll until the probe lands.
        # 启动时的刷新在后台进行；轮询直到探测结果落进 /v1/models。
        models = {}
        deadline = time.time() + 60
        while time.time() < deadline:
            body = client.get("/v1/models", headers=headers()).json() or {}
            models = {model.get("id"): model for model in body.get("data") or []}
            if (models.get("llama3:latest") or {}).get("capabilities"):
                break
            time.sleep(0.3)

        body = client.get("/v1/models", headers=headers()).json() or {}
        models = {model.get("id"): model for model in body.get("data") or []}
        llama = models.get("llama3:latest") or {}

        # --- reasoning efforts -------------------------------------------------
        llama_reasoning = llama.get("reasoning")
        check("[probe] 探测完成后 /v1/models 带 reasoning 字段", llama_reasoning is not None, str(llama))
        if llama_reasoning:
            check(
                "[probe] 两层一致的模型枚举完整",
                llama_reasoning.get("supported_efforts")
                == ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                str(llama_reasoning),
            )
            check(
                "[probe] 含 none 时非强制思考",
                llama_reasoning.get("mandatory") is False,
                str(llama_reasoning),
            )
            check(
                "[probe] 引擎未声明默认挡位时不猜",
                "default_effort" not in llama_reasoning,
                str(llama_reasoning),
            )

        # The regression that motivated this design: the outer schema advertises seven
        # levels, the model's own parser accepts four, and its error text mentions only
        # three of them. Only per-value verification gets this right.
        #
        # 促成这套设计的回归用例：外层 schema 广告 7 个挡位，模型自带解析器只接受 4 个，
        # 而它的报错文本只提到其中 3 个。只有逐值实证才能得到正确答案。
        legacy_reasoning = (models.get("legacy-model") or {}).get("reasoning") or {}
        check(
            "[probe] 两层不一致时按实证结果而非报错文本枚举",
            legacy_reasoning.get("supported_efforts") == ["none", "low", "medium", "xhigh"],
            str(legacy_reasoning),
        )
        check(
            "[probe] 从第二层报错里提取到 (default) 标注",
            legacy_reasoning.get("default_effort") == "xhigh",
            str(legacy_reasoning),
        )

        unprobeable = models.get(REASONING_MODEL) or {}
        check(
            "[probe] 上游不校验哨兵值的模型不带 reasoning 字段",
            "reasoning" not in unprobeable,
            str(unprobeable),
        )
        check(
            "[probe] 不校验挡位也能得出能力与默认思考",
            unprobeable.get("capabilities", {}).get("vision") is True
            and (unprobeable.get("reasoning") is None)
            and unprobeable.get("supported_parameters") is not None,
            str(unprobeable),
        )

        # --- capabilities: probed, not echoed from the default template ---------
        no_tools = (models.get(NO_TOOLS_MODEL) or {}).get("capabilities") or {}
        check(
            "[probe] 无 tool-call parser 的引擎 function_calling=false",
            no_tools.get("function_calling") is False,
            str(no_tools),
        )
        text_only = (models.get(TEXT_ONLY_MODEL) or {}).get("capabilities") or {}
        check(
            "[probe] 非多模态引擎 vision=false",
            text_only.get("vision") is False,
            str(text_only),
        )
        check(
            "[probe] capabilities 只含实证键",
            set(text_only.keys()) <= {"vision", "function_calling", "reasoning", "structured_outputs"},
            str(text_only),
        )
        check(
            "[probe] 模态由视觉探测推导",
            (models.get(TEXT_ONLY_MODEL) or {}).get("architecture")
            == {"modality": "text->text", "input_modalities": ["text"], "output_modalities": ["text"]},
            str((models.get(TEXT_ONLY_MODEL) or {}).get("architecture")),
        )
        check(
            "[probe] 参数支持声明来自实证",
            isinstance(no_tools and (models.get(NO_TOOLS_MODEL) or {}).get("supported_parameters"), list)
            and "tools" not in ((models.get(NO_TOOLS_MODEL) or {}).get("supported_parameters") or []),
            str((models.get(NO_TOOLS_MODEL) or {}).get("supported_parameters")),
        )

        # --- instance metadata moved out of the model capability field ----------
        instance = body.get("x_open_webui") or {}
        check(
            "[probe] 实例级功能开关放在信封的 x_open_webui",
            instance.get("features", {}).get("enable_web_search") is False
            and bool(instance.get("default_model_capabilities")),
            str(instance)[:200],
        )
        check(
            "[probe] 不再把上游默认能力模板当成模型能力",
            "web_search" not in (llama.get("capabilities") or {})
            and "builtin_tools" not in (llama.get("capabilities") or {}),
            str(llama.get("capabilities")),
        )

        # --- retrieve a single model -------------------------------------------
        resp = client.get("/v1/models/llama3:latest", headers=headers())
        check(
            "[probe] GET /v1/models/{id} 返回该模型",
            resp.status_code == 200 and (resp.json() or {}).get("id") == "llama3:latest",
            f"{resp.status_code} {resp.text[:160]}",
        )
        resp = client.get("/v1/models/does-not-exist", headers=headers())
        check(
            "[probe] 未知模型 id 返回 404 JSON 错误体",
            resp.status_code == 404
            and resp.json().get("error", {}).get("code") == "model_not_found",
            f"{resp.status_code} {resp.text[:160]}",
        )

        # --- a live 400 disproves an advertised level and heals the cache -------
        resp = client.post(
            "/v1/chat/completions",
            headers=headers(),
            json={
                "model": "legacy-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
        )
        check(
            "[probe] 上游 400 原样透出（客户端行为不变）",
            resp.status_code == 400
            and resp.json().get("error", {}).get("code") == "upstream_error"
            and "Unexpected reasoning effort high" in resp.text,
            f"{resp.status_code} {resp.text[:160]}",
        )
        healed = None
        deadline = time.time() + 30
        while time.time() < deadline:
            body = client.get("/v1/models", headers=headers()).json() or {}
            entry = next((m for m in body.get("data") or [] if m.get("id") == "legacy-model"), {})
            healed = (entry.get("reasoning") or {}).get("supported_efforts")
            if healed and "high" not in healed:
                break
            time.sleep(0.3)
        check(
            "[probe] 线上 400 之后不再广告该挡位",
            healed is not None and "high" not in healed,
            str(healed),
        )

        check(
            "[probe] 缓存文件已生成（v2）",
            cache_file.exists() and '"version": 2' in (cache_file.read_text(encoding="utf-8") or ""),
        )

        client.close()
    finally:
        proxy.stop()


def main() -> int:
    run_bad_config_case()

    from mock_openwebui import MockOpenWebUI

    server = MockOpenWebUI()
    server.start()
    print(f"mock Open WebUI listening on {server.base_url} / mock Open WebUI 监听于 {server.base_url}")
    try:
        run_case(server.base_url, "auto", "/api/v1")
        run_case(server.base_url, "v1", "/api/v1")
        run_case(server.base_url, "legacy", "/api")
        run_missing_session_case(server.base_url)
        run_cors_case(server.base_url)
        run_prefix_5xx_case(server.base_url)
        run_probe_case(server.base_url)
    finally:
        server.stop()

    total = len(PASSED) + len(FAILED)
    print("\n" + "=" * 60)
    print(f"Passed {len(PASSED)}/{total} / 通过 {len(PASSED)}/{total}")
    if FAILED:
        print("Failed items / 失败项：")
        for item in FAILED:
            print(f"  - {item}")
        return 1
    print("All passed / 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
