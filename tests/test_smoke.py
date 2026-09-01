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

from mock_openwebui import ABORT_MODEL, ERROR_MODEL, REASONING_MODEL, VALID_TOKEN  # noqa: E402

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
        self.log_file = Path(tempfile.mkdtemp(prefix="owui-proxy-")) / "proxy.log"
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
                "SESSION_FILE": str(session_file) if session_file else str(Path(tempfile.gettempdir()) / "definitely-missing-session.json"),
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
        check(f"[{style}] 模型条目完整", len(data) == 2, str(len(data)))
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
    print("All passed / 全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
