"""
A minimal Open WebUI upstream simulator, for local smoke tests only.

Implemented with the standard library; no extra dependencies. It simulates the key
behaviors of both upstream versions:
  - Legacy: /api/models, /api/chat/completions
  - Modern: /api/v1/models, /api/v1/chat/completions, /api/v1/embeddings

一个最小化的 Open WebUI 上游模拟器，仅用于本地冒烟测试。

用标准库实现，不引入任何额外依赖。它模拟了上游两个版本的关键行为：
  - 老版本：/api/models、/api/chat/completions
  - 新版本：/api/v1/models、/api/v1/chat/completions、/api/v1/embeddings
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

VALID_TOKEN = "mock-jwt-token"
ERROR_MODEL = "boom-model"
# Requesting this model makes the upstream emit half an SSE stream and then abruptly
# drop the connection; used to verify tolerance of a stream failing mid-way
#
# 请求这个模型时，上游会发半个 SSE 就粗暴断开，用于验证流式中途失败的容错
ABORT_MODEL = "abort-model"
# Reasoning model: the response carries reasoning_content thinking text; used to verify
# the proxy passes extended fields through losslessly
#
# 推理模型：响应带 reasoning_content 思考内容，用于验证代理对扩展字段无损透传
REASONING_MODEL = "reasoning-content-model"

MODELS = {
    "object": "list",
    "data": [
        {
            "id": "llama3:latest",
            "name": "llama3",
            "object": "model",
            "created": 1700000000,
            "owned_by": "ollama",
            "info": {"meta": {"profile_image_url": "/static/x.png", "description": "should be dropped"}},
            "params": {"temperature": 0.7},
            "access_grants": [],
        },
        # Legacy upstreams may lack object / created / owned_by
        # 老版本上游可能缺少 object / created / owned_by
        {"id": "legacy-model", "name": "legacy-model"},
    ],
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockOpenWebUI/1.0"

    # ------------------------------------------------------------------ #
    # Basic utilities
    # 基础工具
    # ------------------------------------------------------------------ #
    # Silence the access log
    # 静音访问日志
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks: list, delay: float = 0.02) -> None:
        # Send SSE with chunked encoding, close to a real streaming response.
        # chunked 编码发送 SSE，贴近真实流式响应。
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in chunks:
            data = chunk.encode("utf-8")
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()
            time.sleep(delay)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _send_sse_then_abort(self) -> None:
        # Send the first SSE chunk then abruptly disconnect, simulating an upstream
        # process crash / a connection being cut.
        #
        # 发出第一个 SSE 分块后立刻粗暴断开，模拟上游进程崩溃 / 连接被掐断。
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        data = b'data: {"id":"chat-1","choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()
        self.close_connection = True
        try:
            self.connection.close()
        except OSError:
            pass

    @staticmethod
    def _authorized(headers: Any) -> bool:
        return headers.get("Authorization") == f"Bearer {VALID_TOKEN}"

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # ------------------------------------------------------------------ #
    # Routes
    # 路由
    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/api/models", "/api/v1/models"):
            if not self._authorized(self.headers):
                self._send_json(401, {"detail": "Not authenticated"})
                return
            self._send_json(200, MODELS)
            return
        if path == "/health":
            self._send_json(200, {"status": True})
            return
        self._send_json(404, {"detail": f"Not Found: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if not self._authorized(self.headers):
            self._send_json(401, {"detail": "Not authenticated"})
            return

        payload = self._read_json_body()
        model = payload.get("model")

        if path in ("/api/chat/completions", "/api/v1/chat/completions"):
            self._handle_chat(payload, model)
            return
        if path in ("/api/embeddings", "/api/v1/embeddings"):
            if model == ERROR_MODEL:
                self._send_json(400, {"detail": "Model is not available"})
                return
            self._send_json(
                200,
                {
                    "object": "list",
                    "model": model,
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                },
            )
            return
        if path == "/api/v1/responses":
            self._send_json(200, {"id": "resp_mock", "object": "response", "model": model})
            return
        if path == "/api/legacy-only":
            # A route that exists only under the legacy /api prefix, used to verify
            # prefix fallback of the catch-all passthrough
            #
            # 只存在于旧版 /api 前缀下的路由，用于验证兜底透传的前缀回退
            self._send_json(200, {"id": "legacy_only_mock", "route": "legacy-only", "model": model})
            return
        self._send_json(404, {"detail": f"Not Found: {path}"})

    def _handle_chat(self, payload: Dict[str, Any], model: Optional[str]) -> None:
        if model == ERROR_MODEL:
            self._send_json(400, {"detail": "Model is not available"})
            return
        if model == ABORT_MODEL:
            self._send_sse_then_abort()
            return

        if model == REASONING_MODEL:
            # Reasoning model: thinking text uses the DeepSeek de-facto standard
            # reasoning_content field, and request parameters are echoed back in
            # "extra", to verify lossless two-way passthrough of request/response.
            #
            # 推理模型：思考内容用 DeepSeek 事实标准的 reasoning_content 字段，
            # 并在 extra 里回显请求参数，用于验证请求/响应双向无损透传。
            if payload.get("stream"):
                self._send_sse(
                    [
                        'data: {"id":"chat-1","object":"chat.completion.chunk","created":1700000000,'
                        f'"model":{json.dumps(model)},"choices":[{{"index":0,"delta":{{"reasoning_content":"思考第一段"}},"finish_reason":null}}]}}\n\n',
                        'data: {"id":"chat-1","object":"chat.completion.chunk","created":1700000000,'
                        f'"model":{json.dumps(model)},"choices":[{{"index":0,"delta":{{"reasoning_content":"思考第二段"}},"finish_reason":null}}]}}\n\n',
                        'data: {"id":"chat-1","object":"chat.completion.chunk","created":1700000000,'
                        f'"model":{json.dumps(model)},"choices":[{{"index":0,"delta":{{"role":"assistant","content":"正式回答"}},"finish_reason":null}}]}}\n\n',
                        "data: [DONE]\n\n",
                    ]
                )
                return
            self._send_json(
                200,
                {
                    "id": "chat-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "hello from mock",
                                "reasoning_content": "这是思考过程：先分析问题再回答。",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
                    "extra": {"reasoning_effort": payload.get("reasoning_effort")},
                },
            )
            return

        if payload.get("stream"):
            self._send_sse(
                [
                    'data: {"id":"chat-1","object":"chat.completion.chunk","created":1700000000,'
                    f'"model":{json.dumps(model)},"choices":[{{"index":0,"delta":{{"role":"assistant","content":"Hel"}},"finish_reason":null}}]}}\n\n',
                    'data: {"id":"chat-1","object":"chat.completion.chunk","created":1700000000,'
                    f'"model":{json.dumps(model)},"choices":[{{"index":0,"delta":{{"content":"lo from mock"}},"finish_reason":null}}]}}\n\n',
                    'data: {"id":"chat-1","object":"chat.completion.chunk","created":1700000000,'
                    f'"model":{json.dumps(model)},"choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}\n\n',
                    "data: [DONE]\n\n",
                ]
            )
            return

        self._send_json(
            200,
            {
                "id": "chat-1",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello from mock"},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            },
        )


class MockOpenWebUI:
    # A mock upstream running in its own thread.
    # 在独立线程里跑起来的 mock 上游。

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "MockOpenWebUI":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


# Handy for manual debugging
# 便于手工调试
if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
    server = MockOpenWebUI(port=port)
    server.start()
    print(f"mock Open WebUI listening on {server.base_url} / mock Open WebUI 监听于 {server.base_url}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.stop()
