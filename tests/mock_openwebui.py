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

import gzip
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Set

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
# A model whose responses are gzip-compressed whenever the client accepts gzip. The
# proxy strips the upstream's Content-Encoding, so it must forward the *decoded* body;
# forwarding the still-compressed bytes would hand the client an unreadable response.
#
# 只要客户端接受 gzip，该模型的响应就带 gzip 压缩。代理会丢掉上游的 Content-Encoding，
# 因此必须转发"已解码"的响应体；若把仍然压缩的字节直接发出去，客户端将无法解析。
GZIP_MODEL = "gzip-model"
# Paths answering HTTP 500, used to verify that a 5xx candidate is not mistaken for "the
# route exists": the proxy must keep probing the other candidate prefixes.
#
# 返回 HTTP 500 的路径，用于验证 5xx 候选不会被当成"路由存在"：代理必须继续探测其它候选前缀。
BROKEN_PATHS: Set[str] = set()
# Paths answering HTTP 200 with an HTML page and no auth check at all, mimicking Open
# WebUI's SPA answering an unknown path: a 2xx on its own must not be read as "the route
# exists and the credentials are valid".
#
# 不做任何鉴权、直接返回 200 + HTML 的路径，模拟 Open WebUI 的 SPA 回答未知路径的行为：
# 单凭 2xx 不得读作"路由存在且凭证有效"。
SPA_PATHS: Set[str] = set()
# Models used to verify that capabilities are established by probing the engine
# rather than echoed from the upstream's default metadata template:
#   * one whose engine has no tool-call parser  -> function_calling must be false;
#   * one that is not multimodal                -> vision must be false.
#
# 用于验证"能力靠探测引擎得出、而非照抄上游默认元数据模板"的模型：
#   * 引擎没带 tool-call parser -> function_calling 必须为 false；
#   * 非多模态                  -> vision 必须为 false。
NO_TOOLS_MODEL = "no-tools-model"
TEXT_ONLY_MODEL = "text-only-model"

# The capability dictionary Open WebUI hands to EVERY model (its "default model
# metadata" template is merged into each of them). It is deliberately the same for all
# models here, and deliberately claims things the specific engines do not support --
# exactly the trap that made the previous implementation advertise vision=true for a
# text-only engine.
#
# Open WebUI 发给**每个**模型的能力字典（它的"默认模型元数据"模板会合并进每个模型）。
# 这里刻意对所有模型都一样，并刻意声明了具体引擎并不支持的能力——这正是让旧实现
# 给纯文本引擎广告 vision=true 的陷阱。
DEFAULT_MODEL_CAPABILITIES = {
    "file_context": True,
    "vision": True,
    "file_upload": True,
    "web_search": True,
    "image_generation": True,
    "code_interpreter": True,
    "terminal": True,
    "citations": True,
    "status_updates": True,
    "builtin_tools": True,
    "usage": True,
}

# Reasoning-effort validation is two-layered upstream, and the proxy has to see both:
#   * `literal`  -- what the engine's request schema accepts (the sentinel probe
#                   reveals exactly this list);
#   * `accepted` -- what the model's own reasoning parser accepts afterwards, which is
#                   the smaller set that actually matters.
# A model whose two layers differ is the regression this file exists to catch.
#
# 上游的思考挡位校验有两层，代理必须都看到：
#   * `literal`  —— 引擎请求 schema 接受的挡位（哨兵探测揭示的正是这一份）；
#   * `accepted` —— 模型自带推理解析器随后真正接受的挡位，即真正重要的那个更小的集合。
# 两层不一致的模型，正是本文件存在意义所在的回归用例。
PROBEABLE_MODELS: Dict[str, Dict[str, Any]] = {
    # Both layers agree: a well-behaved engine.
    # 两层一致：行为良好的引擎。
    "llama3:latest": {
        "literal": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        "accepted": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        "style": "harmony",
        "default": None,
    },
    # Both layers disagree, and the second layer's error text is itself incomplete
    # (it never mentions "none", which the parser does accept) -- so advertising the
    # parsed list instead of the verified one would still be wrong.
    #
    # 两层不一致，且第二层的报错文本本身也不完整（它没提 none，而解析器其实接受
    # none）——因此"照抄解析出的列表"依然是错的，只有逐值实证才对。
    "legacy-model": {
        "literal": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        "accepted": ["none", "low", "medium", "xhigh"],
        "style": "qwen",
        "default": "xhigh",
    },
}

MODELS = {
    "object": "list",
    "data": [
        {
            "id": "llama3:latest",
            "name": "llama3",
            "object": "model",
            "created": 1700000000,
            "owned_by": "ollama",
            "info": {
                "meta": {
                    "profile_image_url": "/static/x.png",
                    "description": "should be dropped",
                    "capabilities": DEFAULT_MODEL_CAPABILITIES,
                }
            },
            "params": {"temperature": 0.7},
            "access_grants": [],
        },
        # Legacy upstreams may lack object / created / owned_by
        # 老版本上游可能缺少 object / created / owned_by
        {
            "id": "legacy-model",
            "name": "legacy-model",
            "info": {"meta": {"capabilities": DEFAULT_MODEL_CAPABILITIES}},
        },
        {
            "id": REASONING_MODEL,
            "name": REASONING_MODEL,
            "object": "model",
            "created": 1700000001,
            "owned_by": "ollama",
            "info": {"meta": {"capabilities": DEFAULT_MODEL_CAPABILITIES}},
        },
        {
            "id": NO_TOOLS_MODEL,
            "name": NO_TOOLS_MODEL,
            "object": "model",
            "created": 1700000002,
            "owned_by": "vllm",
            "max_model_len": 8192,
            "openai": {"owned_by": "vllm", "root": "mock/no-tools", "max_model_len": 8192},
            "info": {"meta": {"capabilities": DEFAULT_MODEL_CAPABILITIES}},
        },
        {
            "id": TEXT_ONLY_MODEL,
            "name": TEXT_ONLY_MODEL,
            "object": "model",
            "created": 1700000003,
            "owned_by": "vllm",
            "max_model_len": 4096,
            "openai": {"owned_by": "vllm", "root": "mock/text-only", "max_model_len": 4096},
            "info": {"meta": {"capabilities": DEFAULT_MODEL_CAPABILITIES}},
        },
    ],
}


def build_literal_error(efforts: list, requested: Any) -> str:
    """
    The pydantic/vLLM "outer schema" error the sentinel probe relies on.

    哨兵探测所依赖的 pydantic/vLLM "外层 schema" 报错。
    """
    quoted = ", ".join(f"'{effort}'" for effort in efforts[:-1]) + f" or '{efforts[-1]}'"
    return (
        "1 validation error:\n"
        "  {'type': 'literal_error', 'loc': ('body', 'reasoning_effort'), "
        f"'msg': \"Input should be {quoted}\", "
        f"'input': {json.dumps(requested)}, "
        f"'ctx': {{'expected': \"{quoted}\"}}}}"
    )


def build_second_layer_error(rules: Dict[str, Any], requested: str) -> str:
    """
    The model-level error of the second validation layer, in one of the two real
    phrasings captured from a live upstream.

    第二层（模型级）校验的报错，采用从真实上游捕获的两种措辞之一。
    """
    accepted = rules["accepted"]
    default = rules.get("default")
    if rules.get("style") == "qwen":
        # Qwen style, and faithfully incomplete: it names the thinking levels but not
        # "none", even though "none" is accepted.
        #
        # Qwen 风格，且忠实地"不完整"：它点名了思考挡位，却没提 none，
        # 而 none 其实是被接受的。
        named = [level for level in accepted if level != "none"]
        listed = []
        for level in reversed(named):
            listed.append(f"{level} (default)" if level == default else level)
        if len(listed) > 1:
            body = ", ".join(listed[:-1]) + f", and {listed[-1]}"
        else:
            body = listed[0]
        return f"Unexpected reasoning effort {requested}. Supported types are {body}."
    return (
        f"reasoning_effort='{requested}' is not supported by Harmony. "
        f"Supported values are: {', '.join(accepted)}."
    )


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

    def _send_html(self, body: str = "<html><body>mock SPA</body></html>") -> None:
        """
        Answer the way Open WebUI's SPA answers an unknown path.

        像 Open WebUI 的 SPA 那样回答未知路径。
        """
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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

    def _accepts_gzip(self) -> bool:
        """
        Whether the caller declared gzip support.

        调用方是否声明支持 gzip。
        """
        return "gzip" in (self.headers.get("Accept-Encoding") or "").lower()

    def _send_gzip(self, body: bytes, content_type: str) -> None:
        """
        Send a gzip-compressed response when the caller accepts gzip.

        Compression stays conditional (httpx only advertises gzip when it can decode
        it), so these tests also work with an httpx built without compression support.

        调用方接受 gzip 时发送压缩响应。

        压缩是有条件的（httpx 只有在能解码时才会声明 gzip），因此测试在缺少压缩支持的
        httpx 上同样能跑。
        """
        compressed = self._accepts_gzip()
        payload = gzip.compress(body) if compressed else body
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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
        # Both handled before the auth check on purpose: a 5xx and the SPA's 200 + HTML
        # page are exactly the answers that say nothing about the credentials.
        #
        # 两者刻意放在鉴权检查之前：5xx 与 SPA 的 200 + HTML 正是"对凭证不说明任何问题"的答案。
        if path in BROKEN_PATHS:
            self._send_json(500, {"detail": "mock upstream failure"})
            return
        if path in SPA_PATHS:
            self._send_html()
            return
        if path in ("/api/models", "/api/v1/models"):
            if not self._authorized(self.headers):
                self._send_json(401, {"detail": "Not authenticated"})
                return
            self._send_json(200, MODELS)
            return
        if path == "/api/config":
            # Instance metadata: the proxy reads it for the envelope's x_open_webui
            # (and deliberately NOT from /api/v1/config, which is not a route here and
            # would answer with an HTML page).
            #
            # 实例元信息：代理读取它用于信封里的 x_open_webui（且刻意不读
            # /api/v1/config —— 那在这里不是路由，会返回一页 HTML）。
            if not self._authorized(self.headers):
                self._send_json(401, {"detail": "Not authenticated"})
                return
            self._send_json(
                200,
                {
                    "status": True,
                    "name": "MockOpenWebUI",
                    "version": "mock-1.0",
                    "features": {
                        "enable_web_search": False,
                        "enable_code_execution": False,
                        "enable_code_interpreter": False,
                    },
                },
            )
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
        if path in ("/api/gzipped", "/api/v1/gzipped"):
            # A route answering with a gzip-compressed JSON body: the proxy must decode
            # it before forwarding (see GZIP_MODEL).
            #
            # 用 gzip 压缩的 JSON 回答的路由：代理必须先解码再转发（见 GZIP_MODEL）。
            self._send_gzip(
                json.dumps(
                    {"route": "gzipped", "model": model, "compressed": self._accepts_gzip()}
                ).encode("utf-8"),
                "application/json",
            )
            return
        if path == "/api/legacy-only":
            # A route that exists only under the legacy /api prefix, used to verify
            # prefix fallback of the catch-all passthrough
            #
            # 只存在于旧版 /api 前缀下的路由，用于验证兜底透传的前缀回退
            self._send_json(200, {"id": "legacy_only_mock", "route": "legacy-only", "model": model})
            return
        self._send_json(404, {"detail": f"Not Found: {path}"})

    @staticmethod
    def _carries_image(payload: Dict[str, Any]) -> bool:
        """
        Whether a chat request contains an image content part.

        聊天请求里是否包含图片内容块。
        """
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
        return False

    def _handle_gzip_chat(self, payload: Dict[str, Any], model: Optional[str]) -> None:
        """
        A gzip-compressed chat response, streaming or not (see GZIP_MODEL).

        gzip 压缩的聊天响应，流式与非流式皆是（见 GZIP_MODEL）。
        """
        if payload.get("stream"):
            chunks = [
                'data: {"id":"chat-1","object":"chat.completion.chunk","created":1700000000,'
                f'"model":{json.dumps(model)},"choices":[{{"index":0,"delta":{{"content":"gzip "}},"finish_reason":null}}]}}\n\n',
                'data: {"id":"chat-1","object":"chat.completion.chunk","created":1700000000,'
                f'"model":{json.dumps(model)},"choices":[{{"index":0,"delta":{{"content":"stream"}},"finish_reason":null}}]}}\n\n',
                "data: [DONE]\n\n",
            ]
            self._send_gzip("".join(chunks).encode("utf-8"), "text/event-stream")
            return
        self._send_gzip(
            json.dumps(
                {
                    "id": "chat-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "hello gzip"},
                        }
                    ],
                }
            ).encode("utf-8"),
            "application/json",
        )

    def _handle_chat(self, payload: Dict[str, Any], model: Optional[str]) -> None:
        if model == ERROR_MODEL:
            self._send_json(400, {"detail": "Model is not available"})
            return
        if model == ABORT_MODEL:
            self._send_sse_then_abort()
            return
        if model == GZIP_MODEL:
            self._handle_gzip_chat(payload, model)
            return

        # Capability probes: this engine was built without a tool-call parser, and
        # this one is not multimodal -- while the upstream's default metadata template
        # claims both. Only a probe can tell the truth.
        #
        # 能力探测：这个引擎没带 tool-call parser，那个不是多模态——而上游的默认元数据
        # 模板却把两者都声明为支持。只有探测能说出真相。
        if model == NO_TOOLS_MODEL and (
            payload.get("tools") or payload.get("tool_choice")
        ):
            self._send_json(
                400,
                {
                    "detail": (
                        '"auto" tool choice requires --enable-auto-tool-choice and '
                        "--tool-call-parser to be set"
                    )
                },
            )
            return
        if model == TEXT_ONLY_MODEL and self._carries_image(payload):
            self._send_json(400, {"detail": f"{model} is not a multimodal model"})
            return

        # Reasoning-effort validation, in its two real layers: the request schema
        # first (which is what the sentinel probe reveals), then the model's own
        # parser (which is the smaller set that actually matters).
        #
        # 思考挡位校验的两个真实层次：先是请求 schema（哨兵探测揭示的那一层），
        # 再是模型自带解析器（真正重要的那个更小的集合）。
        rules = PROBEABLE_MODELS.get(model or "")
        if rules is not None:
            requested = payload.get("reasoning_effort")
            if requested is not None and requested not in rules["literal"]:
                self._send_json(400, {"detail": build_literal_error(rules["literal"], requested)})
                return
            if requested is not None and requested not in rules["accepted"]:
                self._send_json(400, {"detail": build_second_layer_error(rules, requested)})
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


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """
    A threading server that does not dump a traceback when a client vanishes.

    The proxy opens and closes many short-lived connections while probing, and the
    socketserver default handler treats every client-side reset as a server error --
    which buries the real test output in noise.

    一个不会因客户端消失而倾倒 traceback 的多线程服务器。

    代理探测时会开合大量短连接，socketserver 的默认处理器把每次客户端重置都当成
    服务端错误——那会把真正的测试输出埋在噪声里。
    """

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(
            error, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError)
        ):
            return
        super().handle_error(request, client_address)


class MockOpenWebUI:
    # A mock upstream running in its own thread.
    # 在独立线程里跑起来的 mock 上游。

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._server = _QuietThreadingHTTPServer((host, port), _Handler)
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
