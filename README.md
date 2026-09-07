# open-webui-to-openai-api

[English](README.md) | [简体中文](README.zh-CN.md)

Reverse-proxy an **Open WebUI** instance — accessible only via browser login — into an **OpenAI-compatible** API, so that any OpenAI client can connect directly.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why You Need This

Some self-hosted Open WebUI instances — due to network restrictions, SSO single sign-on, outdated versions, or an administrator who has not enabled API keys — cannot give you a standard API key; all you have is the browser login session.

This project's approach: **Playwright opens a real browser once, you complete the login manually, and the script captures the post-login `Authorization` and `Cookie` in the background**, then starts a local OpenAI-compatible proxy service. Credentials are written to `session.json` and reused on subsequent startups, so you never have to log in twice.

## How It Works

```mermaid
flowchart LR
    A["OpenAI client\n(Chatbox / Cherry Studio / LangChain…)"] -->|"POST /v1/chat/completions\nAuthorization: Bearer PROXY_API_KEY"| B["This proxy\nFastAPI"]
    B -->|"Validate proxy key"| B
    B -->|"Swap in browser credentials\nPOST /api/v1/chat/completions"| C["Open WebUI"]
    C -->|"SSE / JSON"| B
    B -->|"Return as-is"| A
    D["Playwright browser login"] -.->|"Writes session.json"| B
```

Key points:

- **Credential swapping**: externally it presents your custom `PROXY_API_KEY`, while internally it swaps in the browser-captured `Authorization` / `Cookie` — the upstream never sees your proxy key.
- **Protocol alignment**: upstream Open WebUI ≥ 0.6 already provides OpenAI-compatible routes `/api/v1/*`; older versions only have the internal routes `/api/*`. This project **auto-detects** and remembers the working prefix at startup, and automatically falls back to the other prefix when a request returns 404 (route not found).
- **Response normalization**: `/v1/models` collapses upstream model objects into the standard `{id, object, created, owned_by}`, plus a whitelist of safe extras (`max_model_len`, `description`, `capabilities`). Private fields (`user_id`, `access_grants`, `permission`, `urlIdx`, ...) are never exposed.

## Features

- **OpenAI compatible**: `/v1/models`, `/v1/chat/completions` (including streaming SSE), `/v1/embeddings`, plus a catch-all passthrough for unimplemented `/v1/*` paths (with prefix fallback as well).
- **Automatic upstream version adaptation**: `auto` / `v1` / `legacy` upstream API styles, with startup probing + per-request fallback.
- **Robust streaming forwarding**: when the client disconnects, the upstream connection is closed proactively instead of hanging until timeout; hop-by-hop response headers are stripped correctly.
- **OpenAI-style error bodies**: returns `{"error": {"message", "type", "code"}}` instead of FastAPI's default `{"detail": ...}`, so clients can show the error reason properly.
- **Startup self-check**: a single `GET /models` performs both prefix probing and credential validation; dead credentials are reported immediately with a re-login hint, instead of being discovered on the first call.
- **Optional CORS**: configure `PROXY_CORS_ORIGINS` to let browser pages call this proxy directly (preflight is answered automatically); off by default to keep the exposure surface small.
- **Model aliases**: map client-requested model names to real upstream model names via `MODEL_ALIASES`.
- **Redacted credentials**: logs print only the token prefix and length; full credentials never end up in logs.

## Directory Structure

```
.
├── app.py                  # FastAPI routes, OpenAI compatibility layer, CLI entry
├── config.py               # All configuration items (env vars / .env)
├── session_store.py        # Credential load/save + Playwright browser login capture
├── upstream.py             # Upstream forwarding: connection pool, prefix probing, streaming
├── requirements.txt        # Minimal dependencies to run the service
├── requirements-browser.txt# Optional: Playwright for browser login
├── .env.example            # Configuration template
└── tests/
    ├── mock_openwebui.py   # Open WebUI simulator built on the standard library
    ├── test_smoke.py       # End-to-end smoke tests (mock upstream + real proxy startup)
    └── test_units.py       # Pure-logic unit tests
```

## Quick Start

### 1. Install dependencies

Python 3.9+:

```bash
pip install -r requirements.txt
```

If you want the "browser login to capture credentials" path, you also need:

```bash
pip install -r requirements-browser.txt
playwright install chromium
```

### 2. Configure

```bash
cp .env.example .env
```

At minimum, change these two:

```env
OPEN_WEBUI_BASE_URL=http://your-open-webui-domain.com
PROXY_API_KEY=sk-your-custom-proxy-key
```

### 3. First login

```bash
python app.py
```

On the first run (or when `session.json` does not exist):

1. A Chromium window opens;
2. You complete the Open WebUI login manually in that window;
3. The script watches requests sent to the upstream `/api/*` and writes `session.json` as soon as it captures a `Bearer` token or session cookie;
4. After quietly observing for `LOGIN_QUIET_PERIOD` seconds to confirm no newer tokens appear, it closes the browser automatically and starts the proxy service.

> "Login succeeded" is judged by **requests to `/api/` carrying identity information**, so the anonymous cookies produced by the first page load are not misdetected.

Common commands:

```bash
python app.py              # Start the service (logs in first if needed)
python app.py --login      # Force re-login and refresh credentials
python app.py --check      # Only validate credentials and upstream connectivity, print a summary, then exit
python app.py --port 9000  # Temporarily override the listen port
```

Output language (logs, banner, CLI help, error messages):

```bash
python app.py --lang zh    # Force Chinese output
python app.py --lang en    # Force English output
python app.py --lang auto  # Follow the system language (default), fall back to English
```

Selection priority: `--lang` flag > system language detection > English.

### 4. Connect a client

| Setting      | Value                        |
| ------------ | ---------------------------- |
| API Base URL | `http://127.0.0.1:8000/v1`   |
| API Key      | `PROXY_API_KEY` from `.env`  |
| Model name   | See the `curl` output below  |

> **Note**: the client should always use `http://127.0.0.1:8000/v1` (local) or `http://<machine-IP>:8000/v1` (LAN devices). `PROXY_HOST=0.0.0.0` is only a server-side listen setting — do **not** put it into the client as an API address. Electron/Chromium clients connecting to `0.0.0.0` fail with `net::ERR_ADDRESS_INVALID` (typical symptom: the model list loads fine but every chat errors).

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer sk-your-custom-proxy-key"
```

Python:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-custom-proxy-key",
    base_url="http://127.0.0.1:8000/v1",
)

resp = client.chat.completions.create(
    model="llama3:latest",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Node.js:

```js
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "sk-your-custom-proxy-key",
  baseURL: "http://127.0.0.1:8000/v1",
});

const stream = await client.chat.completions.create({
  model: "llama3:latest",
  messages: [{ role: "user", content: "Hello!" }],
  stream: true,
});
for await (const part of stream) {
  process.stdout.write(part.choices[0]?.delta?.content ?? "");
}
```

## API Endpoints

| Method | Path                   | Auth | Description                                                        |
| ---- | ---------------------- | -- | ---------------------------------------------------------------- |
| GET  | `/`                    | No* | Service info and registered endpoints; the upstream address is returned only with a valid key |
| GET  | `/healthz`             | No* | Health check always 200; upstream address and probed prefix returned only with a valid key |
| GET  | `/v1/models`           | Yes | Model list, normalized to the OpenAI structure                    |
| POST | `/v1/chat/completions` | Yes | Chat completions, supports `stream: true`                         |
| POST | `/v1/embeddings`       | Yes | Embeddings (upstream must support them)                           |
| ANY  | `/v1/{path}`           | Yes | Catch-all passthrough to the same upstream path                   |

Auth accepts both `Authorization: Bearer <key>` and `X-API-Key: <key>`. If `PROXY_API_KEY` is left empty, no auth is enforced.

\* `/` and `/healthz` remain unauthenticated (probe/readiness-check friendly), but the `upstream` / `upstream_prefix` fields in the response are only returned when the request carries a valid key (or auth is disabled), to avoid leaking the upstream intranet domain on public deployments.

## Configuration

| Env var               | Default                 | Description                                                                                          |
| --------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `OPEN_WEBUI_BASE_URL` | `http://localhost:8080` | Upstream address, must include `http://` or `https://`, no trailing slash                             |
| `UPSTREAM_API_STYLE`  | `auto`                  | `auto` / `v1` / `legacy`                                                                             |
| `UPSTREAM_VERIFY_SSL` | `true`                  | Set `false` when the upstream uses a self-signed certificate                                          |
| `UPSTREAM_TRUST_ENV`  | `true`                  | Whether to honor system proxy env vars; set `false` when the upstream is local/intranet and a system proxy is configured |
| `PROXY_HOST`          | `0.0.0.0`               | Server listen address (server-side only; `0.0.0.0` = all interfaces, NOT the API address to put in a client) |
| `PROXY_PORT`          | `8000`                  | Listen port                                                                                          |
| `PROXY_API_KEY`       | empty                   | Public access key; empty means no auth                                                               |
| `PROXY_CORS_ORIGINS`  | empty                   | Comma-separated list of allowed CORS origins; empty disables CORS                                    |
| `REQUEST_TIMEOUT`     | `300`                   | Total upstream request timeout (seconds)                                                             |
| `CONNECT_TIMEOUT`     | `10`                    | Upstream connect timeout (seconds)                                                                   |
| `SESSION_FILE`        | `session.json`          | Credential file path                                                                                 |
| `MODEL_ALIASES`       | empty                   | JSON object, model name mapping                                                                      |
| `LOG_LEVEL`           | `INFO`                  | `CRITICAL` / `ERROR` / `WARNING` / `INFO` / `DEBUG` / `TRACE`; invalid values fall back to `INFO`     |
| `DEBUG`               | `false`                 | When `true`, equivalent to `LOG_LEVEL=DEBUG`                                                          |
| `LOGIN_TIMEOUT`       | `600`                   | Max seconds to wait for the browser login                                                            |
| `LOGIN_QUIET_PERIOD`  | `6`                     | Seconds to keep observing after credentials are captured                                             |
| `LOGIN_HEADLESS`      | `false`                 | Whether to launch the browser headless                                                               |

## Credentials (session.json)

`session.json` simply holds the request headers of the browser login state:

```json
{
  "Authorization": "Bearer eyJhbGciOiJIUzI1NiIs...",
  "Cookie": "",
  "User-Agent": "Mozilla/5.0 ...",
  "captured_at": 1756000000.0,
  "base_url": "https://your-open-webui-domain.com"
}
```

At least one of `Authorization` and `Cookie` must be present. If you can grab the Open WebUI JWT from the browser DevTools (F12), you can also **write this file by hand** and skip the browser login entirely.

> `session.json` is already listed in `.gitignore` — never commit it. On POSIX systems it is written with `600` permissions automatically.

## Testing

The bundled tests spin up a built-in Open WebUI simulator and then start this proxy for real, covering all three upstream styles:

```bash
pip install -r requirements.txt

python tests/test_units.py    # Pure-logic unit tests, done in seconds, no network needed
python tests/test_smoke.py    # End-to-end: mock upstream + real proxy startup
```

`test_units.py` covers: credential serialization and legacy-format compatibility, login-signal detection, model-list normalization, config-parsing tolerance, error response shape.

`test_smoke.py` covers: health check, auth rejection, model-list normalization, non-streaming/streaming chat, upstream error passthrough, parameter validation, embeddings, catch-all passthrough, and 503 when credentials are missing — each run against all three upstream styles (`auto` / `v1` / `legacy`).

## Troubleshooting

**Q: The browser jumped to a campus/corporate network auth page, and "credentials saved" appears before I even enter my account**

When such a captive portal has not completed authentication, all requests are redirected by the gateway; early in page load, the Open WebUI frontend may also send probe requests with an expired old token from localStorage — neither captured "credential" is a valid login state. This project has a built-in check: captured credentials are only saved if they pass a real upstream authentication (a direct request to the upstream returns 2xx); otherwise it reports "failed upstream validation" and keeps waiting. Complete the campus network login on the auth page, then go back and complete the Open WebUI login.

**Q: Chatting from an Electron client (Cherry Studio / Chatbox etc.) fails with `net::ERR_ADDRESS_INVALID`**

The error happens in the client's own Chromium network stack; the request never reached this proxy. `ERR_ADDRESS_INVALID` means the connection target address is invalid — the most common cause is putting `PROXY_HOST=0.0.0.0` into the client as an API address; Chromium refuses to connect to `0.0.0.0`. Change the API Base URL to `http://127.0.0.1:8000/v1`. The split symptom of "model list works, chats fail" occurs because on Windows, the Node layer treats a connection to `0.0.0.0` as localhost and succeeds, while Chromium rejects it before even connecting. If the address is correct and the error persists, check whether the client itself has an HTTP proxy configured (loopback requests get sent away), and whether this proxy's logs show any incoming requests during chats (if not, the client is not connected).

**Q: The upstream returns 401 / 403 and the logs say "credentials have expired"**

Open WebUI's JWT has a limited lifetime (controlled by the server-side `JWT_EXPIRES_IN`, usually a few days). Delete `session.json` and log in again:

```bash
python app.py --login
```

**Q: Every request returns 404, logs say "all candidate prefixes returned 404"**

`OPEN_WEBUI_BASE_URL` may not point to an Open WebUI instance (e.g. it was set to an Ollama address), or the upstream version is very old. Run `python app.py --check` first to see the probe result, and if necessary set `UPSTREAM_API_STYLE=legacy` explicitly.

**Q: The upstream is on the intranet/local machine, but requests time out or return 502**

The system may have `HTTP_PROXY`/`HTTPS_PROXY` configured; httpx honors them by default, sending loopback requests through the proxy. Set `UPSTREAM_TRUST_ENV=false`.

**Q: Can reasoning models (DeepSeek-R1 / QwQ etc.) return their thinking process?**

Yes. This proxy passes through **both request and response bodies in full**, without parsing or trimming conversation content: request parameters like `reasoning_effort` / `temperature` are forwarded to the upstream as-is, and `reasoning_content` (DeepSeek style), `reasoning` (newer Open WebUI style), and `<think>` tags in the body all reach the client as-is in both streaming and non-streaming modes (covered by regression tests). Whether the thinking content is visible ultimately depends on two things: whether the upstream Open WebUI returns it (requires a reasoning model and a supporting version), and whether the client renders it (both Cherry Studio and Chatbox do).

**Q: Streaming output gets buffered and the client receives everything at once**

Reverse proxies (Nginx etc.) need buffering disabled. This project already sends `X-Accel-Buffering: no` in the response headers; on the Nginx side make sure `proxy_buffering off;` is set.

**Q: Can this run on a headless server without a GUI?**

The browser login step inherently requires human interaction; on a headless server you need a virtual display such as Xvfb. The recommended approach is to log in once locally and copy the generated `session.json` to the server.

## Security Notes

- Always set `PROXY_API_KEY`; otherwise anyone who can reach the port can borrow your Open WebUI identity.
- Never commit `session.json` to version control or bake it into container images.
- Listen on `127.0.0.1` whenever possible; when exposing externally, put a reverse proxy in front and enable HTTPS.
- This proxy forwards request bodies as-is — do not expose it to untrusted callers.
- When enabling CORS (`PROXY_CORS_ORIGINS`), list the minimal set of origins you actually need; avoid `*`.

## Contributing

Issues and PRs are welcome. When a change affects forwarding behavior, please update the cases under `tests/` accordingly and make sure both commands are green:

```bash
python tests/test_units.py
python tests/test_smoke.py
```

## License

This project is licensed under the [MIT License](LICENSE).
