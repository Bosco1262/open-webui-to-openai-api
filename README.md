# open-webui-to-openai-api

[English](README.md) | [简体中文](README.zh-CN.md)

Reverse-proxy an **Open WebUI** instance — accessible only via browser login — into an **OpenAI-compatible** API, so that any OpenAI client can connect directly.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
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
- **Protocol alignment**: upstream Open WebUI ≥ 0.6 already provides OpenAI-compatible routes `/api/v1/*`; older versions only have the internal routes `/api/*`. This project **auto-detects** and remembers the working prefix at startup — a candidate only counts when it answers with a real model list, so a 5xx or an HTML page is never mistaken for the right prefix — and automatically falls back to the other prefix when a request returns 404 (route not found). If the fallback keeps being the one doing the work (3 calls in a row), the cached prefix is flipped to it, so a deployment that moved its routes stops paying for the extra hop.
- **Response normalization**: `/v1/models` collapses upstream model objects into the standard `{id, object, created, owned_by}`, plus a whitelist of useful extras: `name`, `description`, `max_context_length` / `context_length` (with `max_model_len` kept as a compatibility alias) and `quantization` (parsed from the model id, e.g. `NVFP4`). Private fields (`user_id`, `access_grants`, `permission`, `urlIdx`, ...) are never exposed.
- **Probed, not echoed**: `capabilities`, `architecture`, `supported_parameters` and `reasoning` are established by asking the engine (`/v1/models`), never copied from Open WebUI's metadata — see [Per-model probe](#per-model-probe-reasoning-efforts--capabilities). The deployment's own feature switches live in the envelope's `x_open_webui` instead, where they cannot be mistaken for model abilities.

## Features

- **OpenAI compatible**: `/v1/models`, `/v1/models/{id}`, `/v1/chat/completions` (including streaming SSE), `/v1/embeddings`, plus a catch-all passthrough for unimplemented `/v1/*` paths (with prefix fallback as well) — restricted to an allowlist by default.
- **Automatic upstream version adaptation**: `auto` / `v1` / `legacy` upstream API styles, with startup probing + per-request fallback.
- **Robust streaming forwarding**: when the client disconnects, the upstream connection is closed proactively instead of hanging until timeout; hop-by-hop response headers are stripped correctly, and a compressed upstream body (`gzip` / `deflate` / `br` / `zstd`, as supported by the HTTP stack) is decoded before it is handed to the client.
- **OpenAI-style error bodies**: returns `{"error": {"message", "type", "code"}}` instead of FastAPI's default `{"detail": ...}`, so clients can show the error reason properly.
- **Startup self-check**: a single `GET /models` performs both prefix probing and credential validation; dead credentials are reported immediately with a re-login hint, instead of being discovered on the first call.
- **Optional CORS**: configure `PROXY_CORS_ORIGINS` to let browser pages call this proxy directly (preflight is answered automatically); off by default to keep the exposure surface small.
- **Model aliases**: map client-requested model names to real upstream model names via `MODEL_ALIASES`.
- **Redacted credentials**: logs print only the token prefix and length; full credentials never end up in logs.
- **Hardened by default**: the passthrough is an allowlist rather than "forward everything"; upstream redirects are refused instead of followed with your credentials; the upstream's `Set-Cookie` / `WWW-Authenticate` never reach the client; request bodies are size-capped while being read (chunked included); upstream error details stay in the log; and every request carries a correlation id you can grep.

## Security defaults (at a glance)

| Behaviour | Default | How to change it |
| --- | --- | --- |
| `/v1/*` passthrough | allowlist: `images`, `audio`, `files`, `responses` | `PASSTHROUGH_ALLOW=...`, or `PASSTHROUGH_ALLOW=*` for unrestricted |
| Upstream error text in client errors | hidden (log only, plus a request id) | `EXPOSE_UPSTREAM_ERROR=true` |
| Upstream 3xx answers | refused (never followed) | point `OPEN_WEBUI_BASE_URL` at the final address |
| Upstream `Set-Cookie` / `WWW-Authenticate` | stripped | — (clients authenticate with the proxy key) |
| Plain `http` upstream on a LAN/non-loopback host | accepted, with one startup warning | put a TLS reverse proxy in front, or accept the trusted link |
| Request body size | capped while reading (`MAX_BODY_BYTES`, 10 MiB) | `MAX_BODY_BYTES` |
| Response headers | `X-Request-ID`, `Cache-Control: no-store, private`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` | — |

## Directory Structure

```
.
├── app.py                  # FastAPI routes, OpenAI compatibility layer, CLI entry
├── config.py               # All configuration items (env vars / .env)
├── lang.py                 # User-facing message localization (zh / en)
├── models.py               # Model-list normalization + engine fingerprint (pure logic)
├── model_probe.py          # Per-model probe: error parsing, probe payloads, cache
├── probe_runner.py         # Upstream data access + probe orchestration (runtime singletons)
├── session_store.py        # Credential load/save + Playwright browser login capture
├── upstream.py             # Upstream forwarding: connection pool, prefix probing, streaming
├── atomic_json.py          # Shared atomic (flush + fsync + rename) JSON writer
├── request_context.py      # Per-request correlation id (contextvar + log formatter)
├── requirements.txt        # Minimal dependencies to run the service
├── requirements-browser.txt# Optional: Playwright for browser login
├── requirements-dev.txt    # Optional: pytest, for running the tests
├── .env.example            # Configuration template
└── tests/
    ├── mock_openwebui.py   # Open WebUI simulator built on the standard library
    ├── test_smoke.py       # End-to-end smoke tests (mock upstream + real proxy startup)
    └── test_units.py       # Pure-logic unit tests
```

## Quick Start

### 1. Install dependencies

Python 3.11+ (the probe-refresh path uses `asyncio.TaskGroup`):

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
python app.py --probe      # Force a full re-probe of every model (efforts + capabilities) and refresh the cache
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
| GET  | `/healthz`             | No* | Health check always 200; upstream address, probed prefix and probe health returned only with a valid key |
| GET  | `/v1/models`           | Yes | Model list, normalized to the OpenAI structure, with probed capabilities / parameters / reasoning attached. The envelope also carries `x_open_webui` with the deployment's own metadata |
| GET  | `/v1/models/{id}`      | Yes | Retrieve one model (`id` may contain slashes); 404 with an OpenAI-style error body when unknown |
| POST | `/v1/chat/completions` | Yes | Chat completions, supports `stream: true`                         |
| POST | `/v1/embeddings`       | Yes | Embeddings (upstream must support them)                           |
| ANY  | `/v1/{path}`           | Yes | Catch-all passthrough to the same upstream path, limited to `PASSTHROUGH_ALLOW` (default `images`, `audio`, `files`, `responses`); anything else is answered 403 without touching the upstream |

Auth accepts both `Authorization: Bearer <key>` and `X-API-Key: <key>`, and accepts any of the keys from `PROXY_API_KEY` / `PROXY_API_KEYS`. If both are empty, no auth is enforced.

Every response carries an `X-Request-ID` header: the client's own value when it sent a usable one, a generated id otherwise. The same id is in the service log line, so "this request failed" can be answered by grepping one string. `/healthz` with a valid key also reports the probe health (`status`, consecutive credential rejections, the last round's counters).

Query parameters on `/v1/models` are ignored, exactly as OpenAI's own endpoint does (it has no pagination either; only Anthropic's and Gemini's differently-shaped APIs implement it).

\* `/` and `/healthz` remain unauthenticated (probe/readiness-check friendly), but the `upstream` / `upstream_prefix` fields in the response are only returned when the request carries a valid key (or auth is disabled), to avoid leaking the upstream intranet domain on public deployments.

## Configuration

| Env var               | Default                 | Description                                                                                          |
| --------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `OPEN_WEBUI_BASE_URL` | `http://localhost:8080` | Upstream address, must include `http://` or `https://`, no trailing slash. Localhost and LAN addresses are first-class; a non-loopback plain-`http` address is accepted and only produces one startup warning (credentials cross that link in cleartext) |
| `UPSTREAM_API_STYLE`  | `auto`                  | `auto` / `v1` / `legacy`                                                                             |
| `UPSTREAM_VERIFY_SSL` | `true`                  | Set `false` when the upstream uses a self-signed certificate                                          |
| `UPSTREAM_TRUST_ENV`  | `true`                  | Whether to honor system proxy env vars; set `false` when the upstream is local/intranet and a system proxy is configured |
| `PROXY_HOST`          | `0.0.0.0`               | Server listen address (server-side only; `0.0.0.0` = all interfaces, NOT the API address to put in a client) |
| `PROXY_PORT`          | `8000`                  | Listen port                                                                                          |
| `PROXY_API_KEY`       | empty                   | Public access key; empty means no auth                                                               |
| `PROXY_API_KEYS`      | empty                   | Optional named keys instead of one shared key: `name:key` entries, comma-separated (bare entries are auto-named). Lets one client be rotated or revoked on its own; `PROXY_API_KEY` keeps working alongside |
| `PROXY_CORS_ORIGINS`  | empty                   | Comma-separated list of allowed CORS origins; empty disables CORS                                    |
| `REQUEST_TIMEOUT`     | `300`                   | Total upstream request timeout (seconds)                                                             |
| `CONNECT_TIMEOUT`     | `10`                    | Upstream connect timeout (seconds)                                                                   |
| `SESSION_FILE`        | `session.json`          | Credential file path                                                                                 |
| `MODEL_PROBE_CACHE_FILE` | `model_probe_cache.json` | Per-model probe cache path                                                                  |
| `MODEL_PROBE_CONCURRENCY` | `4`             | Concurrency of the probe requests                                                                    |
| `MODEL_PROBE_TIMEOUT`     | `30`            | Per-model probe timeout (seconds)                                                                    |
| `MODEL_PROBE_WAIT`        | `5`             | Max seconds `/v1/models` waits for a probe that is *in flight*; `0` = never wait                     |
| `EXPOSE_INSTANCE_META`    | `true`          | Whether the `/v1/models` envelope carries `x_open_webui` (turn off for strict clients)               |
| `EXPOSE_UPSTREAM_ERROR`   | `false`         | Whether upstream error bodies are quoted in the client-facing error message; `false` (default) keeps them in the log only and answers with a fixed message plus the request id |
| `MAX_BODY_BYTES`          | `10485760`      | Reject JSON request bodies larger than this (413). Enforced while reading, so a chunked body without `Content-Length` is capped too |
| `MODEL_LIST_TTL`          | `10`            | Seconds the upstream model list is reused before refetching; `0` = fetch on every request            |
| `ALLOW_INSECURE`          | `false`         | Safety interlock override: permits starting with no proxy key on a non-loopback address              |
| `PASSTHROUGH_ALLOW`       | `images,audio,files,responses` | Subpaths the `/v1/*` passthrough may forward (exact or subpath match). `*` = unrestricted (the historical behavior); an explicitly empty value denies everything |
| `MODEL_ALIASES`       | empty                   | JSON object, model name mapping                                                                      |
| `LOG_LEVEL`           | `INFO`                  | `CRITICAL` / `ERROR` / `WARNING` / `INFO` / `DEBUG` / `TRACE`; invalid values fall back to `INFO`     |
| `DEBUG`               | `false`                 | When `true`, equivalent to `LOG_LEVEL=DEBUG`                                                          |
| `LOGIN_TIMEOUT`       | `600`                   | Max seconds to wait for the browser login                                                            |
| `LOGIN_QUIET_PERIOD`  | `6`                     | Seconds to keep observing after credentials are captured                                             |
| `LOGIN_HEADLESS`      | `false`                 | Whether to launch the browser headless                                                               |

## Per-model probe (reasoning efforts & capabilities)

`/v1/models` attaches four probed fields to each model:

```json
{
  "id": "Qwen3.8-27B",
  "object": "model",
  "created": 1787109489,
  "owned_by": "vllm",
  "name": "Qwen3.8-27B",
  "max_model_len": 262144,
  "max_context_length": 262144,
  "context_length": 262144,
  "architecture": {
    "modality": "text->text",
    "input_modalities": ["text"],
    "output_modalities": ["text"]
  },
  "supported_parameters": ["reasoning_effort", "response_format", "temperature", "tools", "..."],
  "capabilities": {
    "vision": false,
    "function_calling": true,
    "reasoning": true,
    "structured_outputs": true
  },
  "reasoning": {
    "supported_efforts": ["none", "low", "medium", "xhigh"],
    "mandatory": false,
    "default_effort": "xhigh",
    "default_enabled": true
  }
}
```

### Why probing is the only honest source

Open WebUI reports `info.meta.capabilities` for every model, but that block is its
**default model metadata template merged into each model** — the keys are identical
across models and unrelated to what a given engine can do. On a real deployment every
model claimed `vision: true` while one of them answered an image with HTTP 400
`"... is not a multimodal model"`, and claimed `web_search: true` while the same
instance reported `enable_web_search: false` in `/api/config`. The proxy therefore
establishes capabilities by **asking the engine**, and publishes the upstream template
once, as an instance fact, under the envelope's `x_open_webui`:

```json
{
  "object": "list",
  "data": [ ... ],
  "x_open_webui": {
    "name": "GENAI Chat (Open WebUI)",
    "version": "0.9.2",
    "features": { "enable_web_search": false, "..." : "..." },
    "default_model_capabilities": { "vision": true, "web_search": true, "..." : "..." }
  }
}
```

`default_model_capabilities` holds the keys **every** reporting model agrees on; a key
the upstream does not report uniformly (a deployment had one model without `usage`)
is left out of the template, and the models that do report it carry their own value in
that model's `x_open_webui_deviations.capabilities`. The two keys are deliberately
different names: `x_open_webui` is the deployment's metadata on the envelope,
`x_open_webui_deviations` is one model's disagreement with the template inside `data[]`.
`/v1/models` stays this shape whether or not the template exists, and
`EXPOSE_INSTANCE_META=false` removes the envelope's `x_open_webui` entirely.

### How one probe works

Every step is a real request with `max_tokens=1`, so a full probe costs a handful of
output tokens (measured: ~5–100 prompt tokens per request, 5–13 requests per model):

| Step | Requests | Establishes |
| --- | --- | --- |
| 1. Candidate discovery | 1 (0 tokens) | A sentinel `reasoning_effort` (`"__probe__"`) makes the request schema fail with a 400 that enumerates the levels it accepts |
| 2. **Per-value verification** | ≤7 | Only a 200 for a concrete level counts. This is what catches the **second** validation layer (gpt-oss's Harmony, Qwen's own parser), which rejects a subset of the first |
| 3. Request parameters | 1 (+≤3 retries) | One merged request; a 400 is attributed to a parameter and retried without it. Also yields `function_calling` and `structured_outputs` |
| 4. Vision | 1 | A 1×1 PNG: 400 `"... is not a multimodal model"` means `vision: false` |
| 5. Default behaviour | 1 (0 for an unprobeable model) | The same request without `reasoning_effort`; whether thinking text comes back gives `default_enabled`. When step 1 found the upstream does not validate the field at all, step 3's own 200 already answers this, so the request is skipped |

Every probe request carries `X-WebUI-Proxy-Probe: 1`, so probe traffic can be told
apart from real chat traffic in the upstream's logs (it shares the same connection
pool and credentials).

Step 2 is why `supported_efforts` can be trusted. The outer schema is a superset: a
live Qwen3.8-27B advertises `none/minimal/low/medium/high/xhigh/max`, really accepts
`none/low/medium/xhigh`, and its own error text mentions only three of those four —
so neither the advertised list nor the error text is the answer, only a request per
value is.

### What each field claims — and what it does not

- `supported_efforts`: exactly the levels that answered 200, in canonical `none → max` order;
- `mandatory`: `true` when `none` was rejected, i.e. thinking cannot be turned off;
- `default_effort` / `default_enabled`: **omitted when the engine does not say** (they
  are never guessed; OpenRouter's own `reasoning` object, whose shape this follows,
  omits them the same way);
- `capabilities.function_calling`: the engine **accepts** `tools` / `tool_choice`
  (a 400 is a definite "no", a 200 does not promise the model will actually call a
  tool); `vision` means the engine accepts image content; `structured_outputs` means
  it accepts `response_format: json_schema`;
- `supported_parameters`: parameters the engine did **not reject**, named in
  OpenRouter's namespace. Unknown facts are omitted rather than defaulted.

### When a model is (re-)probed

1. **Startup** — models with no conclusive cache entry, in the background, never blocking the service;
2. **On `/v1/models`** — a model whose entry is missing, whose engine fingerprint changed, or whose backoff expired; `/v1/models` waits (bounded by `MODEL_PROBE_WAIT`, default 5s) **only while a probe for one of those models is actually in flight**, and never for a model that is in backoff or that the upstream does not validate;
3. **On a live 400** that blames the reasoning effort — the disproved level is dropped immediately and the model is re-probed in the background; the client still receives the upstream error unchanged and is never delayed;
4. **`python app.py --probe`** — force a full re-probe of every model and exit.

The engine fingerprint is derived from the model list (`openai.root`, `max_model_len`,
`owned_by`, `info.updated_at`), so checking it costs no request. It deliberately
excludes the top-level `created`: vLLM rebuilds its model card on every response and
stamps it with the current time, so it changes on every fetch.

### Cache, failures and backoff

Results live in `model_probe_cache.json` (version 2, one entry per model, next to
`session.json`) and hold the effort set, capabilities, parameters, the fingerprint and
the probe status. A failed or partial probe is cached too, with exponential backoff
(60s → 6h), so a broken model cannot make `/v1/models` re-probe — and stall — on every
request. A failed re-probe never discards facts established earlier.

## Credentials (session.json)

`session.json` simply holds the request headers of the browser login state:

```json
{
  "authorization": "Bearer eyJhbGciOiJIUzI1NiIs...",
  "cookie": "",
  "user_agent": "Mozilla/5.0 ...",
  "captured_at": 1756000000.0,
  "base_url": "https://your-open-webui-domain.com"
}
```

At least one of `authorization` and `cookie` must be present. Keys are matched case-insensitively (and `user_agent` / `User-Agent` are both accepted), so the capitalized form written by older versions keeps working. If you can grab the Open WebUI JWT from the browser DevTools (F12), you can also **write this file by hand** and skip the browser login entirely.

`base_url` is the upstream the credentials were captured for, and it is enforced: if
it does not match the configured `OPEN_WEBUI_BASE_URL`, the file is refused with an
error naming both addresses. Browser credentials are bound to the site that issued
them, so reusing them against another instance could only ever produce a confusing
upstream 401. Trailing slashes and letter case are ignored, and a file without
`base_url` (written by an older version) is accepted as before. Re-run
`python app.py --login` after moving to a new upstream.

> `session.json` is already listed in `.gitignore` — never commit it, and never bake it into a container image. It is not a sample file: while it exists it is a **working upstream session**, so keep it out of backups, sync folders and screenshots too. Treat it as a password, and delete it once the session it holds has been retired (after a re-login the old one is useless, and losing it costs nothing but a re-login). POSIX systems get `600` permissions automatically; on Windows it keeps the inherited ACLs, so place it in a user-only directory if the machine is shared. The service prints this reminder whenever it writes the file.

## Testing

The bundled tests spin up a built-in Open WebUI simulator and then start this proxy for real, covering all three upstream styles:

```bash
pip install -r requirements.txt

python tests/test_units.py    # Pure-logic unit tests, done in seconds, no network needed
python tests/test_smoke.py    # End-to-end: mock upstream + real proxy startup
```

For pytest (optional, `pip install -r requirements-dev.txt`):

```bash
pytest tests/test_units.py
```

`test_units.py` covers: credential serialization and legacy-format compatibility, login-signal detection, model-list normalization, the language-key table against the keys the source actually uses, config-parsing tolerance, error response shape, the upstream fallback/heal paths, the accepted upstream address forms (localhost / LAN / private / plain-http / custom-port all pass, only a malformed URL is refused) together with the cleartext-notice predicate, the passthrough allowlist parsing, named proxy keys, request-id sanitizing, probe health state transitions, the request-body cap (declared and chunked), and redirect/credential-header handling. Run directly, it keeps the collected summary output.

`test_smoke.py` covers: health check, auth rejection, model-list normalization, non-streaming/streaming chat, upstream error passthrough (redacted by default, with the request id echoed), parameter validation, embeddings, catch-all passthrough, gzip-compressed upstream responses, prefix fallback when the primary candidate answers 5xx, cookie-only credentials, a session captured for another upstream, 503 when credentials are missing, request-id echo and sanitizing, the security response headers, strict `stream` handling, stripped `Set-Cookie`, a 3xx that is refused rather than followed (verified by the redirect target never being hit), the passthrough allowlist (default / explicit / `*` / empty), the body cap, and named keys — each run against all three upstream styles (`auto` / `v1` / `legacy`). How long a probe scenario waits for the background probe to land defaults to 60s and can be raised on a slow machine with `SMOKE_PROBE_TIMEOUT=180`.

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

**Q: Every request returns 404, logs say "no candidate prefix could be confirmed"**

`OPEN_WEBUI_BASE_URL` may not point to an Open WebUI instance (e.g. it was set to an Ollama address), or the upstream version is very old. Run `python app.py --check` first to see the probe result, and if necessary set `UPSTREAM_API_STYLE=legacy` explicitly.

**Q: The upstream is on the intranet/local machine, but requests time out or return 502**

The system may have `HTTP_PROXY`/`HTTPS_PROXY` configured; httpx honors them by default, sending loopback requests through the proxy. Set `UPSTREAM_TRUST_ENV=false`.

**Q: Can reasoning models (DeepSeek-R1 / QwQ etc.) return their thinking process?**

Yes. This proxy passes through **both request and response bodies in full**, without parsing or trimming conversation content: request parameters like `reasoning_effort` / `temperature` are forwarded to the upstream as-is, and `reasoning_content` (DeepSeek style), `reasoning` (newer Open WebUI style), and `<think>` tags in the body all reach the client as-is in both streaming and non-streaming modes (covered by regression tests). Whether the thinking content is visible ultimately depends on two things: whether the upstream Open WebUI returns it (requires a reasoning model and a supporting version), and whether the client renders it (both Cherry Studio and Chatbox do).

**Q: `/v1/models` advertises a `reasoning_effort` level, but calling it returns 400**

It should not happen: every level is verified with a real request before being advertised, and a live 400 that blames the reasoning effort drops that level and re-probes the model in the background. If you still see it, the backend changed after the last probe — check the log for the `probe` lines, and run `python app.py --probe` to re-verify everything immediately.

**Q: Where did the per-model `web_search` / `terminal` / `citations` capability flags go?**

They were never model capabilities: Open WebUI's `info.meta.capabilities` is a deployment-wide default template it merges into every model, so it said the same thing about all of them. They now live once per response, in the `/v1/models` envelope's `x_open_webui.default_model_capabilities`, next to the instance's real feature switches. The per-model `capabilities` object holds only what the engine confirmed by probe.

**Q: Streaming output gets buffered and the client receives everything at once**

Reverse proxies (Nginx etc.) need buffering disabled. This project already sends `X-Accel-Buffering: no` in the response headers; on the Nginx side make sure `proxy_buffering off;` is set.

**Q: Can this run on a headless server without a GUI?**

The browser login step inherently requires human interaction; on a headless server you need a virtual display such as Xvfb. The recommended approach is to log in once locally and copy the generated `session.json` to the server.

**Q: `python app.py --login` stops with a Playwright/browser error**

The two usual causes are a browser that was never downloaded and a machine without a display. Both are reported as one message with the fix instead of a stack trace: run `pip install -r requirements-browser.txt` followed by `playwright install chromium`, and on a machine without a display set `LOGIN_HEADLESS=true` (or run under Xvfb). If neither helps, log in on a desktop machine and copy `session.json` over.

**Q: A `/v1/...` path returns 403 `passthrough_forbidden`**

That route is not in the passthrough allowlist, which defaults to `images,audio,files,responses`. Add the subpath you need (`PASSTHROUGH_ALLOW=responses,images,...`), or set `PASSTHROUGH_ALLOW=*` to forward everything. Remember that every entry lets whoever holds a proxy key use your account on that upstream route.

**Q: Requests fail with "upstream answered a redirect; refusing to follow it"**

The upstream answered a 3xx for a request that carries your credentials. Following it would send those credentials to whatever host the `Location` header names, so the proxy refuses — this is what a deployment that redirects http→https, or a captive portal intercepting traffic, looks like from here. Set `OPEN_WEBUI_BASE_URL` to the final address the redirect points at (and finish any network-portal login first). The log line includes the `Location` value.

**Q: Startup logs "plain http on a non-loopback host"**

That is a notice, not a failure — a LAN Open WebUI (`http://192.168.x.x:3000`) is a perfectly normal setup for this project and keeps working. It exists because `Authorization` and `Cookie` cross that link in cleartext: put a TLS reverse proxy in front when the network is not fully trusted, or ignore it for a segment you control. Loopback addresses never trigger it.

**Q: The error message no longer contains the upstream's own text**

That is the default since 1.1.0 (`EXPOSE_UPSTREAM_ERROR=false`): upstream bodies routinely name internal hosts, paths and network details. Quote the `request id` from the response instead and grep it in the service log, where the full detail is recorded. Set `EXPOSE_UPSTREAM_ERROR=true` if you prefer the old behavior.

## Security Notes

- Always set `PROXY_API_KEY` (or `PROXY_API_KEYS`); otherwise anyone who can reach the port can borrow your Open WebUI identity.
- **The proxy key is equivalent to your upstream session — within the passthrough allowlist**: the `/v1/{path}` catch-all forwards requests with the captured browser credentials attached, so a leaked key is as bad as leaking your Open WebUI login itself. The default allowlist (`images`, `audio`, `files`, `responses`) keeps that equivalence to those routes; every path you add widens it, and `PASSTHROUGH_ALLOW=*` restores the unrestricted behavior. The startup log always states which mode is in effect.
- **Credentials never travel to a redirect target**: the shared upstream client does not follow redirects. A 3xx is reported as an upstream failure and logged with its `Location`, instead of re-sending your `Authorization` / `Cookie` to whatever host it names.
- **The upstream's session cookies never reach clients**: `Set-Cookie`, `Set-Cookie2` and `WWW-Authenticate` are stripped from upstream responses. Otherwise a login-ish endpoint reachable through the passthrough could hand a client a working upstream session that bypasses this proxy entirely.
- **Upstream errors no longer name your internals**: with the default `EXPOSE_UPSTREAM_ERROR=false`, the upstream body goes to the log and the client gets a fixed message plus the request id (`X-Request-ID`), which is also on every other response and log line. Set it to `true` if you would rather debug from the client side and accept the extra disclosure.
- **A cleartext upstream link is reported, not refused**: when `OPEN_WEBUI_BASE_URL` is plain `http` on a non-loopback host, the startup log says so once. Localhost and LAN deployments (`http://192.168.x.x:3000`) are normal, supported setups — the notice only says the credentials travel in cleartext, and suggests a TLS reverse proxy when that matters.
- As an interlock, the service **refuses to start** when no proxy key is configured and `PROXY_HOST` is a non-loopback address; set `ALLOW_INSECURE=true` to override explicitly.
- Never commit `session.json` to version control or bake it into container images; while it exists, it is a live credential.
- Listen on `127.0.0.1` whenever possible; when exposing externally, put a reverse proxy in front and enable HTTPS.
- This proxy forwards request bodies as-is (up to `MAX_BODY_BYTES`) — do not expose it to untrusted callers.
- When enabling CORS (`PROXY_CORS_ORIGINS`), list the minimal set of origins you actually need; avoid `*`.
- **Single process only**: the proxy keeps per-process state (upstream prefix, probe cache, credential cache). Do not run it with `uvicorn --workers N`.

### Behavior changes in 1.1.0

If you are upgrading from 1.0.x, these defaults changed; each is one env var away from the old behavior:

| Change | Was | Now |
| --- | --- | --- |
| `/v1/*` passthrough | everything forwarded | allowlist (`images`, `audio`, `files`, `responses`); `PASSTHROUGH_ALLOW=*` for the old behavior |
| Upstream error text in client errors | quoted by default | hidden by default (`EXPOSE_UPSTREAM_ERROR=true` to restore) |
| Upstream 3xx | followed silently | refused (fix `OPEN_WEBUI_BASE_URL` if your deployment redirects) |
| `stream: "false"` (string) | treated as streaming | treated as non-streaming, like every other non-`true` value |
| Blocked passthrough paths | 404 | 403 `passthrough_forbidden` |

Nothing else changed shape: localhost and LAN upstreams, `UPSTREAM_VERIFY_SSL=false`, proxy env vars and every timeout/limit keep working exactly as before.

## Contributing

Issues and PRs are welcome. When a change affects forwarding behavior, please update the cases under `tests/` accordingly and make sure both commands are green:

```bash
python tests/test_units.py
python tests/test_smoke.py
```

## License

This project is licensed under the [MIT License](LICENSE).
