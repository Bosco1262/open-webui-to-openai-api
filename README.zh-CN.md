# open-webui-to-openai-api

[English](README.md) | [简体中文](README.zh-CN.md)

把只能通过网页登录访问的 **Open WebUI** 实例，反向代理成 **兼容 OpenAI 协议** 的 API，让任何 OpenAI 客户端都能直接连上。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 为什么需要它

有些自部署的 Open WebUI 实例出于网络限制、SSO 单点登录、版本过旧或管理员未开通 API Key 等原因，拿不到标准的 API Key，只有浏览器里的登录态。

本项目的做法是：**用 Playwright 打开一次真实浏览器，你手动完成登录，脚本在后台抓取登录后的 `Authorization` 与 `Cookie`**，之后本地起一个 OpenAI 兼容的代理服务。凭证会落到 `session.json`，后续启动直接复用，不必重复登录。

## 工作原理

```mermaid
flowchart LR
    A["OpenAI 客户端\n(Chatbox / Cherry Studio / LangChain…)"] -->|"POST /v1/chat/completions\nAuthorization: Bearer PROXY_API_KEY"| B["本代理\nFastAPI"]
    B -->|"校验 Proxy Key"| B
    B -->|"换成浏览器凭证\nPOST /api/v1/chat/completions"| C["Open WebUI"]
    C -->|"SSE / JSON"| B
    B -->|"原样返回"| A
    D["Playwright 浏览器登录"] -.->|"写出 session.json"| B
```

关键点：

- **凭证替换**：对外用你自定义的 `PROXY_API_KEY`，对内替换成浏览器抓来的 `Authorization` / `Cookie`，上游永远看不到你的 Proxy Key。
- **协议对齐**：上游 Open WebUI ≥ 0.6 已提供 OpenAI 兼容路由 `/api/v1/*`，旧版本只有内部路由 `/api/*`。本项目启动时会**自动探测**并记住可用前缀，请求返回 404（路由不存在）时还会自动回退到另一个前缀。
- **响应规范化**：`/v1/models` 会把上游模型对象收敛成标准的 `{id, object, created, owned_by}`，并按通用模板白名单透出扩展字段：`max_context_length` / `context_length`（`max_model_len` 作为兼容别名保留）、`quantization`（从模型名解析，如 `NVFP4`）、`capabilities`（含派生的 `function_calling`）与 `description`；私有字段（`user_id`、`access_grants`、`permission`、`urlIdx` 等）一律不透出。

## 特性

- **OpenAI 兼容**：`/v1/models`、`/v1/chat/completions`（含流式 SSE）、`/v1/embeddings`，以及未实现路径的 `/v1/*` 兜底透传（同样具备前缀回退）。
- **自动适配上游版本**：`auto` / `v1` / `legacy` 三种上游 API 风格，启动探测 + 请求级回退。
- **健壮的流式转发**：客户端断开时主动关闭上游连接，不会把连接挂到超时；正确剔除逐跳响应头。
- **OpenAI 风格错误体**：返回 `{"error": {"message", "type", "code"}}`，而不是 FastAPI 默认的 `{"detail": ...}`，客户端能正常显示错误原因。
- **启动自检**：一次 `GET /models` 同时完成前缀探测与凭证校验，失效会直接提示重新登录，不用等第一次调用才发现问题。
- **可选 CORS**：配置 `PROXY_CORS_ORIGINS` 后浏览器页面可以直连本代理（预检自动应答）；默认关闭，不扩大暴露面。
- **模型别名**：通过 `MODEL_ALIASES` 把客户端请求的模型名映射到上游真实模型名。
- **凭证脱敏**：日志里只打印 Token 前缀与长度，不落盘完整凭证。

## 目录结构

```
.
├── app.py                  # FastAPI 路由、OpenAI 兼容层、CLI 入口
├── config.py               # 全部配置项（环境变量 / .env）
├── session_store.py        # 凭证读写 + Playwright 浏览器登录抓取
├── upstream.py             # 上游转发：连接池、前缀探测、流式
├── requirements.txt        # 运行服务的最小依赖
├── requirements-browser.txt# 可选：浏览器登录所需的 Playwright
├── .env.example            # 配置模板
└── tests/
    ├── mock_openwebui.py   # 标准库实现的 Open WebUI 模拟器
    ├── test_smoke.py       # 端到端冒烟测试（mock 上游 + 真实启动代理）
    └── test_units.py       # 纯逻辑单元测试
```

## 快速开始

### 1. 安装依赖

Python 3.9+：

```bash
pip install -r requirements.txt
```

如果要用「浏览器登录抓取凭证」这条路径，还需要：

```bash
pip install -r requirements-browser.txt
playwright install chromium
```

### 2. 配置

```bash
cp .env.example .env
```

至少修改这两项：

```env
OPEN_WEBUI_BASE_URL=http://your-open-webui-domain.com
PROXY_API_KEY=sk-your-custom-proxy-key
```

### 3. 首次登录

```bash
python app.py
```

首次运行（或 `session.json` 不存在时）会：

1. 弹出 Chromium 窗口；
2. 你在窗口里手动完成 Open WebUI 登录；
3. 脚本监听发往上游 `/api/*` 的请求，一旦抓到 `Bearer` Token 或会话 Cookie 就写入 `session.json`；
4. 静默观察 `LOGIN_QUIET_PERIOD` 秒确认没有更新的 Token 后，自动关闭浏览器并启动代理服务。

> 判定"登录成功"的条件是**发往 `/api/` 的请求携带了身份信息**，因此首页首次加载产生的匿名 Cookie 不会被误判。

常用命令：

```bash
python app.py              # 启动服务（必要时先登录）
python app.py --login      # 强制重新登录，刷新凭证
python app.py --check      # 只校验凭证与上游连通性，打印摘要后退出
python app.py --probe      # 强制重探所有模型的思考挡位并刷新缓存
python app.py --port 9000  # 临时覆盖监听端口
```

输出语言（日志、启动横幅、CLI 帮助、错误消息）：

```bash
python app.py --lang zh    # 强制中文输出
python app.py --lang en    # 强制英文输出
python app.py --lang auto  # 跟随系统语言（默认），检测不到时用英文
```

选择优先级：`--lang` 参数 > 系统语言检测 > 英文。

### 4. 接入客户端

| 配置项          | 值                          |
| ------------ | -------------------------- |
| API Base URL | `http://127.0.0.1:8000/v1` |
| API Key      | `.env` 里的 `PROXY_API_KEY`  |
| 模型名          | 见下方 `curl` 输出              |

> **注意**：客户端填的永远是 `http://127.0.0.1:8000/v1`（本机）或 `http://<本机IP>:8000/v1`（局域网设备）。`PROXY_HOST=0.0.0.0` 只是服务端的监听配置，**不要**把它当 API 地址填——Electron/Chromium 客户端连 `0.0.0.0` 会报 `net::ERR_ADDRESS_INVALID`（典型症状：模型列表能拉到、一对话就报错）。

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer sk-your-custom-proxy-key"
```

Python：

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

Node.js：

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

## API 端点

| 方法   | 路径                     | 鉴权 | 说明                        |
| ---- | ---------------------- | -- | ------------------------- |
| GET  | `/`                    | 否* | 服务信息与已注册端点；上游地址仅在带 Key 时返回 |
| GET  | `/healthz`             | 否* | 健康检查恒 200；上游地址与探测前缀仅在带 Key 时返回 |
| GET  | `/v1/models`           | 是  | 模型列表，已规范化为 OpenAI 结构，并附带思考挡位信息   |
| POST | `/v1/chat/completions` | 是  | 对话补全，支持 `stream: true`    |
| POST | `/v1/embeddings`       | 是  | 向量嵌入（上游需支持）               |
| ANY  | `/v1/{path}`           | 是  | 兜底透传，转发到上游同路径             |

鉴权支持 `Authorization: Bearer <key>` 与 `X-API-Key: <key>` 两种写法。若 `PROXY_API_KEY` 留空则不鉴权。

\* `/` 与 `/healthz` 保持免鉴权（探针与就绪检查友好），但响应里的 `upstream` / `upstream_prefix` 字段只在请求携带有效 Key（或未启用鉴权）时返回，避免公网部署时泄漏上游内网域名。

## 配置项

| 环境变量                  | 默认值                     | 说明                                     |
| --------------------- | ----------------------- | -------------------------------------- |
| `OPEN_WEBUI_BASE_URL` | `http://localhost:8080` | 上游地址，必须带 `http://` 或 `https://`，结尾不带斜杠     |
| `UPSTREAM_API_STYLE`  | `auto`                  | `auto` / `v1` / `legacy`               |
| `UPSTREAM_VERIFY_SSL` | `true`                  | 上游为自签证书时设 `false`                      |
| `UPSTREAM_TRUST_ENV`  | `true`                  | 是否读取系统代理环境变量；上游在本机或内网而系统配了代理时设 `false` |
| `PROXY_HOST`          | `0.0.0.0`               | 服务监听地址（仅服务端配置；`0.0.0.0` = 所有网卡，不是客户端要填的 API 地址） |
| `PROXY_PORT`          | `8000`                  | 监听端口                                   |
| `PROXY_API_KEY`       | 空                       | 对外访问密钥，留空表示不鉴权                         |
| `PROXY_CORS_ORIGINS`  | 空                       | 允许跨域的来源列表（逗号分隔）；留空不启用 CORS             |
| `REQUEST_TIMEOUT`     | `300`                   | 上游请求总超时（秒）                             |
| `CONNECT_TIMEOUT`     | `10`                    | 连接上游超时（秒）                              |
| `SESSION_FILE`        | `session.json`          | 凭证文件路径                                 |
| `REASONING_CACHE_FILE` | `reasoning_cache.json` | 思考挡位缓存文件路径                             |
| `REASONING_PROBE_CONCURRENCY` | `4`             | 挡位探测的并发数                               |
| `REASONING_PROBE_TIMEOUT`     | `30`            | 单个模型挡位探测的超时（秒）                        |
| `REASONING_PROBE_WAIT`        | `5`             | `/v1/models` 发现挡位缺失时等待探测完成的最长秒数，`0` = 立即返回 |
| `MODEL_ALIASES`       | 空                       | JSON 对象，模型名映射                          |
| `LOG_LEVEL`           | `INFO`                  | `CRITICAL` / `ERROR` / `WARNING` / `INFO` / `DEBUG` / `TRACE`，非法值回退 `INFO` |
| `DEBUG`               | `false`                 | 为 `true` 时等价于 `LOG_LEVEL=DEBUG`        |
| `LOGIN_TIMEOUT`       | `600`                   | 浏览器登录最长等待秒数                            |
| `LOGIN_QUIET_PERIOD`  | `6`                     | 抓到凭证后继续观察的秒数                           |
| `LOGIN_HEADLESS`      | `false`                 | 是否无头启动浏览器                              |

## 思考挡位探测（reasoning_effort）

`/v1/models` 会为每个模型附带一个 `reasoning` 字段，声明该模型支持的思考挡位：

```json
{
  "id": "GLM-5.3-Flash",
  "object": "model",
  "...": "...",
  "reasoning": {
    "supported_efforts": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
    "default_effort": "medium",
    "default_enabled": true,
    "mandatory": false
  }
}
```

**探测原理**：上游（vLLM 等）把 `reasoning_effort` 声明为 Literal 枚举校验。本代理向每个模型发送一个携带哨兵值 `"__probe__"` 的最小补全请求（`max_tokens=1`），上游校验失败返回 400，错误文本恰好枚举该模型接受的全部挡位（`Input should be 'none', 'low', 'medium' or 'high'`）。校验发生在生成之前，因此**一次探测的 token 成本为零**。

**缓存策略**：挡位与模型挂钩，探测结果持久化到 `reasoning_cache.json`。之后每次启动，只要模型列表没有变化就直接复用缓存（日志显示"跳过探测"）；模型列表有增删时，只探测新增的模型。运行期间 `/v1/models` 发现缓存未覆盖的新模型时，会触发后台补探，并默认最多等待 5 秒（`REASONING_PROBE_WAIT`）让探测完成再返回——探测通常亚秒级完成，客户端首次请求即可拿到完整挡位；超时则先返回现有内容，字段在下次请求出现。

**字段推导规则**（探测只能揭示"接受哪些值"，其余字段为启发式推导）：

- `supported_efforts`：上游校验接受的确切集合，按 `none → max` 规范顺序输出；
- `default_effort`：支持 `medium` 则为 `medium`，否则取中位数挡位；
- `mandatory`：集合中没有 `none` 时为 `true`，即无法关闭思考；
- `default_enabled`：恒为 `true`（字段被接受即思考默认开启）。

强制全量重探：`python app.py --probe`（适用于上游重新部署后模型名未变、挡位却变了的情况）。

## 凭证（session.json）

`session.json` 里就是浏览器登录态的请求头，结构如下：

```json
{
  "Authorization": "Bearer eyJhbGciOiJIUzI1NiIs...",
  "Cookie": "",
  "User-Agent": "Mozilla/5.0 ...",
  "captured_at": 1756000000.0,
  "base_url": "https://your-open-webui-domain.com"
}
```

`Authorization` 与 `Cookie` 至少要有其一。如果你能从浏览器 F12 里拿到 Open WebUI 的 JWT，也可以**手写这个文件**，完全跳过浏览器登录这一步。

> `session.json` 已在 `.gitignore` 中，切勿提交。POSIX 系统下写入时会自动收敛为 `600` 权限。

## 测试

配套测试会拉起一个内置的 Open WebUI 模拟器，再真实启动本代理，覆盖三种上游风格：

```bash
pip install -r requirements.txt

python tests/test_units.py    # 纯逻辑单测，秒级完成，无需联网
python tests/test_smoke.py    # 端到端：拉起 mock 上游 + 真实启动本代理
```

`test_units.py` 覆盖：凭证序列化与旧格式兼容、登录信号判定、模型列表规范化、配置解析容错、错误响应结构。

`test_smoke.py` 覆盖：健康检查、鉴权拒绝、模型列表规范化、非流式/流式对话、上游错误透传、参数校验、embeddings、兜底透传、凭证缺失时的 503；并对 `auto` / `v1` / `legacy` 三种上游风格各跑一遍。

## 故障排除

**Q: 浏览器跳到了校园网 / 公司网认证页，还没输入账号就提示"凭证已保存"**

这类强制门户（captive portal）未完成认证时，所有请求都会被网关重定向；页面加载早期 Open WebUI 前端也可能用 localStorage 里的过期旧 Token 发探测请求——两者抓到的"凭证"都不是有效登录态。本项目已内置校验：抓到的凭证必须能通过上游一次真实鉴权（直连上游返回 2xx）才会保存，否则会提示"未通过上游校验"并继续等待。请在认证页完成校园网登录，再回到 Open WebUI 完成应用登录即可。

**Q: 客户端（Cherry Studio / Chatbox 等 Electron 应用）对话报 `net::ERR_ADDRESS_INVALID`**

错误发生在客户端的 Chromium 网络栈，请求根本没到达本代理。`ERR_ADDRESS_INVALID` 表示连接的目标地址非法，最常见原因是把 `PROXY_HOST=0.0.0.0` 当成 API 地址填进了客户端——Chromium 拒绝连接 `0.0.0.0`。把 API Base URL 改成 `http://127.0.0.1:8000/v1` 即可。"模型列表正常、对话报错"的分裂现象是因为 Windows 上 Node 层连 `0.0.0.0` 会被当作本机成功，而 Chromium 在连接前就直接拒绝。若地址无误仍报错，检查客户端自身是否配置了 HTTP 代理导致回环请求被送走，以及对话时本代理日志里是否有请求进来（没有即客户端未连上）。

**Q: 上游返回 401 / 403，日志提示"凭证已失效"**

Open WebUI 的 JWT 有有效期（默认由服务端的 `JWT_EXPIRES_IN` 控制，通常是数天）。删除 `session.json` 后重新登录：

```bash
python app.py --login
```

**Q: 所有请求都 404，日志提示"所有候选前缀均返回 404"**

`OPEN_WEBUI_BASE_URL` 可能没指向 Open WebUI（例如填成了 Ollama 的地址），或者上游版本非常旧。先用 `python app.py --check` 看探测结果，必要时显式设 `UPSTREAM_API_STYLE=legacy`。

**Q: 上游在内网/本机，但请求超时或 502**

系统里可能配了 `HTTP_PROXY`/`HTTPS_PROXY`，httpx 默认会遵守它们，导致回环请求被送去走代理。设 `UPSTREAM_TRUST_ENV=false`。

**Q: 推理模型（DeepSeek-R1 / QwQ 等）的思考过程能返回吗？**

能。本代理对**请求体和响应体都是全量透传**，不解析、不裁剪对话内容：`reasoning_effort` / `temperature` 等请求参数原样转发给上游，`reasoning_content`（DeepSeek 风格）、`reasoning`（Open WebUI 新版风格）以及正文中的 `<think>` 标签在流式与非流式下都原样到达客户端（有回归测试保障）。思考内容是否可见最终取决于两点：上游 Open WebUI 是否返回（需使用推理模型且版本支持），以及客户端是否识别展示（Cherry Studio / Chatbox 均支持）。

**Q: 流式输出被缓冲，客户端一次性收到全部内容**

反向代理（Nginx 等）需要关闭缓冲。本项目已在响应头里带上 `X-Accel-Buffering: no`，Nginx 侧请确认 `proxy_buffering off;`。

**Q: 可以在无 GUI 的服务器上跑吗？**

浏览器登录这一步本质上需要人工交互，服务器无 GUI 时需要 Xvfb 等虚拟显示。更推荐的做法是在本地登录一次，把生成的 `session.json` 复制到服务器。

## 安全建议

- 一定要设置 `PROXY_API_KEY`，否则任何能访问该端口的人都能借用你的 Open WebUI 身份。
- 不要把 `session.json` 提交进版本库或放进容器镜像。
- 尽量只监听 `127.0.0.1`，需要对外时套一层反向代理并启用 HTTPS。
- 本代理会原样转发请求体，请不要把它暴露给不可信的调用方。
- 启用 CORS 时（`PROXY_CORS_ORIGINS`）请按需列出最小来源集合，避免使用 `*`。

## 贡献

欢迎提交 Issue 和 PR。改动涉及转发行为时，请同步更新 `tests/` 下的用例，并确保两条命令都全绿：

```bash
python tests/test_units.py
python tests/test_smoke.py
```

## 许可证

本项目采用 [MIT License](LICENSE)。
