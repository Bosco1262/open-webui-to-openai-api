# 🌐 Open WebUI Browser Proxy

将无法直接通过 API Key 访问的 **Open WebUI** 实例，通过捕获浏览器登录态，反向代理为 **兼容 OpenAI 格式的 API**。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 项目简介

有些自部署的 Open WebUI 实例由于网络限制、SSO 单点登录、或版本限制，无法生成或使用标准的 API Key 进行对接。本项目旨在解决这一痛点：

它利用 **Playwright** 弹出真实浏览器让你完成手动登录，后台自动抓取登录后的 Authorization 头和 Cookie，随后在本地启动一个兼容 OpenAI 格式的代理服务。你可以使用任何支持 OpenAI API 的客户端（如 Chatbox, Cherry Studio, Langchain 等）直接连接它。

## ✨ 核心功能

- 🧩 **零配置接入**：无需折腾 API Token，只要浏览器能登录就能用。

- 🔌 **OpenAI 兼容**：完美支持 /v1/models 和 /v1/chat/completions (包含流式输出)。

- 🛡️ **凭证隔离**：对外提供自定义 Proxy Key 校验，保护你的代理服务不被滥用。

- 🔄 **自动抓取**：登录成功后自动提取凭证并保存，无需手动 F12 抓包。

## 🛠️ 快速开始

### 1. 克隆仓库

```bash

git clone https://github.com/你的用户名/open-webui-browser-proxy.git

cd open-webui-browser-proxy

```

### 2. 安装依赖

建议在 Python 3.9+ 环境下运行：

```bash

pip install -r requirements.txt

# 安装 Playwright 浏览器内核 (只需执行一次)

playwright install chromium

```

### 3. 配置环境变量

复制示例配置并修改：

```bash

cp .env.example .env

```

编辑 .env 文件：

```env

OPEN_WEBUI_BASE_URL=http://your-open-webui-domain.com

PROXY_API_KEY=sk-your-custom-proxy-key

```

### 4. 启动服务

```bash

python app.py

```

**首次运行会出现以下流程：**

1. 自动弹出一个 Chromium 浏览器窗口。

2. 请在该窗口内手动完成你的 Open WebUI 登录。

3. 脚本在后台监听到登录网络请求后，会自动提取 Token 并保存为 session.json。

4. 浏览器自动关闭，本地 API 代理服务启动。

## 📖 使用方法

服务启动后，即可在第三方客户端中按以下格式配置：

- **API Base URL**: http://127.0.0.1:8000/v1

- **API Key**: 你在 .env 中设置的 PROXY_API_KEY

- **模型名称**: 可通过 curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer <你的KEY>" 获取

### Python 示例

```python

from openai import OpenAI

client = OpenAI(

    api_key="sk-your-custom-proxy-key",

    base_url="http://127.0.0.1:8000/v1"

)

response = client.chat.completions.create(

    model="llama3",

    messages=[{"role": "user", "content": "Hello!"}],

    stream=True

)

for chunk in response:

    if chunk.choices[0].delta.content is not None:

        print(chunk.choices[0].delta.content, end="")

```

## 🔧 故障排除

**Q: 提示 401 Upstream authentication failed 怎么办？**

A: 这说明你的浏览器登录态过期了，或者目标服务端清退了会话。只需删除项目目录下的 session.json 文件，重新运行 python app.py 登录一次即可。

**Q: 可以在服务器上无头运行吗？**

A: 不行。本项目的核心就是弹窗模拟人工登录，如果你能直接获取 Token，建议直接使用标准的 API 接入方式。

## 🤝 贡献

欢迎提交 Issue 和 PR！如果你有针对特定版本 WebUI 的适配改进，请随时提交。

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。

```

```
