import asyncio
import json
import os
import logging
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

load_dotenv()

# ---------------------------- 配置 ---------------------------- #
class Settings:
    OPEN_WEBUI_BASE_URL: str = os.getenv("OPEN_WEBUI_BASE_URL", "http://localhost:8080").rstrip("/")
    PROXY_HOST: str = os.getenv("PROXY_HOST", "0.0.0.0")
    PROXY_PORT: int = int(os.getenv("PROXY_PORT", "8000"))
    PROXY_API_KEY: str = os.getenv("PROXY_API_KEY", "")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

settings = Settings()
SESSION_FILE = "session.json"

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webui-proxy")

# ---------------------------- 浏览器登录抓取 ---------------------------- #
async def perform_browser_login():
    """启动浏览器让用户登录，抓取请求头中的认证信息"""
    if not async_playwright:
        logger.error("未安装 playwright，请运行: pip install playwright && playwright install chromium")
        return False

    logger.info("="*50)
    logger.info(f"正在打开浏览器，请访问 {settings.OPEN_WEBUI_BASE_URL} 并登录...")
    logger.info("登录成功后，脚本会自动捕获凭证并关闭浏览器。")
    logger.info("="*50)

    captured = asyncio.Event()

    async def on_request(request):
        """监听发往目标服务器的请求，抓取认证头"""
        try:
            if not request.url.startswith(settings.OPEN_WEBUI_BASE_URL):
                return
            
            headers = request.headers
            auth_header = headers.get("authorization", "")
            cookie_header = headers.get("cookie", "")
            
            # 只要请求里有 Bearer Token 或者 Cookie，就认为登录成功并抓取
            if "bearer" in auth_header.lower() or cookie_header:
                auth_data = {
                    "Authorization": auth_header,
                    "Cookie": cookie_header,
                    "User-Agent": headers.get("user-agent", "Mozilla/5.0")
                }
                with open(SESSION_FILE, "w", encoding="utf-8") as f:
                    json.dump(auth_data, f, indent=2)
                
                logger.info(f"✅ [捕获成功] 已提取到登录凭证！")
                if auth_header:
                    logger.info(f"   -> 抓取到 Token: {auth_header[:20]}...")
                if cookie_header:
                    logger.info(f"   -> 抓取到 Cookie (长度: {len(cookie_header)})")
                
                captured.set()
        except Exception as e:
            logger.warning(f"抓取请求头时发生错误: {e}")

    async with async_playwright() as p:
        # headless=False 表示弹出可见的浏览器窗口
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        page.on("request", on_request)
        await page.goto(settings.OPEN_WEBUI_BASE_URL)
        
        try:
            # 等待 10 分钟让用户完成登录
            await asyncio.wait_for(captured.wait(), timeout=600)
            await asyncio.sleep(2)  # 多等2秒确保后续Token全部更新
            await browser.close()
            return True
        except asyncio.TimeoutError:
            logger.error("❌ 登录超时(10分钟)，请重新运行程序。")
            await browser.close()
            return False

def load_session_headers() -> Dict[str, str]:
    """从 session.json 读取抓取到的请求头"""
    if not os.path.exists(SESSION_FILE):
        raise HTTPException(status_code=401, detail="未找到登录凭证 session.json，请重启代理脚本并完成浏览器登录。")
    
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------- FastAPI 代理服务 ---------------------------- #
app = FastAPI(title="Browser to API Proxy", version="1.0.0")
_http_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.REQUEST_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _http_client

@app.on_event("shutdown")
async def shutdown_event():
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()

def verify_proxy_key(request: Request):
    if not settings.PROXY_API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if auth[len("Bearer "):].strip() != settings.PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid proxy API key")

def build_upstream_headers(original_headers: Dict[str, str]) -> Dict[str, str]:
    """构造转发给上游的真实请求头"""
    session = load_session_headers()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": session.get("Authorization", ""),
    }
    if session.get("Cookie"):
        headers["Cookie"] = session["Cookie"]
    if session.get("User-Agent"):
        headers["User-Agent"] = session["User-Agent"]
    
    return headers

# ---------------------------- 路由 ---------------------------- #
@app.get("/v1/models", tags=["openai"])
async def list_models(_: None = Depends(verify_proxy_key)):
    client = await get_http_client()
    headers = build_upstream_headers({})
    try:
        r = await client.get(f"{settings.OPEN_WEBUI_BASE_URL}/api/models", headers=headers)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if r.status_code in [401, 403]:
        logger.error("上游返回认证失败！Session 可能已过期，请删除 session.json 并重新运行脚本登录。")
        raise HTTPException(status_code=401, detail="Upstream authentication failed. Session expired.")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return r.json()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False
    model_config = {"extra": "allow"}

@app.post("/v1/chat/completions", tags=["openai"])
async def chat_completions(request: Request, body: ChatCompletionRequest, _: None = Depends(verify_proxy_key)):
    client = await get_http_client()
    payload: Dict[str, Any] = body.model_dump(exclude_none=True)
    upstream_headers = build_upstream_headers(dict(request.headers))
    is_stream = bool(payload.get("stream", False))

    try:
        upstream_req = client.build_request(
            "POST",
            f"{settings.OPEN_WEBUI_BASE_URL}/api/chat/completions",
            headers=upstream_headers,
            json=payload,
        )
        upstream_resp = await client.send(upstream_req, stream=is_stream)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if upstream_resp.status_code in [401, 403]:
        logger.error("上游返回认证失败！Session 可能已过期，请删除 session.json 并重新运行脚本登录。")
        raise HTTPException(status_code=401, detail="Upstream authentication failed. Session expired.")
    
    if upstream_resp.status_code >= 400:
        body_text = upstream_resp.text if not is_stream else await upstream_resp.aread()
        return JSONResponse(status_code=upstream_resp.status_code, content={"error": body_text})

    if not is_stream:
        content = await upstream_resp.aread()
        parsed = json.loads(content)
        return JSONResponse(content=parsed)

    async def stream_generator():
        try:
            async for chunk in upstream_resp.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )

# ---------------------------- 启动入口 ---------------------------- #
if __name__ == "__main__":
    # 1. 如果 session.json 不存在，先触发浏览器登录
    if not os.path.exists(SESSION_FILE):
        login_success = asyncio.run(perform_browser_login())
        if not login_success:
            exit(1)
    
    # 2. 启动 API 代理服务
    logger.info("="*50)
    logger.info(f"🚀 代理服务已启动: http://{settings.PROXY_HOST}:{settings.PROXY_PORT}")
    logger.info(f"🎯 上游目标: {settings.OPEN_WEBUI_BASE_URL}")
    if settings.PROXY_API_KEY:
        logger.info(f"🔑 访问密码 : {settings.PROXY_API_KEY}")
    logger.info("="*50)
    
    uvicorn.run(
        "app:app",
        host=settings.PROXY_HOST,
        port=settings.PROXY_PORT,
        log_level="debug" if settings.DEBUG else "info",
        reload=False
    )