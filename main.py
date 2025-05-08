import secrets
import json
from authlib.integrations.starlette_client import OAuth
from starlette.applications import Starlette
from starlette.responses import JSONResponse, RedirectResponse, StreamingResponse
from starlette.routing import Route
from starlette.middleware.sessions import SessionMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
import uvicorn
import os
from agent import agent_service

# Configuration
class Config:
    PORT = os.getenv("PORT", "8080")
    FQDN = os.getenv("FQDN", "http://localhost")
    CLIENT_ID = os.getenv("CLIENT_ID", "your_client_id")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET", "your_client_secret")
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(16))

config = Config()

oauth = OAuth()
oauth.register(
    name="github",
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET,
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "user:email"},
)

# OAuth Handlers
async def pre_auth(request):
    redirect_uri = f"{config.FQDN}/auth/callback"
    return await oauth.github.authorize_redirect(request, redirect_uri)

async def post_auth(request):
    try:
        token = await oauth.github.authorize_access_token(request)
        user_info = await oauth.github.get("user", token=token)
        user_data = user_info.json()
        return JSONResponse({"message": "Authorization successful", "user": user_data})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# Agent Handler
async def agent_handler(request):
    # return await agent_service.generate_completion(request)
    return await agent_service.function_calling(request)
    # return await sse(request)

async def generator_test():
    event_payload = {"type":"action","title":"Create Issue","message":"Are you sure you want to create an issue in repository copilot-extensions/function-calling-extension with the title \"Issue in function-calling-extension\" and the content \"I have encountered an error while using the function-calling-extension.\"","confirmation":{"owner":"copilot-extensions","repo":"function-calling-extension","title":"Issue in function-calling-extension","body":"I have encountered an error while using the function-calling-extension."}}
    data_payload = {"choices":[{"index":0,"delta":{"role":"assistant","content":"I have created an issue in the \"copilot-extensions/function-calling-extension\" repository. The issue title is \"Issue in function-calling-extension\" and the issue body is \"I have encountered an error while using the function-calling-extension.\""}}]}

    await asyncio.sleep(1)

    yield dict(event="copilot_confirmation", data=json.dumps(event_payload))
    yield dict(data=json.dumps(data_payload))
    yield dict(data="[DONE]")

async def sse(request):
    return EventSourceResponse(generator_test())

async def numbers(minimum, maximum):
    for i in range(minimum, maximum + 1):
        await asyncio.sleep(0.9)
        yield dict(data=i)

# Routes
routes = [
    Route("/auth/authorization", pre_auth),
    Route("/auth/callback", post_auth),
    Route("/agent", agent_handler, methods=["POST"]),
    Route("/", lambda request: JSONResponse({"message": "Welcome to the API!"})),
    Route("/sse", sse),
    Route("/health", lambda request: JSONResponse({"status": "ok"})),
]

# Application Setup
app = Starlette(debug=True, routes=routes)
app.add_middleware(SessionMiddleware, secret_key=config.SECRET_KEY)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(config.PORT))