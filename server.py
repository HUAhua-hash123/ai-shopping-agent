"""AI 导购智能体 —— FastAPI 服务（网页聊天 + SSE 流式）

提供两个能力：
1. GET  /         返回聊天页面（index.html）
2. POST /api/chat 流式对话（SSE）：复用 agent.py 里的 LangGraph Agent

运行：
    python server.py
然后浏览器打开 http://127.0.0.1:8000 即可对话。
"""

from __future__ import annotations

import json
import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import app

load_dotenv()

web = FastAPI(title="AI 导购智能体")

# 访问口令：来自环境变量 DEMO_PASS，默认 "demo123"（部署时务必修改）
ACCESS_PASS = os.getenv("DEMO_PASS", "demo123")


class ChatRequest(BaseModel):
    message: str


class UnlockRequest(BaseModel):
    passwd: str = ""


@web.get("/")
async def index():
    return FileResponse("index.html")


@web.post("/api/unlock")
async def unlock(req: UnlockRequest):
    return {"ok": req.passwd == ACCESS_PASS}


@web.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncGenerator[str, None]:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        inputs = {"messages": [HumanMessage(content=req.message)]}

        async for event in app.astream(inputs, stream_mode="values"):
            for msg in event.get("messages", []):
                # 工具调用过程：作为「步骤」事件推给前端展示
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for call in msg.tool_calls:
                        payload = {
                            "type": "tool_call",
                            "name": call["name"],
                            "args": call["args"],
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if isinstance(msg, ToolMessage):
                    payload = {
                        "type": "tool_result",
                        "content": str(msg.content),
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                # 最终答案：流式逐段推送
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    text = "".join(msg.content) if isinstance(msg.content, list) else str(msg.content)
                    payload = {"type": "answer", "content": text}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(web, host="127.0.0.1", port=8000)
