import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 1. 初始化+跨域（最基础配置）
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 仅测试用
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 核心：SSE接口（只推固定数据）
@app.get("/simple_stream")
async def simple_stream():
    async def event_generator():
        # 模拟推5条消息，每秒1条
        for i in range(5):
            # ✅ 核心：SSE固定格式 data: 内容\n\n
            yield f"data: 这是第{i+1}条测试消息\n\n"
            await asyncio.sleep(1)  # 每秒推1条

    # ✅ 核心：StreamingResponse + media_type=text/event-stream
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)