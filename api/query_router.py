import asyncio
from functools import partial

import httpx
import uvicorn
from fastapi import FastAPI, Depends, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.deps import (
    get_agent_query_service,
    get_query_mode_service,
    get_query_service,
)
from urllib.parse import unquote, quote
from core.deps import get_query_service
from core.paths import get_front_page_dir
from services.agent_query_service import AgentQueryService
from services.query_mode_service import QueryModeService
from services.query_service import QueryService
from schema.query_schema import (
    HistoryResponse,
    QueryMode,
    QueryRequest,
    QueryResponse,
    StreamSubmitResponse,
)
from utils.sse_util import create_sse_queue, push_sse_event, sse_generator

app = FastAPI(description="", version="v1.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件挂载
static_resource_page_dir = get_front_page_dir()
if static_resource_page_dir:
    # html=True：访问 /chat/ 时自动返回 index.html（否则目录请求返回 404）
    app.mount("/chat", StaticFiles(directory=static_resource_page_dir, html=True))

@app.post("/query", response_model=QueryResponse | StreamSubmitResponse)
async def query(
    background_tasks: BackgroundTasks,
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
    agent_query_service: AgentQueryService = Depends(
        get_agent_query_service
    ),
    query_mode_service: QueryModeService = Depends(
        get_query_mode_service
    ),
):
    """提交查询。

    阶段 D 先解析最终模式，再决定使用固定 RAG 还是 Agent。
    """

    session_id = request.session_id
    if not session_id:
        session_id = query_service.gener_session()
    task_id = query_service.gener_task_id()
    is_stream = request.is_stream
    query_text = request.query

    # 根据请求模式、配置和灰度比例解析最终模式。
    routing = query_mode_service.resolve(
        requested_mode=request.mode,
        session_id=session_id,
    )

    # 流式请求先创建队列，并发送实际模式信息。
    # 当前前端暂不监听该事件，后续模式按钮和动态渲染再使用。
    if is_stream:
        create_sse_queue(task_id)
        push_sse_event(
            task_id=task_id,
            event="query_mode",
            data={
                "requested_mode": routing.requested_mode.value,
                "effective_mode": routing.effective_mode.value,
                "reason": routing.reason,
                "canary_bucket": routing.canary_bucket,
            },
        )

    # ==================== Agent 模式 ====================
    if routing.effective_mode == QueryMode.AGENT:
        if is_stream:
            background_tasks.add_task(
                agent_query_service.run_agent_query,
                task_id,
                query_text,
                session_id,
                True,
                routing.max_steps,
                False,
            )
            return StreamSubmitResponse(
                message="正在查询",
                session_id=session_id,
                task_id=task_id,
            )

        # 同步 Agent 查询在线程池中执行，避免阻塞事件循环。
        loop = asyncio.get_event_loop()
        func_with_args = partial(
            agent_query_service.run_agent_query,
            task_id,
            query_text,
            session_id,
            False,
            routing.max_steps,
            True,
        )

        try:
            await loop.run_in_executor(None, func_with_args)
            answer = agent_query_service.get_task_result(task_id)
            return QueryResponse(
                message="查询成功",
                session_id=session_id,
                answer=answer,
            )
        except Exception:
            # 未允许回退时，返回统一的 Agent 失败信息。
            if not routing.fallback_to_legacy:
                return QueryResponse(
                    message="查询失败",
                    session_id=session_id,
                    answer="很抱歉，Agent 查询失败",
                )

            # 允许回退时继续执行下面的固定 RAG 分支。

    # ==================== 旧版固定 RAG ====================
    if is_stream:
        background_tasks.add_task(
            query_service.run_query_graph,
            task_id,
            query_text,
            session_id,
            is_stream,
        )
        return StreamSubmitResponse(
            message="正在查询",
            session_id=session_id,
            task_id=task_id
        )
    else:
        loop = asyncio.get_event_loop()
        func_with_args = partial(
            query_service.run_query_graph,
            task_id,
            query_text,
            session_id,
            is_stream
        )
        await loop.run_in_executor(None, func_with_args)
        answer = query_service.get_task_result(task_id)
        return QueryResponse(
            message="查询成功",
            session_id=session_id,
            answer=answer
        )

@app.get("/stream/{task_id}")
async def stream(task_id: str, request: Request):
    return StreamingResponse(
        content=sse_generator(task_id, request),
        media_type="text/event-stream"
    )


@app.get("/status/{task_id}")
async def status(task_id:str):
    return {
        "status": "processing",
        "done_list": ["entry_node", "document_spliter"],
        "running_list": ["embedding_chunks"],
        "durations": {
            "entry_node": 0.5,
            "document_spliter": 2.3
        }
    }

@app.get("/history/{session_id}",response_model=HistoryResponse)
async def get_history(
        session_id:str,
        query_service: QueryService = Depends(get_query_service)
        ):
    history_list = query_service.get_history(session_id=session_id)
    return HistoryResponse(
        session_id= session_id,
        items=history_list
    )


@app.get("/api/proxy/image")
async def proxy_image(url: str = Query(...)):
    # 第1步：FastAPI 已自动解码过一次，这里再 unquote 消除二次编码
    url = unquote(url)

    # 第2步：安全校验，只允许请求你自己的 MinIO
    if not url.startswith("http://192.168.10.100:9000"):
        return {"error": "invalid url"}

    # 第3步：重新编码非 ASCII 字符（中文等），让 httpx 能正确发送请求
    # safe 参数保留 URL 结构字符，只编码中文
    url = quote(url, safe=":/?#[]@!$&'()*+,;=%")

    # 第4步：后端在内部请求 MinIO
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10.0)
            if resp.status_code != 200:
                return {"error": f"image not found: {resp.status_code}"}

            content_type = resp.headers.get("content-type", "image/jpeg")
            return StreamingResponse(iter([resp.content]), media_type=content_type)
    except Exception as e:
        return {"error": str(e)}

@app.delete("/history/{session_id}")
async def clear_history(
        session_id:str,
        query_service: QueryService = Depends(get_query_service)
):
    deleted_count = query_service.delete_history(session_id=session_id)
    return {
    "message": "历史记录去除成功",
    "deleted_count": deleted_count
    }

if __name__ == "__main__":
    uvicorn.run(
        app = app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )






















