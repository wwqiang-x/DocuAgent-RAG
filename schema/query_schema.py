"""查询请求和响应 Schema。"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    """查询模式。

    legacy：使用当前固定 RAG 流程。
    agent：使用阶段 B 构建的 AgentRuntime。
    auto：由后端配置或灰度比例自动决定。
    """

    LEGACY = "legacy"
    AGENT = "agent"
    AUTO = "auto"


class QueryRequest(BaseModel):
    """查询请求。"""

    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(
        None,
        description="会话 ID，不传则自动生成",
    )
    is_stream: bool = Field(
        False,
        description="是否流式返回",
    )

    # 默认 auto。
    # 在默认配置下 auto 会解析为 legacy，因此现有调用方行为不变。
    mode: QueryMode = Field(
        QueryMode.AUTO,
        description="查询模式：legacy、agent 或 auto",
    )


class QueryResponse(BaseModel):
    """同步查询响应。"""

    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话 ID")
    answer: str = Field("", description="生成的答案")


class StreamSubmitResponse(BaseModel):
    """流式查询提交响应。"""

    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话 ID")
    task_id: str = Field(..., description="用于建立 SSE 连接的任务 ID")


class HistoryItem(BaseModel):
    """单条历史消息。"""

    id: str = Field("", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    item_names: List[str] = Field(default_factory=list)
    ts: Optional[float] = None


class HistoryResponse(BaseModel):
    """历史记录响应。"""

    session_id: str
    items: List[HistoryItem]
