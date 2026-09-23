"""证据领域模型。

Evidence 用于统一描述本地向量检索、网络搜索和重排序结果。
该模型不依赖 Agent，因此普通 RAG 流程未来也可以直接复用。
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    """证据来源。"""

    LOCAL = "local"
    WEB = "web"
    RERANK = "rerank"


class Evidence(BaseModel):
    """统一后的证据结构。

    无论数据来自 Milvus、MCP 网络搜索还是 Reranker，
    上层都只读取这个结构。
    """

    evidence_id: str
    source: EvidenceSource
    chunk_id: str | int | None = None
    file_title: str | None = None
    title: str | None = None
    item_name: str | None = None
    content: str = ""
    url: str | None = None
    score: float | None = None
    score_type: str | None = None
    rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
