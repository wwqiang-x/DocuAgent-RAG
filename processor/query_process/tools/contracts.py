"""工具统一数据契约。

这些结构描述工具执行状态、输入策略和输出格式。
工具层可以依赖 evidence 层，但 evidence 层不反向依赖具体工具实现。
"""

from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from processor.query_process.evidence.models import Evidence


T = TypeVar("T")


class ToolStatus(str, Enum):
    """工具执行状态。"""

    SUCCESS = "success"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    ERROR = "error"


class SearchStrategy(str, Enum):
    """本地检索策略。"""

    BASIC = "basic"
    HYDE = "hyde"


class ProductResolutionStatus(str, Enum):
    """商品识别结果类型。"""

    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class ProductResolution(BaseModel):
    """商品识别工具的结构化输出。"""

    item_names: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    rewritten_query: str = ""
    clarification: str | None = None
    resolution: ProductResolutionStatus


class SearchOutput(BaseModel):
    """本地检索或网络搜索工具的结构化输出。"""

    strategy: SearchStrategy | None = None
    evidences: list[Evidence] = Field(default_factory=list)


class ToolResult(BaseModel, Generic[T]):
    """所有工具统一的返回结构。"""

    tool_name: str
    status: ToolStatus
    output: T | None = None
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
