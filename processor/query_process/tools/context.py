"""工具调用上下文。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    """描述一次工具调用所处的会话和任务环境。

    工具只读取该上下文，不直接修改全局任务状态、SSE 队列或历史记录。
    """

    task_id: str = ""
    session_id: str = ""
    original_query: str = ""
    rewritten_query: str = ""
    item_names: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
