"""Agent 单轮运行状态。

AgentState 只存在于一次 Agent 运行中，不写 task_util，
也不会修改现有查询图状态。
"""

from typing import Any

from pydantic import BaseModel, Field

from processor.query_process.agent.decisions import (
    AgentStep,
    EvidenceAssessment,
    StopReason,
)
from processor.query_process.evidence.models import Evidence
from processor.query_process.tools.contracts import SearchOutput


class AgentState(BaseModel):
    """Agent 一次完整运行的上下文和中间结果。"""

    # ==================== 请求上下文 ====================
    task_id: str = ""
    session_id: str = ""
    original_query: str = ""
    rewritten_query: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)

    # ==================== 商品识别状态 ====================
    item_names: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    clarification: str | None = None
    selection_resolved: bool = False
    product_resolved: bool = False

    # ==================== 检索结果 ====================
    basic_search: SearchOutput | None = None
    hyde_search: SearchOutput | None = None
    web_search: SearchOutput | None = None

    # ==================== 证据处理结果 ====================
    fused_evidence: list[Evidence] = Field(default_factory=list)
    reranked_evidence: list[Evidence] = Field(default_factory=list)
    assessment: EvidenceAssessment | None = None

    # ==================== Agent 执行控制 ====================
    current_step: int = 0
    max_steps: int = 6
    used_actions: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    last_error: str | None = None

    # ==================== 最终输出 ====================
    answer: str = ""
    stop_reason: StopReason | None = None

    def action_used(self, action: str) -> bool:
        """判断某个动作是否已经执行过。"""
        return action in self.used_actions

    def mark_action_used(self, action: str) -> None:
        """记录已执行动作，防止重复调用工具。"""
        if action not in self.used_actions:
            self.used_actions.append(action)
