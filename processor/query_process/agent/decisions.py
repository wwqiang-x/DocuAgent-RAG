"""Agent 决策和质量评估相关数据结构。

这个模块只定义 Controller 可以执行哪些动作，
以及证据质量评估会返回哪些结论，不执行具体业务。
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from processor.query_process.tools.contracts import SearchStrategy


class AgentAction(str, Enum):
    """Agent Controller 可以选择的下一步动作。"""

    RESOLVE_PRODUCT = "resolve_product"
    SEARCH_LOCAL = "search_local"
    SEARCH_WEB = "search_web"
    ASK_USER = "ask_user"
    CANNOT_ANSWER = "cannot_answer"
    ANSWER = "answer"


class StopReason(str, Enum):
    """Agent 本轮运行最终为何停止。"""

    ANSWERED = "answered"
    ASKED_USER = "asked_user"
    CANNOT_ANSWER = "cannot_answer"
    MAX_STEPS = "max_steps"
    ERROR = "error"


class EvidenceAssessmentStatus(str, Enum):
    """证据质量评估结果。"""

    SUFFICIENT = "sufficient"
    NEED_LOCAL_RETRY = "need_local_retry"
    NEED_HYDE = "need_hyde"
    NEED_WEB = "need_web"
    NEED_CLARIFICATION = "need_clarification"
    CANNOT_ANSWER = "cannot_answer"


class EvidenceAssessment(BaseModel):
    """证据质量评估结果。"""

    # 禁止 LLM 返回未定义字段，避免决策结构被意外扩展。
    model_config = ConfigDict(extra="forbid")

    status: EvidenceAssessmentStatus
    reason: str = ""
    recommended_action: AgentAction | None = None
    missing_information: str = ""


class AgentDecision(BaseModel):
    """Controller 每轮输出的结构化决策。

    这里只记录最终动作和简短理由，不保存模型原始思维链。
    """

    model_config = ConfigDict(extra="forbid")

    action: AgentAction
    reason: str = ""
    query: str = ""
    item_names: list[str] = Field(default_factory=list)
    strategy: SearchStrategy | None = None
    clarification: str | None = None


class AgentStep(BaseModel):
    """记录 Agent 执行过的一步，方便调试和评估。"""

    step_no: int
    action: AgentAction
    reason: str = ""
    query: str = ""
    strategy: str | None = None
    tool_name: str | None = None
    tool_status: str | None = None
    evidence_count: int = 0
    duration_ms: float = 0.0
    error: str | None = None
