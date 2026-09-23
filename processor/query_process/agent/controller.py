"""Agent Controller。

Controller 决定下一步调用哪个工具，
不直接执行检索，也不生成最终答案。
"""

import json
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from processor.query_process.agent.decisions import (
    AgentAction,
    AgentDecision,
    EvidenceAssessmentStatus,
)
from processor.query_process.agent.search_policy import requires_web_search
from processor.query_process.agent.state import AgentState
from processor.query_process.tools.contracts import SearchStrategy


class ControllerDecisionProvider(Protocol):
    """Controller 决策提供者接口，便于测试时注入 Fake 实现。"""

    def decide(self, state: AgentState) -> AgentDecision:
        """根据状态返回下一步动作。"""
        ...


class LLMControllerDecisionProvider:
    """使用 LLM 进行动态决策。"""

    def decide(self, state: AgentState) -> AgentDecision:
        """让 LLM 从允许的动作中选择下一步。"""

        evidence_text = "\n".join(
            f"- {item.title}: {item.content[:300]}"
            for item in state.reranked_evidence[:5]
        )

        prompt = f"""
【用户问题】
{state.original_query}

【改写问题】
{state.rewritten_query}

【已确认商品】
{state.item_names}

【候选商品】
{state.candidates}

【已执行动作】
{state.used_actions}

【证据评估】
{state.assessment.model_dump_json() if state.assessment else "暂无"}

【当前证据摘要】
{evidence_text if evidence_text else "暂无"}

可选动作：
- resolve_product
- search_local，strategy 为 basic 或 hyde
- search_web
- ask_user
- cannot_answer
- answer

只输出 JSON：
{{
  "action": "动作名称",
  "reason": "简短决策理由，不要输出思维链",
  "query": "检索问题时填写，否则为空",
  "item_names": ["商品名"],
  "strategy": "basic 或 hyde，非本地检索时为空",
  "clarification": "ask_user 时的澄清问题"
}}
""".strip()

        # 延迟导入，避免仅导入本模块就初始化模型客户端。
        from utils.client.ai_clients import AIClients

        llm_client = AIClients.get_llm_client(response_format=True)
        response = llm_client.invoke([
            SystemMessage(
                content=(
                    "你是 RAG Agent 的调度器。"
                    "只做下一步动作决策，不回答用户问题，不输出额外解释。"
                )
            ),
            HumanMessage(content=prompt),
        ])

        raw = str(response.content).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()

        return AgentDecision.model_validate(json.loads(raw))


class AgentController:
    """单 Agent 的决策中心。"""

    def __init__(
        self,
        decision_provider: ControllerDecisionProvider | None = None,
    ) -> None:
        self.decision_provider = decision_provider

    def decide(self, state: AgentState) -> AgentDecision:
        """计算下一步动作。"""

        # 超过最大步数时停止，防止死循环。
        if state.current_step >= state.max_steps:
            return AgentDecision(
                action=AgentAction.CANNOT_ANSWER,
                reason="已达到最大工具调用步数",
            )

        # 多个候选商品时，必须让用户确认。
        if state.candidates and not state.item_names:
            return AgentDecision(
                action=AgentAction.ASK_USER,
                reason="存在多个候选商品，需要用户确认",
                clarification=state.clarification,
            )

        # 尚未确认商品时，先执行商品识别。
        if (
            not state.product_resolved
            and not state.action_used(AgentAction.RESOLVE_PRODUCT.value)
        ):
            return AgentDecision(
                action=AgentAction.RESOLVE_PRODUCT,
                reason="尚未确认商品，先执行商品识别",
            )

        # 咨询问题至少执行一次本地基础检索。
        if not state.action_used(
            f"{AgentAction.SEARCH_LOCAL.value}:basic"
        ):
            return AgentDecision(
                action=AgentAction.SEARCH_LOCAL,
                reason="先执行默认本地基础检索",
                query=state.rewritten_query or state.original_query,
                item_names=state.item_names,
                strategy=SearchStrategy.BASIC,
            )

        # 价格、库存、最新动态等问题必须补充网络搜索。
        # 即使通用证据评估认为本地证据充足，也不能跳过该步骤。
        if (
            requires_web_search(
                state.original_query,
                state.rewritten_query,
            )
            and not state.action_used(AgentAction.SEARCH_WEB.value)
        ):
            return AgentDecision(
                action=AgentAction.SEARCH_WEB,
                reason="问题涉及时效性或外部信息，需要网络搜索",
                query=state.rewritten_query or state.original_query,
            )

        # 已有证据评估时，优先按评估建议执行。
        if state.assessment is not None:
            decision = self._decision_from_assessment(state)
            if decision is not None:
                return decision

        # 规则无法决定时，允许 LLM 动态调度。
        if self.decision_provider is not None:
            try:
                decision = self.decision_provider.decide(state)
                if self._is_valid(state, decision):
                    return decision
            except Exception:
                pass

        # 最后兜底，不再盲目调用工具。
        return AgentDecision(
            action=AgentAction.CANNOT_ANSWER,
            reason="当前状态无法继续有效检索",
        )

    def _decision_from_assessment(
        self,
        state: AgentState,
    ) -> AgentDecision | None:
        """把证据评估结论转换成可执行动作。"""

        assessment = state.assessment
        if assessment is None:
            return None

        status = assessment.status

        if status == EvidenceAssessmentStatus.SUFFICIENT:
            return AgentDecision(
                action=AgentAction.ANSWER,
                reason=assessment.reason or "证据已经足够",
            )

        if status == EvidenceAssessmentStatus.NEED_CLARIFICATION:
            return AgentDecision(
                action=AgentAction.ASK_USER,
                reason=assessment.reason,
                clarification=state.clarification,
            )

        if status == EvidenceAssessmentStatus.CANNOT_ANSWER:
            return AgentDecision(
                action=AgentAction.CANNOT_ANSWER,
                reason=assessment.reason,
            )

        if status in (
            EvidenceAssessmentStatus.NEED_HYDE,
            EvidenceAssessmentStatus.NEED_LOCAL_RETRY,
        ):
            # 本地重试优先使用 HyDE，避免重复基础检索。
            if not state.action_used(
                f"{AgentAction.SEARCH_LOCAL.value}:hyde"
            ):
                return AgentDecision(
                    action=AgentAction.SEARCH_LOCAL,
                    reason=assessment.reason,
                    query=state.rewritten_query or state.original_query,
                    item_names=state.item_names,
                    strategy=SearchStrategy.HYDE,
                )

            if not state.action_used(AgentAction.SEARCH_WEB.value):
                return AgentDecision(
                    action=AgentAction.SEARCH_WEB,
                    reason="HyDE 后仍无有效证据，尝试网络搜索",
                    query=state.rewritten_query or state.original_query,
                )

        if status == EvidenceAssessmentStatus.NEED_WEB:
            if not state.action_used(AgentAction.SEARCH_WEB.value):
                return AgentDecision(
                    action=AgentAction.SEARCH_WEB,
                    reason=assessment.reason,
                    query=state.rewritten_query or state.original_query,
                )

            return AgentDecision(
                action=AgentAction.CANNOT_ANSWER,
                reason="网络搜索也未找到足够证据",
            )

        return None

    def _is_valid(
        self,
        state: AgentState,
        decision: AgentDecision,
    ) -> bool:
        """校验 LLM 返回的决策是否允许执行。"""

        # 不允许无证据直接回答。
        if (
            decision.action == AgentAction.ANSWER
            and not state.reranked_evidence
        ):
            return False

        # 不允许重复执行本地基础检索或 HyDE。
        if decision.action == AgentAction.SEARCH_LOCAL:
            strategy = decision.strategy or SearchStrategy.BASIC
            return not state.action_used(
                f"{AgentAction.SEARCH_LOCAL.value}:{strategy.value}"
            )

        # 不允许重复执行网络搜索。
        if decision.action == AgentAction.SEARCH_WEB:
            return not state.action_used(AgentAction.SEARCH_WEB.value)

        return True
