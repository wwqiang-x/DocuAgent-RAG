"""证据质量评估。

EvidenceQualityGate 负责判断当前证据是否足够回答，
只提供下一步建议，不直接生成答案。
"""

import json
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from processor.query_process.agent.decisions import (
    AgentAction,
    EvidenceAssessment,
    EvidenceAssessmentStatus,
)
from processor.query_process.agent.search_policy import requires_web_search
from processor.query_process.agent.state import AgentState


class EvidenceAssessor(Protocol):
    """证据评估器接口，测试时可以替换为 Fake 实现。"""

    def assess(self, state: AgentState) -> EvidenceAssessment:
        """评估当前证据是否足够。"""
        ...


class RuleBasedEvidenceAssessor:
    """规则版证据评估器。

    规则评估用于兜底，保证 LLM 失败时流程仍然可以确定地停止。
    """

    def assess(self, state: AgentState) -> EvidenceAssessment:
        """根据当前状态给出确定性的质量结论。"""

        # 多个候选商品时不应继续盲目检索，应先让用户确认。
        if state.candidates and not state.item_names:
            return EvidenceAssessment(
                status=EvidenceAssessmentStatus.NEED_CLARIFICATION,
                reason="商品存在多个候选，需要用户确认具体型号",
                recommended_action=AgentAction.ASK_USER,
                missing_information="具体商品型号",
            )

        evidence = state.reranked_evidence

        # 价格、库存、最新动态等问题必须使用网络信息。
        if (
            requires_web_search(
                state.original_query,
                state.rewritten_query,
            )
            and state.web_search is None
        ):
            return EvidenceAssessment(
                status=EvidenceAssessmentStatus.NEED_WEB,
                reason="问题涉及时效性或外部信息，需要网络搜索",
                recommended_action=AgentAction.SEARCH_WEB,
            )

        # 没有证据时，根据已经执行过的检索路径决定下一步。
        if not evidence:
            if state.hyde_search is None:
                return EvidenceAssessment(
                    status=EvidenceAssessmentStatus.NEED_HYDE,
                    reason="基础检索未返回有效证据，建议使用 HyDE 补充",
                    recommended_action=AgentAction.SEARCH_LOCAL,
                )

            if state.web_search is None:
                return EvidenceAssessment(
                    status=EvidenceAssessmentStatus.NEED_WEB,
                    reason="本地检索和 HyDE 均无有效证据，建议网络搜索",
                    recommended_action=AgentAction.SEARCH_WEB,
                )

            return EvidenceAssessment(
                status=EvidenceAssessmentStatus.CANNOT_ANSWER,
                reason="所有可用检索路径均未找到证据",
                recommended_action=AgentAction.CANNOT_ANSWER,
            )

        # 有效证据较少时，保守地补充一次网络搜索。
        if len(evidence) < 2 and state.web_search is None:
            return EvidenceAssessment(
                status=EvidenceAssessmentStatus.NEED_WEB,
                reason="有效证据数量较少，建议使用网络搜索补充",
                recommended_action=AgentAction.SEARCH_WEB,
            )

        return EvidenceAssessment(
            status=EvidenceAssessmentStatus.SUFFICIENT,
            reason="当前证据可以用于生成答案",
            recommended_action=AgentAction.ANSWER,
        )


class LLMEvidenceAssessor:
    """使用 LLM 判断证据是否足够。"""

    def __init__(self) -> None:
        self.system_prompt = (
            "你是知识库检索质量评估专家。"
            "只能根据提供的证据判断，不能使用内部知识补全答案，"
            "只输出 JSON。"
        )

    def assess(self, state: AgentState) -> EvidenceAssessment:
        """调用 LLM 进行证据评估。"""

        # 截断证据内容，避免上下文过长。
        evidence_text = "\n\n".join(
            (
                f"[证据{index + 1}] "
                f"source={item.source.value}, "
                f"title={item.title}, score={item.score}\n"
                f"{item.content[:1000]}"
            )
            for index, item in enumerate(state.reranked_evidence[:5])
        )

        prompt = f"""
【用户问题】
{state.rewritten_query or state.original_query}

【已确认商品】
{state.item_names}

【当前证据】
{evidence_text if evidence_text else "无"}

请从以下状态中选择一个：
- sufficient：证据足够回答
- need_local_retry：需要改写查询后重新检索
- need_hyde：需要 HyDE 补充本地召回
- need_web：需要网络搜索补充
- need_clarification：需要用户补充商品或问题信息
- cannot_answer：现有手段无法回答

只输出 JSON：
{{
  "status": "上述状态之一",
  "reason": "一句话说明",
  "recommended_action": "resolve_product/search_local/search_web/ask_user/cannot_answer/answer",
  "missing_information": "缺少的信息，没有则为空字符串"
}}
""".strip()

        # 延迟导入，避免仅导入本模块就初始化 AI 客户端。
        from utils.client.ai_clients import AIClients

        llm_client = AIClients.get_llm_client(response_format=True)
        response = llm_client.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ])

        raw = str(response.content).strip()

        # 清理模型可能返回的 Markdown 代码围栏。
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()

        return EvidenceAssessment.model_validate(json.loads(raw))


class EvidenceQualityGate:
    """组合规则评估和 LLM 评估。"""

    def __init__(
        self,
        assessor: EvidenceAssessor | None = None,
        fallback_assessor: EvidenceAssessor | None = None,
    ) -> None:
        self.assessor = assessor
        self.fallback_assessor = fallback_assessor or RuleBasedEvidenceAssessor()

    def assess(self, state: AgentState) -> EvidenceAssessment:
        """评估证据，LLM 失败时自动回退到规则评估。"""

        # 商品候选不明确时，不需要调用 LLM。
        if state.candidates and not state.item_names:
            return self.fallback_assessor.assess(state)

        # 没有证据时，直接使用确定性规则。
        if not state.reranked_evidence:
            return self.fallback_assessor.assess(state)

        # 有证据时优先使用 LLM 判断。
        if self.assessor is not None:
            try:
                return self.assessor.assess(state)
            except Exception:
                pass

        return self.fallback_assessor.assess(state)
