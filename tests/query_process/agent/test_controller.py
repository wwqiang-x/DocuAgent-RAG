"""验证阶段 B 的 Controller 决策。"""

import unittest

from processor.query_process.agent.controller import AgentController
from processor.query_process.agent.decisions import (
    AgentAction,
    EvidenceAssessment,
    EvidenceAssessmentStatus,
)
from processor.query_process.agent.state import AgentState
from processor.query_process.tools.contracts import SearchStrategy


class AgentControllerTest(unittest.TestCase):
    """Controller 基础决策测试。"""

    def test_resolve_product_first(self) -> None:
        """未确认商品时应先调用商品识别。"""
        state = AgentState(original_query="RS-12怎么测电阻")

        decision = AgentController().decide(state)

        self.assertEqual(decision.action, AgentAction.RESOLVE_PRODUCT)

    def test_basic_search_after_product_resolution(self) -> None:
        """确认商品后应先执行基础本地检索。"""
        state = AgentState(
            original_query="RS-12怎么测电阻",
            item_names=["RS PRO RS-12 数字万用表"],
            product_resolved=True,
        )

        decision = AgentController().decide(state)

        self.assertEqual(decision.action, AgentAction.SEARCH_LOCAL)
        self.assertEqual(decision.strategy, SearchStrategy.BASIC)

    def test_sufficient_evidence_answers(self) -> None:
        """证据足够时应生成答案。"""
        state = AgentState(
            original_query="RS-12怎么测电阻",
            item_names=["RS PRO RS-12 数字万用表"],
            product_resolved=True,
            used_actions=["search_local:basic"],
            assessment=EvidenceAssessment(
                status=EvidenceAssessmentStatus.SUFFICIENT,
                reason="证据足够",
                recommended_action=AgentAction.ANSWER,
            ),
        )

        decision = AgentController().decide(state)

        self.assertEqual(decision.action, AgentAction.ANSWER)

    def test_market_price_forces_web_search(self) -> None:
        """价格问题即使本地证据充足也必须执行网搜。"""
        state = AgentState(
            original_query="它的市场价是多少",
            item_names=["RS PRO RS-12 数字万用表"],
            product_resolved=True,
            used_actions=["search_local:basic"],
            assessment=EvidenceAssessment(
                status=EvidenceAssessmentStatus.SUFFICIENT,
                reason="本地说明书证据充足",
                recommended_action=AgentAction.ANSWER,
            ),
        )

        decision = AgentController().decide(state)

        self.assertEqual(decision.action, AgentAction.SEARCH_WEB)


if __name__ == "__main__":
    unittest.main()
