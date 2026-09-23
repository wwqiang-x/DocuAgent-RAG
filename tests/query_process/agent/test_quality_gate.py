"""验证阶段 B 的证据质量评估。"""

import unittest

from processor.query_process.agent.decisions import (
    EvidenceAssessmentStatus,
)
from processor.query_process.agent.quality_gate import (
    EvidenceQualityGate,
    RuleBasedEvidenceAssessor,
)
from processor.query_process.agent.state import AgentState


class QualityGateTest(unittest.TestCase):
    """证据质量评估测试。"""

    def setUp(self) -> None:
        self.gate = EvidenceQualityGate(
            fallback_assessor=RuleBasedEvidenceAssessor()
        )

    def test_ambiguous_product_needs_clarification(self) -> None:
        """多个候选商品时应要求用户澄清。"""
        state = AgentState(
            original_query="这个怎么使用",
            candidates=["设备A", "设备B"],
        )

        assessment = self.gate.assess(state)

        self.assertEqual(
            assessment.status,
            EvidenceAssessmentStatus.NEED_CLARIFICATION,
        )

    def test_empty_basic_search_needs_hyde(self) -> None:
        """基础检索为空时应建议 HyDE。"""
        from processor.query_process.tools.contracts import SearchOutput

        state = AgentState(
            original_query="设备怎么使用",
            basic_search=SearchOutput(),
        )

        assessment = self.gate.assess(state)

        self.assertEqual(
            assessment.status,
            EvidenceAssessmentStatus.NEED_HYDE,
        )

    def test_market_price_needs_web(self) -> None:
        """价格问题应优先建议网络搜索。"""
        state = AgentState(
            original_query="它的市场价是多少",
            reranked_evidence=[],
        )

        assessment = self.gate.assess(state)

        self.assertEqual(
            assessment.status,
            EvidenceAssessmentStatus.NEED_WEB,
        )


if __name__ == "__main__":
    unittest.main()
