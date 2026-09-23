"""验证阶段 B Runtime 的离线串联逻辑。"""

import unittest

from processor.query_process.agent.controller import AgentController
from processor.query_process.agent.decisions import (
    AgentAction,
    AgentDecision,
    EvidenceAssessment,
    EvidenceAssessmentStatus,
)
from processor.query_process.agent.quality_gate import EvidenceQualityGate
from processor.query_process.agent.runtime import AgentRuntime
from processor.query_process.evidence.models import Evidence, EvidenceSource
from processor.query_process.tools.contracts import (
    ProductResolution,
    ProductResolutionStatus,
    SearchOutput,
    ToolResult,
    ToolStatus,
)


class _FakeRegistry:
    """提供 Runtime 测试所需的假工具。"""

    def __init__(self) -> None:
        self.tools = {
            "resolve_product": _FakeProductTool(),
            "search_local": _FakeLocalSearchTool(),
            "search_web": _FakeWebSearchTool(),
        }

    def get(self, name: str):
        return self.tools[name]


class _FakeProductTool:
    """假商品识别工具。"""

    def run(self, context):
        return ToolResult(
            tool_name="resolve_product",
            status=ToolStatus.SUCCESS,
            output=ProductResolution(
                item_names=["RS PRO RS-12 数字万用表"],
                rewritten_query="RS-12数字万用表如何测量电阻",
                resolution=ProductResolutionStatus.CONFIRMED,
            ),
        )


class _FakeLocalSearchTool:
    """假本地检索工具。"""

    def run(self, context, request):
        evidence = Evidence(
            evidence_id="local:1",
            source=EvidenceSource.LOCAL,
            chunk_id=1,
            title="电阻测量",
            content="测量电阻的步骤。",
        )
        return ToolResult(
            tool_name="search_local",
            status=ToolStatus.SUCCESS,
            output=SearchOutput(
                strategy=request.strategy,
                evidences=[evidence],
            ),
        )


class _FakeWebSearchTool:
    """假网络搜索工具。"""

    def run(self, context, request):
        return ToolResult(
            tool_name="search_web",
            status=ToolStatus.EMPTY,
            output=SearchOutput(),
        )


class _FakeEvidencePipeline:
    """假证据管线，直接返回输入证据。"""

    def fuse_local(self, basic, hyde):
        return list(basic.evidences if basic else [])

    def rerank(self, query, local_evidence, web_evidence=None):
        return list(local_evidence) + list(web_evidence or [])


class _FakeQualityGate:
    """第一次评估返回足够，避免测试触发网络搜索。"""

    def assess(self, state):
        return EvidenceAssessment(
            status=EvidenceAssessmentStatus.SUFFICIENT,
            reason="测试证据充足",
            recommended_action=AgentAction.ANSWER,
        )


class _FakeAnswerService:
    """假答案服务，避免调用真实 LLM。"""

    def generate(self, state, delta_callback=None):
        if delta_callback is not None:
            delta_callback("测试答案")
        return "测试答案"


class AgentRuntimeTest(unittest.TestCase):
    """Runtime 离线执行测试。"""

    def test_confirmed_product_can_answer(self) -> None:
        """商品确认且证据足够时应生成答案。"""
        runtime = AgentRuntime(
            controller=AgentController(),
            quality_gate=_FakeQualityGate(),
            answer_service=_FakeAnswerService(),
            tool_registry=_FakeRegistry(),
            evidence_pipeline=_FakeEvidencePipeline(),
            enable_selection_preflight=False,
        )

        state = runtime.run("RS-12数字万用表如何测量电阻")

        self.assertEqual(state.answer, "测试答案")
        self.assertEqual(state.item_names, ["RS PRO RS-12 数字万用表"])
        self.assertIn("search_local:basic", state.used_actions)


if __name__ == "__main__":
    unittest.main()
