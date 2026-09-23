"""验证 Agent 答案生成支持 delta 流式回调。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from processor.query_process.agent.answer_service import AgentAnswerService
from processor.query_process.agent.state import AgentState
from processor.query_process.evidence.models import Evidence, EvidenceSource


class _FakeStreamingClient:
    """模拟 ChatOpenAI 的流式接口。"""

    def stream(self, prompt):
        """逐段返回文本。"""
        yield SimpleNamespace(content="流式")
        yield SimpleNamespace(content="答案")

    def invoke(self, prompt):
        """同步接口不应在流式测试中调用。"""
        raise AssertionError("流式模式不应调用 invoke")


class AgentAnswerStreamingTest(unittest.TestCase):
    """Agent 流式答案测试。"""

    @patch(
        "processor.query_process.nodes.answer_output_node."
        "AnswerOutputNode._build_prompt",
        return_value="测试 prompt",
    )
    @patch(
        "utils.client.ai_clients.AIClients.get_llm_client",
        return_value=_FakeStreamingClient(),
    )
    def test_generate_streams_delta(
        self,
        mock_get_llm_client,
        mock_build_prompt,
    ) -> None:
        """流式回调应收到每个文本片段和完整答案。"""
        state = AgentState(
            original_query="测试问题",
            reranked_evidence=[
                Evidence(
                    evidence_id="local:1",
                    source=EvidenceSource.LOCAL,
                    chunk_id=1,
                    content="测试证据",
                )
            ],
        )
        received = []

        answer = AgentAnswerService().generate(
            state,
            delta_callback=received.append,
        )

        self.assertEqual(received, ["流式", "答案"])
        self.assertEqual(answer, "流式答案")


if __name__ == "__main__":
    unittest.main()
