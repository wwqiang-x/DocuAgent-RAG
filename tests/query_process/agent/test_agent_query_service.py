"""验证 AgentQueryService 的同步和流式行为。"""

import unittest
from unittest.mock import patch

from services.agent_query_service import AgentQueryService


class AgentQueryServiceTest(unittest.TestCase):
    """Agent 查询服务测试。"""

    @patch("services.agent_query_service.save_chat_message")
    @patch("services.agent_query_service.AgentRuntime")
    def test_sync_query_returns_answer(
        self,
        mock_runtime_class,
        mock_save_history,
    ) -> None:
        """同步查询应返回 Agent 答案并保存两条历史。"""
        runtime = mock_runtime_class.return_value
        runtime.run.return_value = _build_state(answer="测试答案")

        service = AgentQueryService()
        answer = service.run_agent_query(
            task_id="task-001",
            query="测试问题",
            session_id="session-001",
            is_stream=False,
        )

        self.assertEqual(answer, "测试答案")
        self.assertEqual(service.get_task_result("task-001"), "测试答案")
        self.assertEqual(mock_save_history.call_count, 2)

    @patch("services.agent_query_service.push_sse_event")
    @patch("services.agent_query_service.save_chat_message")
    @patch("services.agent_query_service.AgentRuntime")
    def test_stream_query_uses_clarification(
        self,
        mock_runtime_class,
        mock_save_history,
        mock_push_event,
    ) -> None:
        """流式查询没有 answer 时应发送 clarification。"""
        runtime = mock_runtime_class.return_value
        runtime.run.return_value = _build_state(
            answer="",
            clarification="请补充具体型号",
        )

        service = AgentQueryService()
        answer = service.run_agent_query(
            task_id="task-002",
            query="这个怎么使用",
            session_id="session-002",
            is_stream=True,
        )

        self.assertEqual(answer, "请补充具体型号")
        self.assertTrue(mock_push_event.called)
        self.assertEqual(mock_save_history.call_count, 2)

    @patch("services.agent_query_service.push_sse_event")
    def test_delta_callback_pushes_delta_event(self, mock_push_event) -> None:
        """答案增量应通过 delta 事件推送。"""
        callback = AgentQueryService()._build_delta_callback("task-003")

        callback("流式片段")

        self.assertEqual(mock_push_event.call_count, 1)
        self.assertEqual(
            mock_push_event.call_args.kwargs["event"],
            "delta",
        )


def _build_state(answer: str, clarification: str | None = None):
    """构造测试使用的 AgentState。"""
    from processor.query_process.agent.state import AgentState

    return AgentState(
        original_query="测试问题",
        rewritten_query="改写后的测试问题",
        item_names=["测试商品"],
        answer=answer,
        clarification=clarification,
    )


if __name__ == "__main__":
    unittest.main()
