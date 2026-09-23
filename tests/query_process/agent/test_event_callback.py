"""验证 Agent Runtime 的步骤事件回调。"""

import unittest

from processor.query_process.agent.decisions import AgentAction
from processor.query_process.agent.runtime import AgentRuntime
from processor.query_process.agent.state import AgentState


class AgentEventCallbackTest(unittest.TestCase):
    """AgentStep 回调测试。"""

    def test_append_step_calls_callback(self) -> None:
        """记录步骤后应把同一个 AgentStep 传给回调。"""
        received_steps = []
        runtime = AgentRuntime(
            controller=object(),
            quality_gate=object(),
            answer_service=object(),
            tool_registry=object(),
            evidence_pipeline=object(),
            enable_selection_preflight=False,
            event_callback=received_steps.append,
        )
        state = AgentState(original_query="测试问题")

        runtime._append_step(
            state=state,
            action=AgentAction.RESOLVE_PRODUCT,
            reason="测试步骤",
            tool_name="resolve_product",
            tool_status="success",
            duration_ms=12.5,
        )

        self.assertEqual(len(received_steps), 1)
        self.assertEqual(received_steps[0].step_no, 1)
        self.assertEqual(
            received_steps[0].action,
            AgentAction.RESOLVE_PRODUCT,
        )

    def test_callback_error_does_not_break_runtime(self) -> None:
        """回调异常不能影响 Agent 主流程。"""

        def broken_callback(_step) -> None:
            raise RuntimeError("callback failed")

        runtime = AgentRuntime(
            controller=object(),
            quality_gate=object(),
            answer_service=object(),
            tool_registry=object(),
            evidence_pipeline=object(),
            enable_selection_preflight=False,
            event_callback=broken_callback,
        )
        state = AgentState(original_query="测试问题")

        runtime._append_step(
            state=state,
            action=AgentAction.SEARCH_WEB,
            reason="测试回调异常",
        )

        self.assertEqual(state.current_step, 1)
        self.assertEqual(len(state.steps), 1)


if __name__ == "__main__":
    unittest.main()
