"""Agent 查询服务。

该服务把阶段 B 的 AgentRuntime 接入 FastAPI。
现有 QueryService 保持不变，两种查询模式互不影响。
"""

import logging
import traceback
import uuid

from processor.import_process.base import setup_logging
from processor.query_process.agent.decisions import AgentStep
from processor.query_process.agent.runtime import AgentRuntime
from utils.mongo_history_util import save_chat_message
from utils.sse_util import SSEEvent, push_sse_event
from utils.task_util import get_task_result, set_task_result


class AgentQueryService:
    """Agent 模式的查询服务。"""

    @staticmethod
    def gener_task_id() -> str:
        """生成与旧查询服务规格一致的任务 ID。"""
        return str(uuid.uuid4().hex[:12])

    def run_agent_query(
        self,
        task_id: str,
        query: str,
        session_id: str,
        is_stream: bool,
        max_steps: int = 6,
        raise_on_error: bool = False,
    ) -> str:
        """执行 Agent 查询。

        该方法既可以用于同步查询，也可以交给 BackgroundTasks 执行。

        raise_on_error=True：
            同步模式使用，异常交给 API 决定是否回退 legacy。
        raise_on_error=False：
            流式模式使用，异常转换成 final 错误答案。
        """

        try:
            setup_logging(logging.INFO)

            # 只有流式模式才注册步骤回调，向 SSE 队列推送 agent_step。
            event_callback = (
                self._build_step_callback(task_id)
                if is_stream
                else None
            )
            delta_callback = (
                self._build_delta_callback(task_id)
                if is_stream
                else None
            )

            # 每次请求创建新的 Runtime，避免不同请求共享运行状态。
            runtime = AgentRuntime(
                event_callback=event_callback,
                delta_callback=delta_callback,
            )

            # 执行 Agent 决策循环。
            state = runtime.run(
                query=query,
                session_id=session_id,
                task_id=task_id,
                max_steps=max_steps,
            )

            # ask_user 场景通常只有 clarification，没有 answer。
            # 对外统一转换成可以直接展示的答案。
            answer = (
                state.answer.strip()
                or (state.clarification or "").strip()
                or "根据现有资料无法回答"
            )

            # 保存任务结果，供非流式接口通过 task_id 读取。
            set_task_result(
                task_id=task_id,
                key="answer",
                value=answer,
            )

            # AgentRuntime 保持无副作用，由 Service 统一保存历史。
            self._save_history(
                session_id=session_id,
                query=query,
                answer=answer,
                state=state,
            )

            # 流式模式最后发送完整答案。
            if is_stream:
                push_sse_event(
                    task_id=task_id,
                    event=SSEEvent.FINAL,
                    data={"answer": answer},
                )

            return answer

        except Exception as exc:
            traceback.print_exc()

            # 同步模式允许 API 捕获异常并回退普通检索。
            if raise_on_error:
                raise

            # Agent 异常时返回可展示信息，避免流式连接一直等待。
            answer = f"很抱歉，Agent 查询失败: {exc}"
            set_task_result(
                task_id=task_id,
                key="answer",
                value=answer,
            )

            if is_stream:
                push_sse_event(
                    task_id=task_id,
                    event=SSEEvent.FINAL,
                    data={"answer": answer},
                )

            return answer

    def _build_step_callback(self, task_id: str):
        """构造 AgentStep 的 SSE 回调。"""

        def callback(step: AgentStep) -> None:
            # mode=json 把枚举转换成字符串，确保 SSE JSON 可以序列化。
            push_sse_event(
                task_id=task_id,
                event="agent_step",
                data=step.model_dump(mode="json"),
            )

        return callback

    def _build_delta_callback(self, task_id: str):
        """构造 LLM 答案增量的 SSE 回调。"""

        def callback(delta: str) -> None:
            # 每个 delta 都是 LLM 返回的一小段文本。
            push_sse_event(
                task_id=task_id,
                event=SSEEvent.DELTA,
                data={"answer": delta},
            )

        return callback

    def _save_history(
        self,
        session_id: str,
        query: str,
        answer: str,
        state,
    ) -> None:
        """保存用户问题和 Agent 最终回答。"""

        try:
            # 保存用户消息。
            save_chat_message(
                session_id=session_id,
                role="user",
                text=query,
                rewritten_query=state.rewritten_query,
                item_names=state.item_names,
            )

            # 保存 Agent 最终回答或澄清话术。
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query=state.rewritten_query,
                item_names=state.item_names,
            )
        except Exception as exc:
            # 历史写入失败不应影响已经生成的答案返回。
            logging.error(f"Agent 历史记录保存失败: {exc}")

    def get_task_result(self, task_id: str) -> str:
        """读取同步 Agent 查询结果。"""
        return get_task_result(task_id=task_id, key="answer")
