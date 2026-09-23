"""Agent 答案生成服务。

该服务只负责生成答案文本。
是否推送 SSE 由调用方传入的 delta_callback 决定。
"""

from collections.abc import Callable

from processor.query_process.agent.state import AgentState
from processor.query_process.nodes.answer_output_node import AnswerOutputNode
from processor.query_process.state import get_default_state


class AgentAnswerService:
    """基于已经重排序的证据生成答案。"""

    def generate(
        self,
        state: AgentState,
        delta_callback: Callable[[str], None] | None = None,
    ) -> str:
        """生成最终答案。

        delta_callback 不为空时使用 LLM.stream 逐段输出；
        否则使用 LLM.invoke 一次性返回。
        """

        # 没有证据时禁止调用 LLM 编造答案。
        if not state.reranked_evidence:
            return "根据现有资料无法回答"

        # 复用现有 AnswerOutputNode 的上下文和 Prompt 组装逻辑。
        legacy_state = get_default_state()
        legacy_state["rewritten_query"] = (
            state.rewritten_query or state.original_query
        )
        legacy_state["item_names"] = state.item_names
        legacy_state["history"] = state.history
        legacy_state["reranked_docs"] = [
            {
                "chunk_id": item.chunk_id,
                "content": item.content,
                "title": item.title,
                "source": item.source.value,
                "url": item.url,
                "score": item.score,
            }
            for item in state.reranked_evidence
        ]

        answer_node = AnswerOutputNode()
        prompt = answer_node._build_prompt(legacy_state)

        # 延迟导入，避免仅导入本模块就初始化模型客户端。
        from utils.client.ai_clients import AIClients

        llm_client = AIClients.get_llm_client(response_format=False)

        # 流式模式逐段推送 delta，最终仍返回完整文本供 final 使用。
        if delta_callback is not None:
            answer_parts = []
            for section in llm_client.stream(prompt):
                delta = str(getattr(section, "content", "") or "")
                if not delta:
                    continue
                answer_parts.append(delta)
                delta_callback(delta)

            return "".join(answer_parts) or "很抱歉，获取答案失败"

        # 同步模式保持原有一次性调用。
        response = llm_client.invoke(prompt)
        return str(getattr(response, "content", "") or "很抱歉，获取答案失败")
