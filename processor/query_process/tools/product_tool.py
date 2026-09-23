"""商品识别工具适配器。"""

import re
from time import perf_counter

from processor.query_process.nodes.item_name_confirmed_node import (
    ItemNameConfirmedNode,
)
from processor.query_process.state import get_default_state
from processor.query_process.tools.context import ToolContext
from processor.query_process.tools.contracts import (
    ProductResolution,
    ProductResolutionStatus,
    ToolResult,
    ToolStatus,
)


class ResolveProductTool:
    """包装现有商品识别节点，并输出结构化结果。"""

    name = "resolve_product"

    def run(self, context: ToolContext) -> ToolResult[ProductResolution]:
        """执行商品识别。"""
        started = perf_counter()

        try:
            # 为一次工具调用创建独立 state，避免污染旧图状态。
            state = get_default_state()
            state["session_id"] = context.session_id
            state["task_id"] = context.task_id
            state["original_query"] = context.original_query
            state["is_stream"] = False

            # 调用 process 而不是 __call__，从而跳过任务进度和 SSE 副作用。
            state = ItemNameConfirmedNode().process(state)

            item_names = list(state.get("item_names") or [])
            clarification = str(state.get("answer") or "").strip()
            candidates = self._extract_candidates(clarification)

            if item_names:
                resolution = ProductResolutionStatus.CONFIRMED
                status = ToolStatus.SUCCESS
            elif candidates:
                resolution = ProductResolutionStatus.AMBIGUOUS
                status = ToolStatus.AMBIGUOUS
            else:
                resolution = ProductResolutionStatus.UNRESOLVED
                status = ToolStatus.UNRESOLVED

            output = ProductResolution(
                item_names=item_names,
                candidates=candidates,
                rewritten_query=str(state.get("rewritten_query") or ""),
                clarification=clarification or None,
                resolution=resolution,
            )

            return ToolResult(
                tool_name=self.name,
                status=status,
                output=output,
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
        except Exception as exc:
            # 工具层把异常转成结构化错误，供后续 Controller 决策。
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(exc),
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )

    @staticmethod
    def _extract_candidates(text: str) -> list[str]:
        """从现有澄清语句中提取候选商品。"""
        marker = "请问你是在询问以下内容吗"
        if marker not in text:
            return []

        match = re.search(r"\[([^\]]+)\]", text)
        if not match:
            return []

        return [
            item.strip()
            for item in match.group(1).split(",")
            if item.strip()
        ]
