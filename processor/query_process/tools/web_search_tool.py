"""网络搜索工具适配器。"""

from time import perf_counter

from pydantic import BaseModel

from processor.query_process.evidence.mappers import from_web_doc
from processor.query_process.nodes.web_mcp_search_node import WebMcpSearchNode
from processor.query_process.state import get_default_state
from processor.query_process.tools.context import ToolContext
from processor.query_process.tools.contracts import (
    SearchOutput,
    ToolResult,
    ToolStatus,
)


class WebSearchRequest(BaseModel):
    """网络搜索参数。"""

    query: str = ""


class WebSearchTool:
    """包装现有 MCP 网络搜索节点。"""

    name = "search_web"

    def run(
        self,
        context: ToolContext,
        request: WebSearchRequest,
    ) -> ToolResult[SearchOutput]:
        """执行网络搜索，并转换为统一证据。"""
        started = perf_counter()

        query = (
            request.query.strip()
            or context.rewritten_query.strip()
            or context.original_query.strip()
        )
        if not query:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error="网络搜索查询不能为空",
            )

        state = get_default_state()
        state["task_id"] = context.task_id
        state["rewritten_query"] = query
        state["item_names"] = context.item_names
        state["is_stream"] = False

        try:
            # 复用现有 MCP 搜索节点。
            state = WebMcpSearchNode().process(state)
            docs = state.get("web_search_docs") or []

            # 统一为 Evidence，后续 Reranker 不需要判断搜索结果来自哪个接口。
            evidences = [
                from_web_doc(doc, index + 1)
                for index, doc in enumerate(docs)
            ]

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS if evidences else ToolStatus.EMPTY,
                output=SearchOutput(evidences=evidences),
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(exc),
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
