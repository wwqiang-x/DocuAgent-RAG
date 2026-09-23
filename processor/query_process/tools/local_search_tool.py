"""本地混合检索工具适配器。"""

from time import perf_counter

from pydantic import BaseModel, Field

from processor.query_process.evidence.mappers import from_milvus_hit
from processor.query_process.nodes.hybrid_vector_search_node import (
    HybridVectorSearchNode,
)
from processor.query_process.nodes.hyde_vector_search_node import (
    HyDEVectorSearchNode,
)
from processor.query_process.state import get_default_state
from processor.query_process.tools.context import ToolContext
from processor.query_process.tools.contracts import (
    SearchOutput,
    SearchStrategy,
    ToolResult,
    ToolStatus,
)


class LocalSearchRequest(BaseModel):
    """本地检索参数。"""

    query: str = ""
    item_names: list[str] = Field(default_factory=list)
    strategy: SearchStrategy = SearchStrategy.BASIC


class LocalSearchTool:
    """统一封装基础检索和 HyDE 检索。"""

    name = "search_local"

    def run(
        self,
        context: ToolContext,
        request: LocalSearchRequest,
    ) -> ToolResult[SearchOutput]:
        """按指定策略执行一次本地检索。"""
        started = perf_counter()

        # 查询优先级：本次请求 > 上下文改写问题 > 上下文原始问题。
        query = (
            request.query.strip()
            or context.rewritten_query.strip()
            or context.original_query.strip()
        )
        if not query:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error="本地检索查询不能为空",
            )

        # 请求中的商品过滤条件优先于上下文。
        item_names = request.item_names or context.item_names

        state = get_default_state()
        state["task_id"] = context.task_id
        state["session_id"] = context.session_id
        state["original_query"] = context.original_query
        state["rewritten_query"] = query
        state["item_names"] = item_names
        state["is_stream"] = False

        try:
            if request.strategy == SearchStrategy.HYDE:
                # 复用现有 HyDE 检索节点。
                state = HyDEVectorSearchNode().process(state)
                raw_hits = state.get("hyde_embedding_chunks") or []
            else:
                # 复用现有基础混合检索节点。
                state = HybridVectorSearchNode().process(state)
                raw_hits = state.get("embedding_chunks") or []

            # 统一转换为 Evidence，屏蔽 Milvus 原始对象结构。
            evidences = [
                from_milvus_hit(hit, request.strategy, index + 1)
                for index, hit in enumerate(raw_hits)
            ]

            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS if evidences else ToolStatus.EMPTY,
                output=SearchOutput(
                    strategy=request.strategy,
                    evidences=evidences,
                ),
                duration_ms=round((perf_counter() - started) * 1000, 2),
                metadata={"strategy": request.strategy.value},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                error=str(exc),
                duration_ms=round((perf_counter() - started) * 1000, 2),
                metadata={"strategy": request.strategy.value},
            )
