"""RRF 和 Reranker 的内部证据处理管线。"""

from processor.query_process.evidence.mappers import (
    from_rerank_doc,
    from_rrf_chunk,
)
from processor.query_process.evidence.models import (
    Evidence,
    EvidenceSource,
)
from processor.query_process.nodes.reranker_node import ReRankerNode
from processor.query_process.nodes.rrf_merge_node import RRFMergeNode
from processor.query_process.state import get_default_state
from processor.query_process.tools.contracts import SearchOutput


class EvidencePipeline:
    """复用现有 RRF 和 Reranker 节点处理统一证据。"""

    def fuse_local(
        self,
        basic: SearchOutput | None,
        hyde: SearchOutput | None,
    ) -> list[Evidence]:
        """融合基础检索和 HyDE 检索结果。"""

        # 现有 RRF 节点仍依赖原始结构，因此在这里做一次临时转换。
        basic_items = [
            self._to_milvus_raw(item)
            for item in (basic.evidences if basic else [])
        ]
        hyde_items = [
            self._to_milvus_raw(item)
            for item in (hyde.evidences if hyde else [])
        ]

        state = get_default_state()
        state["embedding_chunks"] = basic_items
        state["hyde_embedding_chunks"] = hyde_items
        state["is_stream"] = False

        # 直接调用 process，避免写入任务状态或推送 SSE。
        state = RRFMergeNode().process(state)
        chunks = state.get("rrf_chunks") or []

        return [
            from_rrf_chunk(chunk, index + 1)
            for index, chunk in enumerate(chunks)
        ]

    def rerank(
        self,
        query: str,
        local_evidence: list[Evidence],
        web_evidence: list[Evidence] | None = None,
    ) -> list[Evidence]:
        """对本地和网络证据统一重排序。"""
        web_evidence = web_evidence or []

        # 没有证据时直接返回，避免调用空的 Reranker。
        if not local_evidence and not web_evidence:
            return []

        state = get_default_state()
        state["rewritten_query"] = query

        # Reranker 节点要求本地证据放在 rrf_chunks 中。
        state["rrf_chunks"] = [
            {
                "chunk_id": item.chunk_id,
                "content": item.content,
                "title": item.title,
            }
            for item in local_evidence
            if item.source != EvidenceSource.WEB
        ]

        # Reranker 节点要求网络证据放在 web_search_docs 中。
        state["web_search_docs"] = [
            {
                "snippet": item.content,
                "title": item.title,
                "url": item.url,
            }
            for item in web_evidence
        ]
        state["is_stream"] = False

        state = ReRankerNode().process(state)
        docs = state.get("reranked_docs") or []

        return [
            from_rerank_doc(doc, index + 1)
            for index, doc in enumerate(docs)
        ]

    @staticmethod
    def _to_milvus_raw(evidence: Evidence) -> dict:
        """把统一 Evidence 转回 RRF 节点需要的临时结构。"""
        return {
            "id": evidence.chunk_id,
            "entity": {
                "file_title": evidence.file_title,
                "title": evidence.title,
                "item_name": evidence.item_name,
                "content": evidence.content,
            },
        }
