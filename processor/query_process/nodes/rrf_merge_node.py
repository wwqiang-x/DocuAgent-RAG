import json

from processor.query_process.base import BaseNode
from processor.query_process.state import QueryGraphState


class RRFMergeNode(BaseNode):
    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 从state中获取两路结果
        embedding_chunks = state.get("embedding_chunks", [])
        hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])

        # 标准化两路结果数据结构
        validated_embedding_chunks = self._validate_search_chunks(embedding_chunks)
        validated_hyde_embedding_chunks = self._validate_search_chunks(hyde_embedding_chunks)

        # 记录一下过滤后的数量，方便排查
        self.logger.info(
            f"RRF 输入：embedding={len(validated_embedding_chunks)}, hyde={len(validated_hyde_embedding_chunks)}")

        # 设置两路权重
        rrf_inputs = [
            (validated_embedding_chunks, 1.0),
            (validated_hyde_embedding_chunks, 1.0)
        ]

        # rrf融合
        rrf_chunks = self._rrf_merge(rrf_inputs)

        # 取出Top_k (加个安全兜底：万一配置是0，默认给5)
        max_results = self.config.rrf_max_results if self.config.rrf_max_results > 0 else 5
        rrf_chunks = rrf_chunks[:max_results]

        # 设置state
        state["rrf_chunks"] = rrf_chunks
        return state

    def _validate_search_chunks(self, embedding_chunks):
        formatted_chunk_list = []
        if not embedding_chunks or not isinstance(embedding_chunks, list):
            return []

        for chunk in embedding_chunks:
            # 防御性检查：如果是空字典或者不是字典，跳过这一条，而不是清空全部
            if not chunk or not isinstance(chunk, dict):
                continue

            entity = chunk.get("entity")
            # 防御性检查：如果 entity 为空或者不是字典，也跳过
            if not entity or not isinstance(entity, dict):
                continue

            formatted_chunk = {
                "chunk_id": chunk.get("id"),
                "file_title": entity.get("file_title"),
                "title": entity.get("title"),
                "item_name": entity.get("item_name"),
                "content": entity.get("content"),
            }
            formatted_chunk_list.append(formatted_chunk)

        return formatted_chunk_list

    def _rrf_merge(self, rrf_inputs):
        # rrf算法：weight/（k+rank） rank排名 k = 60
        chunks_score = {}
        chunk_data = {}
        for chunks_list, weight in rrf_inputs:
            for index, chunk in enumerate(chunks_list):
                chunks_id = chunk.get("chunk_id")
                # 防御性检查：如果 chunk_id 为空，跳过
                if not chunks_id:
                    continue

                rank = index + 1
                rrf_score = weight / (self.config.rrf_k + rank)

                chunks_score[chunks_id] = chunks_score.get(chunks_id, 0.0) + rrf_score
                chunk_data.setdefault(chunks_id, chunk)

        # 结果处理
        rrf_results = []
        for chunks_id, score in chunks_score.items():
            chunk = chunk_data.get(chunks_id)
            rrf_results.append((chunk, score))

        # 按分数排序
        rrf_results = sorted(rrf_results, key=lambda x: x[1], reverse=True)
        rrf_chunks = [chunk for chunk, score in rrf_results]
        return rrf_chunks


if __name__ == "__main__":
    rrf_node = RRFMergeNode()
    # 你的测试代码可以保留