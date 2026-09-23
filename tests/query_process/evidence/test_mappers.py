"""验证统一契约和证据映射。"""

import unittest

from processor.query_process.evidence.mappers import (
    from_milvus_hit,
    from_rerank_doc,
    from_web_doc,
)
from processor.query_process.evidence.models import EvidenceSource
from processor.query_process.tools.contracts import SearchStrategy


class ContractsAndMappersTest(unittest.TestCase):
    """阶段 A 数据契约测试。"""

    def test_milvus_hit_mapping(self) -> None:
        """Milvus 原始字段应映射到统一 Evidence。"""
        hit = {
            "id": 101,
            "distance": 0.82,
            "entity": {
                "file_title": "万用表说明",
                "title": "电阻测量",
                "item_name": "RS-12数字万用表",
                "content": "测量电阻前应断开电源。",
            },
        }

        evidence = from_milvus_hit(
            hit,
            strategy=SearchStrategy.BASIC,
            rank=1,
        )

        self.assertEqual(evidence.chunk_id, 101)
        self.assertEqual(evidence.source, EvidenceSource.LOCAL)
        self.assertEqual(evidence.score, 0.82)
        self.assertEqual(evidence.metadata["strategy"], "basic")

    def test_web_result_mapping(self) -> None:
        """网页 snippet 应映射为 Evidence 正文。"""
        evidence = from_web_doc(
            {
                "title": "万用表测电阻",
                "url": "https://example.com/tool",
                "snippet": "测量时先断电。",
            },
            rank=1,
        )

        self.assertEqual(evidence.source, EvidenceSource.WEB)
        self.assertEqual(evidence.content, "测量时先断电。")
        self.assertEqual(evidence.url, "https://example.com/tool")

    def test_rerank_result_mapping(self) -> None:
        """Reranker 分数应保留，并标记为 rerank 类型。"""
        evidence = from_rerank_doc(
            {
                "chunk_id": 101,
                "title": "电阻测量",
                "content": "测量电阻步骤",
                "source": "local",
                "score": 6.75,
            },
            rank=2,
        )

        self.assertEqual(evidence.score, 6.75)
        self.assertEqual(evidence.score_type, "rerank")
        self.assertEqual(evidence.rank, 2)


if __name__ == "__main__":
    unittest.main()
