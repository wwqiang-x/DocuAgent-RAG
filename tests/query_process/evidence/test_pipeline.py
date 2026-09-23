"""验证 RRF 和 Reranker 适配逻辑。"""

import unittest
from unittest.mock import patch

from processor.query_process.evidence.models import (
    Evidence,
    EvidenceSource,
)
from processor.query_process.evidence.pipeline import EvidencePipeline
from processor.query_process.tools.contracts import SearchOutput


class EvidencePipelineTest(unittest.TestCase):
    """证据处理管线测试。"""

    @patch(
        "processor.query_process.evidence.pipeline."
        "RRFMergeNode.process"
    )
    def test_fuse_local(self, mock_process) -> None:
        """RRF 输出应转换为统一 Evidence。"""
        mock_process.return_value = {
            "rrf_chunks": [
                {
                    "chunk_id": 1,
                    "file_title": "万用表说明",
                    "title": "电阻测量",
                    "item_name": "RS-12数字万用表",
                    "content": "测量电阻步骤",
                }
            ]
        }
        evidence = Evidence(
            evidence_id="local:1",
            source=EvidenceSource.LOCAL,
            chunk_id=1,
            content="测量电阻步骤",
        )

        result = EvidencePipeline().fuse_local(
            SearchOutput(evidences=[evidence]),
            None,
        )

        self.assertEqual(result[0].chunk_id, 1)
        self.assertEqual(result[0].score_type, "rrf")

    @patch(
        "processor.query_process.evidence.pipeline."
        "ReRankerNode.process"
    )
    def test_rerank(self, mock_process) -> None:
        """Reranker 输出应转换为统一 Evidence。"""
        mock_process.return_value = {
            "reranked_docs": [
                {
                    "chunk_id": 1,
                    "title": "电阻测量",
                    "content": "测量电阻步骤",
                    "source": "local",
                    "score": 6.5,
                }
            ]
        }
        evidence = Evidence(
            evidence_id="local:1",
            source=EvidenceSource.LOCAL,
            chunk_id=1,
            content="测量电阻步骤",
        )

        result = EvidencePipeline().rerank(
            query="如何测量电阻",
            local_evidence=[evidence],
        )

        self.assertEqual(result[0].score, 6.5)
        self.assertEqual(result[0].score_type, "rerank")


if __name__ == "__main__":
    unittest.main()
