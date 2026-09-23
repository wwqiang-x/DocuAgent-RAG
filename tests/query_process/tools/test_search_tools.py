"""验证本地检索和网络搜索工具。"""

import unittest
from unittest.mock import patch

from processor.query_process.evidence.models import EvidenceSource
from processor.query_process.tools.context import ToolContext
from processor.query_process.tools.contracts import (
    SearchStrategy,
    ToolStatus,
)
from processor.query_process.tools.local_search_tool import (
    LocalSearchRequest,
    LocalSearchTool,
)
from processor.query_process.tools.web_search_tool import (
    WebSearchRequest,
    WebSearchTool,
)


class SearchToolsTest(unittest.TestCase):
    """检索工具测试。"""

    @patch(
        "processor.query_process.tools.local_search_tool."
        "HybridVectorSearchNode.process"
    )
    def test_basic_local_search(self, mock_process) -> None:
        """基础检索应输出统一本地证据。"""
        mock_process.return_value = {
            "embedding_chunks": [
                {
                    "id": 1,
                    "distance": 0.9,
                    "entity": {
                        "title": "电阻测量",
                        "content": "测量电阻步骤",
                        "item_name": "RS-12数字万用表",
                    },
                }
            ]
        }

        result = LocalSearchTool().run(
            ToolContext(original_query="如何测量电阻"),
            LocalSearchRequest(strategy=SearchStrategy.BASIC),
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(
            result.output.evidences[0].source,
            EvidenceSource.LOCAL,
        )

    @patch(
        "processor.query_process.tools.web_search_tool."
        "WebMcpSearchNode.process"
    )
    def test_web_search(self, mock_process) -> None:
        """网络搜索结果应输出统一网络证据。"""
        mock_process.return_value = {
            "web_search_docs": [
                {
                    "title": "数字万用表测电阻",
                    "url": "https://example.com/1",
                    "snippet": "先断开电源。",
                }
            ]
        }

        result = WebSearchTool().run(
            ToolContext(original_query="数字万用表如何测量电阻"),
            WebSearchRequest(),
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(
            result.output.evidences[0].source,
            EvidenceSource.WEB,
        )


if __name__ == "__main__":
    unittest.main()
