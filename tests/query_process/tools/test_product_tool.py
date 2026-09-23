"""验证商品识别工具适配器。"""

import unittest
from unittest.mock import patch

from processor.query_process.tools.context import ToolContext
from processor.query_process.tools.contracts import (
    ProductResolutionStatus,
    ToolStatus,
)
from processor.query_process.tools.product_tool import ResolveProductTool


class ResolveProductToolTest(unittest.TestCase):
    """商品识别工具测试。"""

    @patch(
        "processor.query_process.tools.product_tool."
        "ItemNameConfirmedNode.process"
    )
    def test_confirmed_product(self, mock_process) -> None:
        """确认商品时应返回 confirmed。"""
        mock_process.return_value = {
            "item_names": ["RS PRO RS-12 数字万用表"],
            "rewritten_query": "RS-12数字万用表如何测量电阻",
            "answer": "",
        }

        result = ResolveProductTool().run(
            ToolContext(
                original_query="RS-12数字万用表如何测量电阻",
            )
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(
            result.output.resolution,
            ProductResolutionStatus.CONFIRMED,
        )

    @patch(
        "processor.query_process.tools.product_tool."
        "ItemNameConfirmedNode.process"
    )
    def test_ambiguous_product(self, mock_process) -> None:
        """存在多个候选时应返回 ambiguous。"""
        mock_process.return_value = {
            "item_names": [],
            "rewritten_query": "数字万用表如何测量电阻",
            "answer": (
                "我不确定您指的是什么，请问你是在询问以下内容吗:\n"
                "[RS-12数字万用表, RS-05数字万用表]"
            ),
        }

        result = ResolveProductTool().run(
            ToolContext(original_query="数字万用表如何测量电阻")
        )

        self.assertEqual(result.status, ToolStatus.AMBIGUOUS)
        self.assertEqual(
            result.output.candidates,
            ["RS-12数字万用表", "RS-05数字万用表"],
        )


if __name__ == "__main__":
    unittest.main()
