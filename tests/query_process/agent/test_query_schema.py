"""验证查询模式 Schema 的兼容性。"""

import unittest

from schema.query_schema import QueryMode, QueryRequest


class QuerySchemaTest(unittest.TestCase):
    """查询请求 Schema 测试。"""

    def test_default_mode_is_auto(self) -> None:
        """不传 mode 时使用 auto，最终模式由后端路由决定。"""
        request = QueryRequest(query="测试问题")

        self.assertEqual(request.mode, QueryMode.AUTO)

    def test_agent_mode_can_be_selected(self) -> None:
        """可以显式选择 Agent 模式。"""
        request = QueryRequest(query="测试问题", mode="agent")

        self.assertEqual(request.mode, QueryMode.AGENT)

    def test_legacy_mode_can_be_selected(self) -> None:
        """可以显式选择固定 RAG 模式。"""
        request = QueryRequest(query="测试问题", mode="legacy")

        self.assertEqual(request.mode, QueryMode.LEGACY)


if __name__ == "__main__":
    unittest.main()
