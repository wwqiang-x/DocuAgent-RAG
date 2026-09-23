"""验证查询 API 在阶段 C 后可以正常导入。"""

import unittest

from api.query_router import app


class QueryRouterImportTest(unittest.TestCase):
    """查询路由导入测试。"""

    def test_query_routes_exist(self) -> None:
        """阶段 C 不应破坏原有查询和流式路由。"""
        paths = {route.path for route in app.routes}

        self.assertIn("/query", paths)
        self.assertIn("/stream/{task_id}", paths)
        self.assertIn("/history/{session_id}", paths)


if __name__ == "__main__":
    unittest.main()
