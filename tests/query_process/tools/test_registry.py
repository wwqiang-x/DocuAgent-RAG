"""验证工具注册表的惰性行为。"""

import unittest

from processor.query_process.tools.registry import ToolRegistry


class ToolRegistryTest(unittest.TestCase):
    """注册表基础测试。"""

    def test_names(self) -> None:
        """阶段 A 应注册三个公开工具。"""
        registry = ToolRegistry()

        self.assertEqual(
            registry.names(),
            ["resolve_product", "search_local", "search_web"],
        )

    def test_unknown_tool_raises(self) -> None:
        """未知工具应在导入外部节点前直接报错。"""
        registry = ToolRegistry()

        with self.assertRaises(KeyError):
            registry.get("unknown_tool")


if __name__ == "__main__":
    unittest.main()
