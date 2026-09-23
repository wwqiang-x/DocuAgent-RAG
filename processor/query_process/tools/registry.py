"""Agent 工具注册表。

注册表使用惰性导入，导入注册表本身不会初始化模型、数据库或网络客户端。
"""

import importlib
from typing import Any


class ToolRegistry:
    """集中管理阶段 A 暴露给 Agent 的工具。"""

    _TOOL_TYPES = {
        "resolve_product": (
            "processor.query_process.tools.product_tool",
            "ResolveProductTool",
        ),
        "search_local": (
            "processor.query_process.tools.local_search_tool",
            "LocalSearchTool",
        ),
        "search_web": (
            "processor.query_process.tools.web_search_tool",
            "WebSearchTool",
        ),
    }

    def __init__(self) -> None:
        self._tool_cache: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        """按名称获取工具，并在首次访问时惰性创建。"""
        if name not in self._TOOL_TYPES:
            raise KeyError(f"未注册的工具: {name}")

        if name not in self._tool_cache:
            module_name, class_name = self._TOOL_TYPES[name]
            module = importlib.import_module(module_name)
            tool_class = getattr(module, class_name)
            self._tool_cache[name] = tool_class()

        return self._tool_cache[name]

    def names(self) -> list[str]:
        """返回所有已注册工具名称。"""
        return list(self._TOOL_TYPES)
