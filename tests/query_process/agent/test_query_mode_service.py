"""验证阶段 D 的查询模式路由和灰度逻辑。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from schema.query_schema import QueryMode
from services.query_mode_service import QueryModeService


class QueryModeServiceTest(unittest.TestCase):
    """查询模式路由测试。"""

    def _config(
        self,
        *,
        enabled: bool = True,
        default_mode: str = "legacy",
        canary_percent: int = 0,
        fallback: bool = True,
    ) -> SimpleNamespace:
        """构造测试配置。"""
        return SimpleNamespace(
            agentic_rag_enabled=enabled,
            default_query_mode=default_mode,
            agent_canary_percent=canary_percent,
            agent_canary_salt="test-salt",
            agent_fallback_to_legacy=fallback,
            agent_max_steps=6,
        )

    @patch("services.query_mode_service.get_config")
    def test_explicit_legacy_always_wins(self, mock_get_config) -> None:
        """显式 legacy 必须始终走固定 RAG。"""
        mock_get_config.return_value = self._config(default_mode="agent")

        decision = QueryModeService().resolve(
            requested_mode=QueryMode.LEGACY,
            session_id="session-001",
        )

        self.assertEqual(decision.effective_mode, QueryMode.LEGACY)

    @patch("services.query_mode_service.get_config")
    def test_disabled_agent_falls_back(self, mock_get_config) -> None:
        """Agent 总开关关闭时 auto 回退 legacy。"""
        mock_get_config.return_value = self._config(enabled=False)

        decision = QueryModeService().resolve(
            requested_mode=QueryMode.AUTO,
            session_id="session-001",
        )

        self.assertEqual(decision.effective_mode, QueryMode.LEGACY)
        self.assertIn("未启用", decision.reason)

    @patch("services.query_mode_service.get_config")
    def test_explicit_agent_is_enabled(self, mock_get_config) -> None:
        """Agent 启用后显式 agent 请求进入 Agent。"""
        mock_get_config.return_value = self._config()

        decision = QueryModeService().resolve(
            requested_mode=QueryMode.AGENT,
            session_id="session-001",
        )

        self.assertEqual(decision.effective_mode, QueryMode.AGENT)

    @patch("services.query_mode_service.get_config")
    def test_canary_is_stable_for_session(self, mock_get_config) -> None:
        """同一个 session 在 canary 模式下的分桶结果必须稳定。"""
        mock_get_config.return_value = self._config(
            default_mode="canary",
            canary_percent=50,
        )
        service = QueryModeService()

        first = service.resolve(
            requested_mode=QueryMode.AUTO,
            session_id="stable-session",
        )
        second = service.resolve(
            requested_mode=QueryMode.AUTO,
            session_id="stable-session",
        )

        self.assertEqual(first.effective_mode, second.effective_mode)
        self.assertEqual(first.canary_bucket, second.canary_bucket)


if __name__ == "__main__":
    unittest.main()
