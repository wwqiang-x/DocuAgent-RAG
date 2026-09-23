"""查询模式路由和灰度服务。"""

import hashlib
from dataclasses import dataclass

from processor.query_process.config import get_config
from schema.query_schema import QueryMode


@dataclass(frozen=True)
class QueryRoutingDecision:
    """一次查询的模式路由结果。"""

    requested_mode: QueryMode
    effective_mode: QueryMode
    reason: str
    canary_bucket: int | None
    fallback_to_legacy: bool
    max_steps: int


class QueryModeService:
    """根据请求、配置和灰度比例解析最终查询模式。"""

    def resolve(
        self,
        requested_mode: QueryMode,
        session_id: str,
    ) -> QueryRoutingDecision:
        """解析实际使用 legacy 还是 agent。"""

        config = get_config()

        # 明确选择 legacy 时，永远不走 Agent。
        if requested_mode == QueryMode.LEGACY:
            return self._decision(
                requested_mode=requested_mode,
                effective_mode=QueryMode.LEGACY,
                reason="请求明确指定 legacy",
                config=config,
            )

        # Agent 总开关关闭时，一切请求回退 legacy。
        if not config.agentic_rag_enabled:
            return self._decision(
                requested_mode=requested_mode,
                effective_mode=QueryMode.LEGACY,
                reason="Agent 模式未启用",
                config=config,
            )

        # 明确选择 agent 时，不参与 canary 分流。
        if requested_mode == QueryMode.AGENT:
            return self._decision(
                requested_mode=requested_mode,
                effective_mode=QueryMode.AGENT,
                reason="请求明确指定 agent",
                config=config,
            )

        # auto 模式由后端默认配置决定。
        default_mode = config.default_query_mode.lower()

        if default_mode == "agent":
            return self._decision(
                requested_mode=requested_mode,
                effective_mode=QueryMode.AGENT,
                reason="后端默认模式为 agent",
                config=config,
            )

        if default_mode == "canary":
            bucket = self._stable_bucket(
                session_id=session_id,
                salt=config.agent_canary_salt,
            )
            effective_mode = (
                QueryMode.AGENT
                if bucket < config.agent_canary_percent
                else QueryMode.LEGACY
            )
            return self._decision(
                requested_mode=requested_mode,
                effective_mode=effective_mode,
                reason=(
                    f"canary 分桶={bucket}, "
                    f"阈值={config.agent_canary_percent}"
                ),
                config=config,
                canary_bucket=bucket,
            )

        return self._decision(
            requested_mode=requested_mode,
            effective_mode=QueryMode.LEGACY,
            reason="后端默认模式为 legacy",
            config=config,
        )

    @staticmethod
    def _stable_bucket(session_id: str, salt: str) -> int:
        """根据 session_id 生成稳定的 0-99 分桶。"""
        digest = hashlib.sha256(
            f"{salt}:{session_id}".encode("utf-8")
        ).hexdigest()
        return int(digest[:8], 16) % 100

    @staticmethod
    def _decision(
        requested_mode: QueryMode,
        effective_mode: QueryMode,
        reason: str,
        config,
        canary_bucket: int | None = None,
    ) -> QueryRoutingDecision:
        """构造路由结果。"""
        return QueryRoutingDecision(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            reason=reason,
            canary_bucket=canary_bucket,
            fallback_to_legacy=config.agent_fallback_to_legacy,
            max_steps=config.agent_max_steps,
        )
