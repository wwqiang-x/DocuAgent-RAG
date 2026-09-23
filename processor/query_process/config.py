"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。所有属性均采用懒加载模式。
"""
from dataclasses import dataclass, field
from typing import Optional
import os
from pathlib import Path

from dotenv import load_dotenv

# config.py 位于 D:\knowledge_base\processor\query_process\ 下
# 往上三级就是项目根目录 D:\knowledge_base
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 强制加载项目根目录的 .env，并 override=True 保证覆盖 PyCharm 可能注入的空字符串
load_dotenv(BASE_DIR / ".env", override=True)

# 指定绝对路径加载
load_dotenv(BASE_DIR / ".env")


def _env_bool(key: str, default: bool = False) -> bool:
    """读取布尔环境变量。"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    """读取整数环境变量，非法值使用默认值。"""
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== Agent 灰度配置 ====================
    # 是否允许进入 Agent 模式。
    agentic_rag_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "AGENTIC_RAG_ENABLED",
            True,
        )
    )

    # auto 模式的默认策略：legacy、agent 或 canary。
    default_query_mode: str = field(
        default_factory=lambda: os.getenv(
            "QUERY_DEFAULT_MODE",
            "legacy",
        ).strip().lower()
    )

    # canary 模式下进入 Agent 的 session 百分比。
    agent_canary_percent: int = field(
        default_factory=lambda: max(
            0,
            min(
                100,
                _env_int("AGENT_CANARY_PERCENT", 0),
            ),
        )
    )

    # 灰度分桶盐值，修改后同一个 session 的灰度结果会变化。
    agent_canary_salt: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_CANARY_SALT",
            "knowledge-base-agent-v1",
        )
    )

    # Agent 同步执行失败时是否自动回退旧链路。
    agent_fallback_to_legacy: bool = field(
        default_factory=lambda: _env_bool(
            "AGENT_FALLBACK_TO_LEGACY",
            True,
        )
    )

    # Agent 最大工具调用步数。
    agent_max_steps: int = field(
        default_factory=lambda: max(
            1,
            _env_int("AGENT_MAX_STEPS", 6),
        )
    )

    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "8"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.15"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "5"))
    )

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "5"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "5"))
    )

    # ==================== 商品确认节点配置 ====================
    item_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.75"))
    )
    item_name_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_MID_CONFIDENCE", "0.45"))
    )
    item_name_score_gap: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SCORE_GAP", "0.08"))
    )
    item_name_max_options: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_MAX_OPTIONS", "3"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.5"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.5"))
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )

    mcp_dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_API_KEY", "")
    )
    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。

        Returns:
            配置实例。
        """
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。

    Returns:
        全局配置实例。
    """
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config
