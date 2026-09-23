from anyio.functools import lru_cache

from services.agent_query_service import AgentQueryService
from services.query_mode_service import QueryModeService
from services.query_service import QueryService
from services.upload_service import UploadService


@lru_cache
def get_upload_service():
    return UploadService()


@lru_cache
def get_query_service():
    return QueryService()


@lru_cache
def get_agent_query_service() -> AgentQueryService:
    """获取 Agent 模式查询服务。"""
    return AgentQueryService()


@lru_cache
def get_query_mode_service() -> QueryModeService:
    """获取查询模式路由和灰度服务。"""
    return QueryModeService()

