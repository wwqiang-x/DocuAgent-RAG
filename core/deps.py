from anyio.functools import lru_cache

from services.query_service import QueryService
from services.upload_service import UploadService

@lru_cache
def get_upload_service():
    return UploadService()

@lru_cache()
def get_query_service():
    return QueryService()

