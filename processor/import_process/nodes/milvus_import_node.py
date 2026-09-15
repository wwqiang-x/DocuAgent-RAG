from dataclasses import dataclass
from pymilvus import MilvusClient
from typing import Optional, Sequence, List, Dict, Any
from pymilvus import DataType
from pymilvus import IndexType
from pymilvus.milvus_client import IndexParams

from processor.import_process.base import BaseNode
from processor.import_process.exceptions import StateFieldError
from processor.import_process.state import ImportGraphState
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients


class MlivusImportNode(BaseNode):
    name = "milvus_import_node"

    def process(self,state:ImportGraphState)->ImportGraphState:
        #校验参数
        chunks = self._validate_state(state)
        #获取milvus客户端
        milvus_client = StorageClients.get_milvus_client()
        #创建集合
        collection_name = self.config.chunks_collection
        if not milvus_client.has_collection(collection_name):
            self._crate_chunks_collection(milvus_client,collection_name)
        #将chunks存储到milvus中
        chunks = _MilvusInserter(milvus_client,collection_name).insert_rows(chunks)
        state["chunks"] = chunks
        return state
    def _validate_state(self,state):
        chunks = state.get("chunks")
        #校验非空和类型
        if not chunks or not isinstance(chunks,list):
            raise StateFieldError(
                node_name="milvus_import_node",
                field_name="chunks",
                message="chunks is required",
                expected_type=list
            )
        #校验每个chunks
        for chunk in chunks:
            if not isinstance(chunk,dict):
                raise StateFieldError(
                    node_name="milvus_import_node",
                    field_name="chunks",
                    message="chunks is not dict",
                    expected_type=dict
                )
            if "sparse_vector" not in chunk or "dense_vector" not in chunk:
                raise StateFieldError(
                    node_name="milvus_import_node",
                    field_name="chunks",
                    message="chunks没有向量的生成",
                    expected_type=dict
                )
        return chunks


    def _crate_chunks_collection(self,milvus_client,collection_name):
        #建造者模式
        schema = _MilvusSchemaBuilder.build_milvus_schema(milvus_client)
        #索引创建
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)

        milvus_client.create_collection(
            collection_name = collection_name,
            schema = schema,
            index_params = index_params
        )
        self.logger.info(f"Milvus集合{collection_name}创建成功")





@dataclass
class _SCALAR_FIELD_SPC:
    field_name:str
    datatype:DataType
    max_length:Optional[int]=None
#标量字段列表
_SCALAR_FIELDS : Sequence[_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="content",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535),
)

class _MilvusSchemaBuilder:
    @staticmethod
    def build_milvus_schema(milvus_client):
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        #主键字段
        schema.add_field(field_name="id",datatype = DataType.INT64 , is_primary=True ,auto_id = True)
        #标量字段
        for scalar_field in _SCALAR_FIELDS:
            args: Dict = {
                "field_name": scalar_field.field_name,
                "datatype": scalar_field.datatype
            }
            if scalar_field.max_length:
                args["max_length"] = scalar_field.max_length
            schema.add_field(**args)
        #向量字段
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim = 1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        return schema

class _MilvusIndexBuilder:
    @staticmethod
    def build_index_params(milvus_client):
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name = "dense_vector",
            index_name = "dense_vector_index",
            index_type = "AUTOINDEX",
            metric_type = "COSINE",
        )

        index_params.add_index(
            field_name = "sparse_vector",
            index_name = "sparse_vector_index",
            index_type = "SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        return index_params

class _MilvusInserter:

    def __init__(self,milvus_client:MilvusClient,collection_name):
        self.milvus_client = milvus_client
        self.collection_name = collection_name

    def insert_rows(self,chunks:List[Dict[str,Any]]):
        # 1.执行插入操作
        insert_results = self.milvus_client.insert(self.collection_name,chunks)
        # 2.将自动生成id回填到chunks中
        chunk_ids = insert_results.get('ids')
        for id,chunk in zip(chunk_ids,chunks):
            chunk["chunk_id"] = id
        # 3.返回
        return chunks
















