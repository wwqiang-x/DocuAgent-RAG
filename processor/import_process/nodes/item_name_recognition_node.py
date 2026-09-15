from typing import Tuple,List,Dict
from pymilvus import MilvusClient,DataType
from processor.import_process.base import BaseNode
from processor.import_process.exceptions import StateFieldError
from processor.import_process.state import ImportGraphState
from prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors


class ItemNameRecognitionNode(BaseNode):
    name = 'item_name_recognition_node'


    def process(self,state:ImportGraphState)->ImportGraphState:
        #校验参数
        file_title,chunks = self._validate_state(state)
        #构造上下文提示词
        item_name_content = self._bulid_llm_context(chunks)
        #调用llm进行商品名称的识别
        item_name = self._recognize_item_name_by_llm(item_name_content,file_title)
        #文本嵌入
        bge_m3_client = AIClients.get_bge_m3_client()
        result = generate_bge_m3_hybrid_vectors(bge_m3_client,[item_name])
        item_name_dense = result["dense"][0]
        item_name_sparse = result["sparse"][0]
        #存储到Milvus向量数据库中
        self._store_to_milvus(file_title,item_name,item_name_dense,item_name_sparse)
        #item_name回填chunks
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name
        state["chunks"] = chunks
        return state

    def _validate_state(self,state)->Tuple[str,List]:
        self.log_step("Step1", "校验参数")
        #判断state节点的传入是否正常
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(
                node_name=self.name,
                field_name="file_title",
                message="file_title为空"
            )

        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks,list):
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                message="chunks有误",
                expected_type=list
            )

        return file_title,chunks

    def _bulid_llm_context(self,chunks)->str:
        self.log_step("Step2","构造llm识别商品")
        final_context=[]
        total_length=0
        for i,chunk in enumerate(chunks[:self.config.item_name_chunk_k]):
            content = chunk.get("content")
            content_str = f"【切片{i+1}】=>{content}"
            if total_length + len(content_str) < self.config.item_name_chunk_size:
                final_context.append(content_str)
                total_length += len(content_str)
        return "\n".join(final_context)

    def _recognize_item_name_by_llm(self,item_name_content,file_title)->str:
        self.log_step("Step3", "调用llm进行商品名称的识别")
        #在import_prompt中定义调用
        #创建客户端
        llm_client = AIClients.get_llm_client(response_format= False)
        #提示词的调用
        system_prompt = ITEM_NAME_SYSTEM_PROMPT
        user_prompt = ITEM_NAME_USER_PROMPT.format(
            file_title=file_title,
            context = item_name_content
        )
        result = llm_client.invoke([
            {"role":"system","content":system_prompt},
            {"role": "user", "content":user_prompt}
        ])

        #处理结果
        str = result.content.strip()
        if str == "UNKNOWN":
            return file_title
        else:
            return str

    def _store_to_milvus(self,file_title,item_name,item_name_dense,item_name_sparse):
        #创建客户端
        milvus_client = StorageClients.get_milvus_client()

        #判断collection的存在
        collection_name = self.config.item_name_collection
        if not milvus_client.has_collection(collection_name):
            self._creat_item_name_collection(collection_name,milvus_client)

        #插入数据
        result = milvus_client.insert(
            collection_name = collection_name,
            data = [
                    {
                    "file_title":file_title,
                    "item_name":item_name,
                    "dense_vector":item_name_dense,
                    "sparse_vector":item_name_sparse
                }
            ]
        )
        self.logger.info(f"向milvus数据库中插入数据成功：{result}")

    def _creat_item_name_collection(self,collection_name,milvus_client):
        # 1.创建约束
        schema = milvus_client.create_schema()
        # 1.1 主键字段
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        # 1.2 标量字段
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=256
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=256
        )
        # 1.3 向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 2.创建索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",  # 建立索引的字段
            index_name="dense_vector_index",  # 索引名称
            index_type="AUTOINDEX",  # 索引类型  https://milvus.io/docs/zh/hnsw.md
            metric_type="COSINE",  # 向量度量类型(L2/IP/COSINE)，Milvus计算出来的稠密向量已经进行了归一化处理，所以这里使用COSINE和IP效果一样
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",  # IP 和 BM25
        )

        # 3.创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"Milvus集合{collection_name}创建成功")



























