import json
from typing import Dict, Any
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest
from processor.query_process.base import BaseNode, T
from processor.query_process.state import QueryGraphState
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors
from utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class HybridVectorSearchNode(BaseNode):

    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> Dict[str,Any]:
        # 1.校验参数
        rewritten_query = state.get("rewritten_query","")
        item_names = state.get("item_names",[])

        if not rewritten_query or not rewritten_query.strip():
            rewritten_query = state.get("original_query", "")
            if not rewritten_query or not rewritten_query.strip():
                return {"embedding_chunks": []}

        # 2.获取嵌入模型客户端 bge-m3  (AiClients)
        bge_m3 = AIClients.get_bge_m3_client()

        # 3.查询问题向量化(rewritten_query)
        vectors = generate_bge_m3_hybrid_vectors(bge_m3,[rewritten_query])
        dense_vector = vectors["dense"][0]
        sparse_vector = vectors["sparse"][0]

        # 4.获取Milvus客户端  (StorageClients)
        milvus_client = StorageClients.get_milvus_client()

        # 5.创建混合搜索请求(带过滤条件)
        # 修改后的代码
        item_names = state.get("item_names", [])

        # 动态判断是否需要过滤条件
        expr = "item_name in {list}" if item_names else None
        expr_params = {"list": item_names} if item_names else None

        hybrid_requests = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            expr_params=expr_params,
            limit=8
        )

        # 6.执行混合搜索
        hybird_result = execute_hybrid_search_query(
            milvus_client = milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_requests,
            output_fields = ["file_title","title","content","item_name"]
        )

        # 7.处理搜索结果,并返回state
        if not hybird_result or not hybird_result[0]:
            #return state
            return {"embedding_chunks":[]}

        # state["embedding_chunks"] = hybird_result[0]
        # return state
        return {"embedding_chunks":hybird_result[0]}


if __name__ == '__main__':
    node = HybridVectorSearchNode()
    # state = {
    #     "session_id": "",
    #     "task_id": "",
    #     "message_id": "",
    #     "original_query": "如何使用RS-12s数字万用表测量电阻？",
    #     "embedding_chunks": [],
    #     "hyde_embedding_chunks": [],
    #     "rrf_chunks": [],
    #     "web_search_docs": [],
    #     "reranked_docs": [],
    #     "prompt": "",
    #     "answer": "",
    #     "item_names": [
    #         "RS PRO RS-12 数字万用表"
    #     ],
    #     "rewritten_query": "如何使用RS-12s数字万用表测量电阻？",
    #     "history": [],
    #     "is_stream": False
    # }

    state = {
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "RS-12数字万用表如何测量电阻",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "rrf_chunks": [],
    "web_search_docs": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [
        "RS PRO RS-12 数字万用表"
    ],
    "rewritten_query": "RS-12数字万用表如何测量电阻？",
    "history": [],
    "is_stream": False
}

    state = node.process(state)
    json_str = json.dumps(state,indent=4,ensure_ascii=False)
    print(json_str)