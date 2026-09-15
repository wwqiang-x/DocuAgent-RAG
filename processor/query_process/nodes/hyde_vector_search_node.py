from typing import Dict, Any

from pymilvus import SearchResult

from langchain.messages import SystemMessage,HumanMessage
from processor.query_process.base import BaseNode, T
from processor.query_process.state import QueryGraphState
from prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors
from utils.milvus_util import milvus_client, create_hybrid_search_requests, execute_hybrid_search_query


class HyDEVectorSearchNode(BaseNode):

    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> Dict[str,Any]:
        # 1.参数校验 (rewritten_query,item_names)
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])

        # 2.利用LLM生成原始查询的假设性文档
        # 获取llm客户端
        llm_client = AIClients.get_llm_client(response_format=False)
        # 提示词
        system_prompt = f"您是一位{item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=item_names,
            rewritten_query=rewritten_query
        )
        # 调用大模型
        llm_res = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        hyde_doc = llm_res.content.strip() if llm_res else None

        # 3.对（修改后的原始查询+生成的假设性文档）进行向量化
        document = f"{rewritten_query}\n{hyde_doc}"
        bge_m3 = AIClients.get_bge_m3_client()
        vectors = generate_bge_m3_hybrid_vectors(bge_m3,[document])
        dense_vector = vectors["dense"][0]
        sparse_vector = vectors["sparse"][0]

        # 4.获取Milvus客户端
        milvus_client = StorageClients.get_milvus_client()
        # 5.生成混合向量检索请求
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr="item_name in {list}",
            expr_params={"list":item_names}
        )
        # 6.进行混合向量检索
        res:SearchResult = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_requests,
            output_fields=["file_title","title","content","item_name"]
        )
        # 7.处理搜索结果 返回处理后的state
        if not res or not res[0]:
            return {"hyde_embedding_chunks":[]}
        return {"hyde_embedding_chunks":res[0]}

if __name__ == '__main__':
    node = HyDEVectorSearchNode()
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
    import json
    state = node.process(state)
    json_str = json.dumps(state,indent=4,ensure_ascii=False)
    print(json_str)