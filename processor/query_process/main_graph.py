import json
from bson import json_util
import logging

from langgraph.graph import StateGraph
from processor.import_process.base import setup_logging
from processor.query_process.nodes.answer_output_node import AnswerOutputNode
from processor.query_process.nodes.hybrid_vector_search_node import HybridVectorSearchNode
from processor.query_process.nodes.hyde_vector_search_node import HyDEVectorSearchNode
from processor.query_process.nodes.item_name_confirmed_node import ItemNameConfirmedNode
from processor.query_process.nodes.reranker_node import ReRankerNode
from processor.query_process.nodes.rrf_merge_node import RRFMergeNode
from processor.query_process.nodes.web_mcp_search_node import WebMcpSearchNode
from processor.query_process.state import QueryGraphState, get_default_state


def my_router(state: QueryGraphState) -> QueryGraphState:
    # 只有当下游已经明确生成了 answer（比如真正的拒答），才短路
    # 如果答案是“询问用户候选”这种类型，必须继续走检索链路，让 RAGAS 拿到上下文
    answer = state.get("answer", "")
    if answer and "请问你是在询问以下内容吗" not in answer:
        return True
    else:
        return False

def create_query_graph():
    graph = StateGraph(QueryGraphState)
    #添加节点
    graph.add_node("item_name_confirmed_node",ItemNameConfirmedNode())

    #添加虚拟节点(多路召回)
    graph.add_node("multi_search",lambda x:x)

    graph.add_node("hybrid_vector_search_node",HybridVectorSearchNode())
    graph.add_node("hyde_vector_search_node", HyDEVectorSearchNode())
    graph.add_node("web_mcp_search_node",WebMcpSearchNode())
    #添加一个汇集的虚拟节点
    graph.add_node("join_node", lambda x: x)  # 原样返回输入
    graph.add_node("rrf_merge_node", RRFMergeNode())
    graph.add_node("reranker_node",ReRankerNode())
    graph.add_node("answer_output_node",AnswerOutputNode())
    # 边保持不变

    #添加边
    graph.add_edge("__start__","item_name_confirmed_node")
    graph.add_conditional_edges("item_name_confirmed_node",my_router,{
        True:"answer_output_node",
        False:"multi_search"
    })

    graph.add_edge("multi_search","hybrid_vector_search_node")
    graph.add_edge("multi_search","hyde_vector_search_node")
    graph.add_edge("multi_search", "web_mcp_search_node")

    graph.add_edge("hybrid_vector_search_node","join_node")
    graph.add_edge("hyde_vector_search_node", "join_node")
    graph.add_edge("web_mcp_search_node", "join_node")

    graph.add_edge("join_node", "rrf_merge_node")
    graph.add_edge("rrf_merge_node", "reranker_node")
    graph.add_edge("reranker_node", "answer_output_node")
    graph.add_edge("answer_output_node", "__end__")

    return graph.compile()


if __name__ == "__main__":
    # setup_logging(logging.INFO)
    graph = create_query_graph()
    # state = get_default_state()
    # state["session_id"] = "s101"
    # state["original_query"] = "它市场价多少钱"
    # state= graph.invoke(state)
    # json_str = json_util.dumps(state, indent=4, ensure_ascii=False)
    # print(json_str)
    graph.get_graph().draw_ascii()
if __name__ == "__main__":
    graph = create_query_graph()
    print(graph.get_graph().draw_ascii())











