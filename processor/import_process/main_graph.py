import json
import logging

from langgraph.graph import StateGraph
from processor.import_process.base import setup_logging
from processor.import_process.nodes.document_spliter_node import DocumentSpliterNode
from processor.import_process.nodes.embedding_chunks_node import EmbeddingChunksNode
from processor.import_process.nodes.entry_node import EntryNode
from processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from processor.import_process.nodes.md_img_node import MdImgNode
from processor.import_process.nodes.milvus_import_node import MlivusImportNode
from processor.import_process.nodes.pdf_to_md_node import PDfToMdNode
from processor.import_process.state import ImportGraphState, get_default_state


def my_router(state:ImportGraphState):
    # is_md_read_enabled: bool  # 是否启用 MD 读取
    # is_pdf_read_enabled: bool  # 是否启用 PDF 读取
    is_md_read_enabled = state.get('is_md_read_enabled')
    is_pdf_read_enabled = state.get("is_pdf_read_enabled")

    if is_md_read_enabled:
        return "md"
    elif is_pdf_read_enabled:
        return "pdf"
    else:
        return "unknown"

def create_import_graph()->StateGraph:
    #1.创建图实例
    graph = StateGraph(ImportGraphState)
    #2.添加节点
    graph.add_node("entry_node",EntryNode())
    graph.add_node("pdf_to_md_node",PDfToMdNode())
    graph.add_node("md_img_node",MdImgNode())
    graph.add_node("document_spliter_node",DocumentSpliterNode())
    graph.add_node("item_name_recognition_node",ItemNameRecognitionNode())
    graph.add_node("embedding_chunks_node",EmbeddingChunksNode())
    graph.add_node("milvus_import_node",MlivusImportNode())

    #3.定义边
    graph.add_edge("__start__","entry_node")
    #添加条件边
    graph.add_conditional_edges("entry_node",my_router,{
        "md":"md_img_node",
        "pdf":"pdf_to_md_node",
        "unknown":"__end__"
    })
    graph.add_edge("pdf_to_md_node","md_img_node")
    graph.add_edge("md_img_node","document_spliter_node")
    graph.add_edge("document_spliter_node","item_name_recognition_node")
    graph.add_edge("item_name_recognition_node","embedding_chunks_node")
    graph.add_edge("embedding_chunks_node","milvus_import_node")
    graph.add_edge("milvus_import_node","__end__")
    #4.编译状态
    return graph.compile()
    #5.返回状态图

if __name__ == '__main__':
    #开启日志
    setup_logging(logging.DEBUG)

    graph = create_import_graph()

    state = get_default_state()
    state["import_file_path"] = r"D:\knowledge_base\processor\import_process\improcess_files\hak180产品安全手册.pdf"
    state["file_dir"] = r"D:\knowledge_base\processor\import_process\improcess_files"

    state = graph.invoke(state)

    json_str = json.dumps(state,indent = 4,ensure_ascii=False)
    print(json_str)