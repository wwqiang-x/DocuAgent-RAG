import json
import logging

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import StateFieldError
from processor.import_process.state import ImportGraphState
from utils.client.ai_clients import AIClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors


class EmbeddingChunksNode(BaseNode):

    name = 'embedding_chunks_node'

    def process(self, state:ImportGraphState)->ImportGraphState:

        chunks = self._validate_state(state)

        #调用bge-m3生成向量
        #获取客户端
        bge_m3_client = AIClients.get_bge_m3_client()

        #文本嵌入
        dense_list = []
        sparse_list = []
        batch_count = self.config.embedding_batch_size # 8
        for index in range(0,len(chunks),batch_count):
            batch_start = index
            batch_end = batch_start + batch_count
            if batch_end > len(chunks):
                batch_end = len(chunks)
            chunks_batch = chunks[batch_start:batch_end]
            chunks_batch_content = [c['content'] for c in chunks_batch]
            result = generate_bge_m3_hybrid_vectors(bge_m3_client,chunks_batch_content)
            dense_list.extend(result['dense'])
            sparse_list.extend(result['sparse'])
            self.logger.info(f"处理完成{batch_start+1}~{batch_end}")

        #向量回填
        for index,chunk in enumerate(chunks):
            chunk['dense_vector'] = dense_list[index]
            chunk['sparse_vector'] = sparse_list[index]
        state["chunks"] = chunks
        return state

    def _validate_state(self,state):
        self.log_step("Step1", "校验chunks")
        chunks = state.get('chunks')
        if not chunks or not isinstance(chunks,list):
            raise StateFieldError(
                node_name="embedding_chunks_node",
                field_name="chunks",
                message="chunks有误",
                expected_type=list
            )

        for index,chunk in enumerate(chunks):
            if not isinstance(chunk,dict):
                raise StateFieldError(
                    node_name="embedding_chunks_node",
                    field_name="chunks",
                    message=f"[chunks_{index}不是期望的字典结果]",
                    expected_type = list
                )

        return chunks




if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    state = ImportGraphState()
    state["chunks"] = [{"content":f"chunk_{i}"}for i in range(20)]

    node = EmbeddingChunksNode()
    new_state = node.process(state)
    json_str = json.dumps(new_state,indent=4,ensure_ascii=False)
    print(json_str)






