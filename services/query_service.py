import logging
import uuid
import traceback

from processor.import_process.base import setup_logging
from processor.query_process.state import get_default_state
from processor.query_process.main_graph import create_query_graph
from utils.mongo_history_util import get_recent_messages, clear_history
from utils.task_util import get_task_result


class QueryService:

    @staticmethod
    def gener_session():
        return str(uuid.uuid4())

    @staticmethod
    def gener_task_id():
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self, task_id, query, session_id, is_stream):
        try:
            setup_logging(logging.INFO)
            graph = create_query_graph()
            state = get_default_state()

            state["session_id"] = session_id
            state["task_id"] = task_id
            state["original_query"] = query
            state["is_stream"] = is_stream

            # 这里坚决不加任何 push_sse_event，完全交给 answer_output_node 去处理！
            state = graph.invoke(state)
            return state["answer"]

        except Exception as e:
            traceback.print_exc()
            raise e

    def get_task_result(self, task_id):
        return get_task_result(task_id=task_id, key="answer")

    def get_history(self,session_id:str):
        history_list = get_recent_messages(session_id=session_id,limit=20)
        for item in history_list:
            if "_id" in item:
                item["_id"] = str(item["_id"])
        return history_list

    def delete_history(self,session_id:str):
        deleted_count = clear_history(session_id)
        return deleted_count












