import time
from pathlib import Path
from json import dumps
from processor.import_process import state
from processor.import_process.base import BaseNode
from processor.import_process.exceptions import StateFieldError
from processor.import_process.state import ImportGraphState, get_default_state
from utils.task_util import add_running_task, add_done_task, add_node_duration


class EntryNode(BaseNode):
    name = "entry_node"

    def process(self,state:ImportGraphState)-> ImportGraphState:
    # 1.从参数中获取import_file_path file_dir
        self.log_step("Step1","获取并校验state参数")
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")

    #2.检查文件
        if not import_file_path:
            raise StateFieldError(
                node_name = self.name,
                field_name = "import_file_path",
                message = "导入文件路径不能为空"
            )

        if not file_dir:
            raise StateFieldError(
                node_name = self.name,
                field_name = "file_dir",
                message="导入文件目录不能为空"
            )

    #3.检查文件是否存在
        self.log_step("Step2", "检查文件是否存在")
        path = Path(import_file_path)
        if not path.is_file():
            raise StateFieldError(
                node_name = self.name,
                field_name = "import_file_path",
                message="导入文件不存在"
            )

    #4.判断文件类型
        self.log_step("Step3", "判断文件类型")
        ext = path.suffix.lower()
        if ext == ".md":
            state["md_path"] = import_file_path
            state["is_md_read_enabled"] = True
        elif ext == ".pdf":
            state["pdf_path"] = import_file_path
            state["is_pdf_read_enabled"] = True
        else:
            raise StateFieldError(
            node_name= self.name,
            field_name = "import_file_path",
            message="不支持该文件类型"
        )

    #5.生成标题
        state["file_title"] = path.stem #..../标题2.pdf  读取“标题2”这一部分
        return state


if __name__ == "__main__":
    state = get_default_state()
    state["import_file_path"] = r"D:\knowledge_base\processor\import_process\improcess_files\万用表RS-12的使用.pdf"
    state["file_dir"] = r"D:\knowledge_base\processor\import_process\improcess_files"

    node = EntryNode()
    state = node(state)

    json_str = dumps(state, indent = 4, ensure_ascii=False)
    print(json_str)

# {
#     "task_id": "",
#     "is_pdf_read_enabled": true,
#     "is_md_read_enabled": false,
#     "file_dir": "D:\\knowledge_base\\processor\\import_process\\improcess_files",
#     "import_file_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
#     "pdf_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
#     "md_path": "",
#     "file_title": "万用表RS-12的使用",
#     "md_content": "",
#     "chunks": [],
#     "item_name": ""
# }






