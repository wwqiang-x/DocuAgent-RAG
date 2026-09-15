from subprocess import Popen
import subprocess
import time
from typing import Tuple
from pathlib import Path
from processor.import_process.base import BaseNode
from processor.import_process.exceptions import StateFieldError
from processor.import_process.state import ImportGraphState


class PDfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    #校验文件路径以及目录是否存在
    def process(self, state: ImportGraphState):
        #校验参数
        pdf_path_obj,file_dir_obj = self._validate_state(state)
        #调用mineru工具
        process_code = self._execute_mineru(pdf_path_obj,file_dir_obj)
        #获取md路径
        md_path: str = self._get_md_path(pdf_path_obj,file_dir_obj)
        #把md文档传递给state状态
        state["md_path"] = md_path
        #返回state
        return state

    #校验函数
    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        #从上一个节点中获取文件路径和文件目录
        self.log_step("Step1", "校验参数")
        pdf_path = state["pdf_path"]
        file_dir = state["file_dir"]

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name="pdf_path",
                message="文件不存在"
            )

        file_dir_obj = Path(file_dir)
        if not file_dir_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name="file_dir",
                message="文件目录不存在"
            )

        return pdf_path_obj, file_dir_obj

    def _execute_mineru(self, pdf_path_obj: Path , file_dir_obj: Path)->int:
        #构建指令 mineru -p <pdf路径> -o <文件目录> -b pipeline --soure local
        self.log_step("Step2", "执行Mineru")
        cmd = [
            "mineru",
            "-p",str(pdf_path_obj),
            "-o",str(file_dir_obj),
            "-b","pipeline",
            "--source","local"
        ]

        #执行指令
        start_time = time.time()
        process = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            text=True,
            errors="replace",
            bufsize=1
        )

        #打印日志
        for line in process.stdout:
            self.logger.info(line.strip())

        #判断mineru的转换结果
        process_code = process.wait()
        end_time = time.time()
        if process_code != 0:
            raise StateFieldError(
                node_name=self.name,
                message="Mineru转换失败"
            )
        else:
            self.logger.info(f"Mineru转化成功,耗时:{end_time-start_time}秒")

        return process_code

    def _get_md_path(self, pdf_path_obj: Path, file_dir_obj: Path)->str:
        # 获取md文档路径
        self.log_step("Step3", "获取md文件路径")
        md_path = file_dir_obj / pdf_path_obj.stem / "auto" / f"{pdf_path_obj.stem}.md"
        return str(md_path)

if __name__ == "__main__":
    import json
    state = {
            "task_id": "",
            "is_pdf_read_enabled": True,
            "is_md_read_enabled": False,
            "file_dir": "D:\\knowledge_base\\processor\\import_process\\improcess_files",
            "import_file_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
            "pdf_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
            "md_path": "",
            "file_title": "万用表RS-12的使用",
            "md_content": "",
            "chunks": [],
            "item_name": ""
    }

    node = PDfToMdNode()
    state = node(state)
    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)
#测试输出
# {
#     "task_id": "",
#     "is_pdf_read_enabled": true,
#     "is_md_read_enabled": false,
#     "file_dir": "D:\\knowledge_base\\processor\\import_process\\improcess_files",
#     "import_file_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
#     "pdf_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
#     "md_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md",
#     "file_title": "万用表RS-12的使用",
#     "md_content": "",
#     "chunks": [],
#     "item_name": ""
# }








