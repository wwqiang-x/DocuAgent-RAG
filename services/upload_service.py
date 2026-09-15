import datetime
import logging
import shutil

from fastapi import UploadFile
import uuid
import os

from langgraph.graph import StateGraph
from processor.import_process.state import get_default_state
from core.paths import get_local_base_dir
from processor.import_process.base import setup_logging
from processor.import_process.exceptions import FileProcessingError
from processor.import_process.main_graph import create_import_graph
from utils.client.storage_clients import StorageClients
from processor.import_process.config import get_config
from utils.task_util import get_task_info, update_task_status,add_running_task, add_done_task, add_node_duration


class UploadService:

    def process_upload_file(self,file:UploadFile):
        #生成当前文件上传的task——id
        task_id = uuid.uuid4().hex[:8]
        #获取文件在服务器的目录路径
        base_dir = get_local_base_dir()
        file_dir = os.path.join(base_dir,task_id)

        #将文件保存到file_dir
        import_file_path = self.save_upload_file_to_local(file,file_dir)
        print(import_file_path)


        #将用户上传文件保存到minio：用户可以查看下载
        remote_url = self.save_upload_file_to_minio(import_file_path,file.filename)

        #存储文件及路径信息

        message = "文件上传成功，文件处理中"
        return message, task_id,import_file_path, file_dir


    def run_import_graph(self,import_file_path,file_dir,task_id):
        setup_logging(logging.DEBUG)
        graph: StateGraph = create_import_graph()
        state = get_default_state()
        state["import_file_path"] = import_file_path
        state["file_dir"] = file_dir
        state["task_id"] = task_id
        try:
            update_task_status(task_id,"running")
            for event in graph.stream(state):
                for node, output in event.items():
                    logging.info(f"{node}节点完成，输出结果:{output}")
            update_task_status(task_id, "completed")
        except Exception as e:
            update_task_status(task_id, "failed")
            logging.error(f"任务处理失败: {e}")



    def save_upload_file_to_local(self,file,file_dir):
        #创建目录
        os.makedirs(file_dir,exist_ok=True)
        #生成文件路径
        import_file_path = os.path.join(file_dir, file.filename)  # 修正这里
        try:
            with open(import_file_path, 'wb') as f:
                # 文件拷贝
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            raise FileProcessingError(f"文件保存失败:{e}")
        return import_file_path


    def save_upload_file_to_minio(self,import_file_path,filename):
        #获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            logging.logger.info(f"获取minio客户端失败{e}")
            return
        #上传文件
        bucket_name = get_config().minio_bucket
        object_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{filename}"
        try:
            minio_client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=import_file_path,
            )
        except Exception as e:
            logging.logger.error(f"上传文件失败:{e}")
            return
        #返回pdf在minio的路径
        remoto_url = f"{get_config().get_minio_base_url()}/{bucket_name}/{object_name}"
        return remoto_url


    def get_status(self,task_id):
        return get_task_info(task_id)
















