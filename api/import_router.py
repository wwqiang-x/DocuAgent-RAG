import json

import uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks
from fastapi.params import Depends
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from services.upload_service import UploadService
from core.deps import get_upload_service
from core.paths import get_front_page_dir
from schema.upload_schema import UploadResponse, TaskStatusResponse
from services.upload_service import UploadService

#创建api实例
app = FastAPI(description="文档导入服务",version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,      # ← 默认值，可以省略
    allow_methods=["*"],
    allow_headers=["*"],
)

#挂载
page_path = get_front_page_dir()
if page_path:
    # html=True：访问 /front/ 时自动返回 index.html（否则目录请求返回 404）
    app.mount("/front",StaticFiles(directory = page_path, html=True))

@app.post("/upload",response_model=UploadResponse) #前端要以post发请求 用upload
def upload_file(
        file:UploadFile,
        background_tasks:BackgroundTasks, #后台任务管理器
        upload_service: UploadService = Depends(get_upload_service)
):
    #调用UploadService中的process方法
    message, task_id, import_file_path, file_dir = upload_service.process_upload_file(file)

    # 处理文件上传保存到minio以及处理文档时间花费较长时间
    # 我们将耗时较长的业务放在后台执行返回给用户一个结果
    background_tasks.add_task(
        upload_service.run_import_graph,
        import_file_path,file_dir,task_id
    )

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return UploadResponse(
        message=message,
        task_id=task_id
    )

@app.get("/status/{task_id}",response_model=TaskStatusResponse) # http//ip:8000/status/021
def status(
        task_id:str,
        upload_service: UploadService = Depends(get_upload_service)
):
    result = upload_service.get_status(task_id)
    ison_str = json.dumps(result, indent=4,ensure_ascii=False)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(ison_str)
    return result

class Student(BaseModel):
    name: str = Field(...,title="姓名",description="学生姓名")
    age: int = Field(...,title="年龄",description="学生年龄")


if __name__ == "__main__":
    #启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000,log_level="info")