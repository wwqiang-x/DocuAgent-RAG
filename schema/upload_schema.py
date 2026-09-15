
from pydantic import BaseModel,Field
from typing import List,Dict,Any

class UploadResponse(BaseModel):
    message: str = Field(...,description="提示信息")
    task_id: str = Field(...,description="任务id用于追踪日志(web前端交互可以看见处理节点的日志)")


class TaskStatusResponse(BaseModel):
    """任务状态响应 —— GET /status/{task_id} 返回"""
    status: str = Field(..., description="任务状态")
    done_list: List[str] = Field(..., description="已完成节点列表")
    running_list: List[str] = Field(..., description="正在运行节点列表")
    durations: Dict[str, float] = Field(..., description="各节点耗时(秒)")

