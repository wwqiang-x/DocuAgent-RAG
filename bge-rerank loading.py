import os
from dotenv import load_dotenv
from modelscope import snapshot_download

load_dotenv()

# 使用 Path 对象或者 os.path.join 来拼接，完全避免转义问题
import pathlib
safe_local_dir = pathlib.Path("D:/my_models/bge-rerank")  # 使用正斜杠，Windows 也支持

local_dir = snapshot_download(
    model_id=os.getenv("BGE_RERANKER", "BAAI/bge-reranker-large"),
    local_dir=str(safe_local_dir)  # 转成字符串传给 modelscope
)

if __name__ == '__main__':
    print(local_dir)