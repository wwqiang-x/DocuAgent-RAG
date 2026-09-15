# 掌柜智库知识库系统 — 接口文档索引

> 基于 RAG 架构的知识库系统后端接口文档，由工程代码扫描归纳生成。
> 生成日期：2026-08-30

## 一、服务概览

系统后端由 **两个独立部署的 FastAPI 服务** 组成：

| 服务 | 入口文件 | 默认端口 | 服务描述 |
| --- | --- | --- | --- |
| 文档导入服务 | `api/import_router.py` | **8000** | 上传 PDF/Markdown 文档，后台执行解析、切分、向量化并导入 Milvus |
| 智能查询服务 | `api/query_router.py` | **8001** | 接收用户问题，执行混合检索（向量 + HyDE + 网络搜索）、重排序、LLM 生成答案 |

> ⚠️ 根目录下的 `router.py` / `services.py` 是查询服务的**旧版本**（history/status 为占位实现），
> 请以 `api/` 目录下的文件为准。

## 二、接口一览

### 查询流程（端口 8001）

| 方法 | 路径 | 说明 | 文档章节 |
| --- | --- | --- | --- |
| POST | `/query` | 提交查询（同步 / 流式两种模式） | [查询流程接口文档](查询流程接口文档.md#21-post-query-提交查询) |
| GET | `/stream/{task_id}` | 建立 SSE 连接，接收流式答案与进度 | [查询流程接口文档](查询流程接口文档.md#22-get-streamtask_id-流式结果SSE) |
| GET | `/status/{task_id}` | 查询任务处理进度 | [查询流程接口文档](查询流程接口文档.md#23-get-statustask_id-任务进度) |
| GET | `/history/{session_id}` | 获取会话历史记录（最近 20 条） | [查询流程接口文档](查询流程接口文档.md#24-get-historysession_id-会话历史) |
| DELETE | `/history/{session_id}` | 清空会话历史记录 | [查询流程接口文档](查询流程接口文档.md#25-delete-historysession_id-清空历史) |
| GET | `/chat` | 前端聊天页面静态资源 | — |

### 导入流程（端口 8000）

| 方法 | 路径 | 说明 | 文档章节 |
| --- | --- | --- | --- |
| POST | `/upload` | 上传文档，后台执行导入流水线 | [导入流程接口文档](导入流程接口文档.md#21-post-upload-上传文档) |
| GET | `/status/{task_id}` | 查询导入任务处理进度 | [导入流程接口文档](导入流程接口文档.md#22-get-statustask_id-任务进度) |
| GET | `/front` | 前端页面静态资源 | — |

## 三、通用约定

- **协议**：HTTP/JSON；上传接口为 `multipart/form-data`；流式接口为 SSE（`text/event-stream`）。
- **认证**：当前所有接口均**无鉴权**。
- **跨域**：两个服务均开启 CORS，`allow_origins=["*"]`。
- **任务 ID**：
  - 查询任务 `task_id`：UUID hex 前 12 位（如 `3f8a2c1b9d04`）；
  - 导入任务 `task_id`：UUID hex 前 8 位（如 `3f8a2c1b`）；
  - 会话 `session_id`：完整 UUID（如 `3f8a2c1b-9d04-4e5a-8b7c-6d5e4f3a2b1c`）。
- **任务状态存储**：任务状态/进度保存在服务进程内存中（`utils/task_util.py`），**服务重启后状态丢失**。
- **服务启动方式**：

  ```bash
  # 导入服务（端口 8000）
  uvicorn api.import_router:app --host 0.0.0.0 --port 8000

  # 查询服务（端口 8001）
  uvicorn api.query_router:app --host 0.0.0.0 --port 8001
  ```

## 四、依赖的外部组件

| 组件 | 用途 | 相关配置项（.env） |
| --- | --- | --- |
| Milvus | 向量数据库：切片集合 `kb_chunks`、商品名集合 `kb_item_names` | `MILVUS_URL`、`CHUNKS_COLLECTION`、`ITEM_NAME_COLLECTION` |
| MongoDB | 对话历史存储：库 `kb001`，集合 `chat_message` | `MONGO_URL`、`MONGO_DB_NAME` |
| MinIO | 原始文件与文档图片对象存储：桶 `knowledge-base` | `MINIO_ENDPOINT`、`MINIO_BUCKET_NAME` |
| LLM（DashScope 兼容） | 商品名识别、查询改写、答案生成 | `OPENAI_API_BASE`、`LLM_DEFAULT_MODEL`、`VL_MODEL`、`ITEM_MODEL` |
| BGE-M3（本地） | 稠密 + 稀疏向量生成 | `BGE_M3_PATH`、`BGE_DEVICE` |
| BGE-Reranker（本地） | 检索结果重排序 | `BGE_RERANKER_LARGE` |
| MinerU（本地） | PDF 转 Markdown | `MINERU_MODEL_SOURCE`、`MODELSCOPE_CACHE` |
| DashScope MCP WebSearch | 网络搜索补充召回 | `MCP_DASHSCOPE_BASE_URL` |

## 五、详细文档

- [查询流程接口文档](查询流程接口文档.md)
- [导入流程接口文档](导入流程接口文档.md)
