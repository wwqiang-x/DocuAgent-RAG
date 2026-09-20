"""
基于 ragas 的 RAG 评估测试程序（eval/eval.py）

【功能概述】
    1. 从 eval/qs.csv 读取评估问题集（两列：question、ground_truth）
    2. 逐条调用 RAG 查询流程（入口：processor/query_process/main_graph.py）
    3. 从流程执行后的 state 中提取答案与检索上下文
    4. 使用 ragas 计算五项指标：
       Faithfulness（忠实度）、Answer Relevancy（答案相关性）、
       Context Precision（上下文精确率）、Context Recall（上下文召回率）、
       Answer Correctness（答案正确性）
    5. 将评估结果写入 eval/qa_result.csv（UTF-8 BOM 编码，共 9 列）

【运行前准备】
    1. 安装 ragas：uv add ragas（或 pip install ragas）
    2. 确保 MongoDB、Milvus 及 BGE-M3 / 重排序模型环境可用（RAG 流程依赖）

【运行方式】
    在项目根目录执行：python eval/eval.py
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# 加载 .env 环境变量（与项目其他模块保持一致）
load_dotenv()

# ==================== 路径引导 ====================
# 将项目根目录加入 sys.path，保证从任意位置运行本脚本时都能 import 到 processor / utils 等模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==================== 第三方依赖 ====================
import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# ==================== 项目内部依赖 ====================
from processor.import_process.base import setup_logging
from processor.query_process.main_graph import create_query_graph
from processor.query_process.state import get_default_state
from utils.client.ai_clients import AIClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors

# ==================== 常量定义 ====================

EVAL_DIR = PROJECT_ROOT / "eval"  # 评估目录
INPUT_CSV_PATH = EVAL_DIR / "qs.csv"  # 输入：问题集文件
OUTPUT_CSV_PATH = EVAL_DIR / "qa_result.csv"  # 输出：评估结果文件

# 输出 CSV 的 9 列（顺序固定，列名按需求原文保留）
RESULT_COLUMNS = [
    "question",
    "context",
    "answer",
    "ground_trush",
    "Faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
    "Answer Correctness",
]

# ragas 指标名 → 输出 CSV 列名 的映射
METRIC_COLUMN_MAP = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_correctness": "Answer Correctness",
}

logger = logging.getLogger("eval.ragas")


class BgeM3RagasEmbeddings(BaseRagasEmbeddings):
    """BGE-M3 嵌入模型适配器：把项目中的 BGE-M3 客户端包装成 ragas 需要的嵌入接口

    BGE-M3 客户端由 utils/client/ai_clients.py 提供；
    向量生成复用 utils/embedding_util.py 中的 generate_bge_m3_hybrid_vectors
    （BGE-M3 输出稠密 + 稀疏向量，ragas 只需要稠密 dense 部分）。
    """

    def __init__(self, bge_m3_client):
        """初始化适配器

        Args:
            bge_m3_client: AIClients.get_bge_m3_client() 返回的 BGEM3EmbeddingFunction 客户端
        """
        super().__init__()
        # 保存 BGE-M3 客户端（私有属性，不参与 pydantic 校验）
        self._bge_m3_client = bge_m3_client

    def _encode_dense(self, texts: List[str]) -> List[List[float]]:
        """调用项目统一入口（utils/embedding_util.py）为文本生成稠密向量

        Args:
            texts: 待向量化的文本列表

        Returns:
            稠密向量列表，每个元素为一个文本对应的向量
        """
        if not texts:
            return []
        vectors = generate_bge_m3_hybrid_vectors(self._bge_m3_client, list(texts))
        return vectors["dense"]

    def embed_query(self, text: str) -> List[float]:
        """生成单条查询文本的稠密向量（ragas 同步接口）"""
        return self._encode_dense([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文档文本的稠密向量（ragas 同步接口）"""
        return self._encode_dense(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """异步版本：生成单条查询文本的稠密向量（在线程池中执行同步逻辑）"""
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步版本：批量生成文档文本的稠密向量（在线程池中执行同步逻辑）"""
        return await asyncio.to_thread(self.embed_documents, texts)


def step_1_read_questions() -> pd.DataFrame:
    """步骤1：从 eval/qs.csv 读取评估问题集

    Returns:
        DataFrame，包含两列：question（问题）、ground_truth（参考答案）

    Raises:
        FileNotFoundError: qs.csv 不存在时抛出
        ValueError: 缺少必需的列时抛出
    """
    # 1. 校验输入文件是否存在
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"评估问题文件不存在：{INPUT_CSV_PATH}")

    # 2. 读取 CSV（utf-8-sig 兼容带 / 不带 BOM 的文件）
    question_df = pd.read_csv(INPUT_CSV_PATH, encoding="utf-8-sig")

    # 3. 校验必需的列
    missing_columns = [col for col in ("question", "ground_truth") if col not in question_df.columns]
    if missing_columns:
        raise ValueError(f"qs.csv 缺少必需列：{missing_columns}，实际列：{list(question_df.columns)}")

    # 4. 过滤掉问题为空的行，并重置索引
    question_df = question_df.dropna(subset=["question"]).reset_index(drop=True)
    return question_df


def step_2_build_query_graph():
    """步骤2：构建 RAG 查询流程图（流程入口：processor/query_process/main_graph.py 的 create_query_graph）"""
    return create_query_graph()


def step_3_invoke_graph_for_answer(graph, question: str) -> dict:
    """步骤3：调用 RAG 查询流程，返回执行完成后的最终 state

    每个问题使用独立的 session_id / task_id（与 services.QueryService.run_query_graph 保持一致），
    避免历史对话相互污染；is_stream=False 时答案直接写入 state["answer"]。

    Args:
        graph: 步骤2 构建的已编译 LangGraph 图
        question: 待评估的问题

    Returns:
        图执行完成后的最终 state，包含 answer、reranked_docs 等字段
    """
    # 1. 构建默认状态（深拷贝，避免修改全局默认值）
    state = get_default_state()

    # 2. 填充调用参数：独立的会话与任务 ID，防止历史记录串扰
    state["session_id"] = f"eval_{uuid.uuid4().hex[:12]}"
    state["task_id"] = f"eval_{uuid.uuid4().hex[:12]}"
    state["original_query"] = question
    state["is_stream"] = False

    # 3. 执行整条 RAG 查询流程并返回最终状态
    return graph.invoke(state)


def step_4_extract_context_from_state(state: dict) -> List[str]:
    """步骤4：从 state 中提取检索上下文（作为 ragas 的 context）

    说明：state["history"] 保存的是历史对话记录（user/assistant 文本），并非检索到的知识片段；
    ragas 的 Faithfulness / ContextPrecision / ContextRecall 依赖的是“检索上下文”，
    而真正参与答案生成的是 reranked_docs（重排序后的切片，字段结构与
    answer_output_node._format_context 使用的一致），因此优先从该字段提取；
    若为空则回退到 rrf_chunks（RRF 融合后的切片）。
    如需改用其他字段，只需修改本函数。

    Args:
        state: 步骤3 中图执行完成后的最终 state

    Returns:
        上下文文本列表（每个元素是一段检索到的知识切片 content）
    """
    # 1. 依次尝试从 reranked_docs、rrf_chunks 提取切片内容
    for field_name in ("reranked_docs", "rrf_chunks"):
        docs = state.get(field_name) or []
        contexts = [
            doc.get("content", "")
            for doc in docs
            if isinstance(doc, dict) and doc.get("content")
        ]
        # 2. 只要提取到非空上下文就返回
        if contexts:
            return contexts
    return []


def step_5_build_ragas_llm() -> LangchainLLMWrapper:
    """步骤5：构建 ragas 评估使用的 LLM（来源：utils/client/ai_clients.py 的 AIClients）"""
    # 1. 获取项目统一的 LLM 客户端（文本模式，不带 JSON 输出约束）
    llm_client = AIClients.get_llm_client(response_format=False)

    # 2. 关闭流式标记：ragas 内部走 ainvoke 异步调用，流式标记可能影响其工作稳定性
    #    （该客户端是进程内单例，评估脚本独立运行，不影响其他进程使用）
    llm_client.streaming = False

    # 3. 包装为 ragas 认识的 LLM 接口
    return LangchainLLMWrapper(llm_client)


def step_6_build_ragas_embeddings() -> "BgeM3RagasEmbeddings":
    """步骤6：构建 ragas 评估使用的嵌入模型

    BGE-M3 客户端来自 utils/client/ai_clients.py，
    向量生成复用 utils/embedding_util.py 的 generate_bge_m3_hybrid_vectors。
    """
    # 1. 获取项目统一的 BGE-M3 嵌入客户端
    bge_m3_client = AIClients.get_bge_m3_client()

    # 2. 包装为 ragas 认识的嵌入接口
    return BgeM3RagasEmbeddings(bge_m3_client)


def step_7_create_ragas_metrics(ragas_llm: LangchainLLMWrapper,
                                ragas_embeddings: BaseRagasEmbeddings) -> List:
    """步骤7：创建 ragas 五项评估指标

    Args:
        ragas_llm: 步骤5 构建的评估用 LLM
        ragas_embeddings: 步骤6 构建的评估用嵌入模型

    Returns:
        五项指标实例列表：
        - Faithfulness: 忠实度（答案是否忠实于检索上下文）
        - AnswerRelevancy: 答案相关性（答案是否切题，需要嵌入模型）
        - ContextPrecision: 上下文精确率（检索上下文与问题的相关程度）
        - ContextRecall: 上下文召回率（参考答案的关键信息是否被检索上下文覆盖）
        - AnswerCorrectness: 答案正确性（答案与参考答案的一致性）
    """
    return [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]


def step_8_evaluate_one_question(metrics: List,
                                 question: str,
                                 contexts: List[str],
                                 answer: str,
                                 ground_truth: str) -> Dict[str, Optional[float]]:
    """步骤8：对单条问答执行 ragas 五项指标评估

    Args:
        metrics: 步骤7 创建的五项指标实例列表
        question: 问题
        contexts: 步骤4 提取的检索上下文列表
        answer: RAG 流程生成的答案
        ground_truth: 参考答案

    Returns:
        指标名 → 分数 的字典（评估失败或无法评估时对应值为 None）
    """
    # 1. 初始化分数（默认 None，表示未评估）
    scores: Dict[str, Optional[float]] = {metric.name: None for metric in metrics}

    # 2. 答案或上下文为空时无法计算指标（如 Faithfulness 依赖上下文），直接返回
    if not answer:
        print("    警告：答案为空，跳过 ragas 评估")
        return scores
    if not contexts:
        print("    警告：检索上下文为空，跳过 ragas 评估")
        return scores

    # 3. 构建单样本数据集（ragas 字段：user_input=问题、retrieved_contexts=上下文、
    #    response=答案、reference=参考答案）
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts,
        response=answer,
        reference=ground_truth,
    )
    dataset = EvaluationDataset(samples=[sample])

    # 4. 执行评估并解析分数（单样本评估，结果只有一行）
    try:
        result = evaluate(dataset=dataset, metrics=metrics, show_progress=False)
        score_row = result.to_pandas().iloc[0]
        for metric in metrics:
            metric_name = metric.name
            if metric_name in score_row.index and pd.notna(score_row[metric_name]):
                scores[metric_name] = round(float(score_row[metric_name]), 4)
    except Exception as exc:
        print(f"    警告：ragas 评估失败：{exc}")

    return scores


def step_9_save_result_csv(result_rows: List[Dict[str, object]], output_path: Path) -> None:
    """步骤9：将评估结果保存为 CSV（UTF-8 BOM 编码，共 9 列）

    Args:
        result_rows: 每行是一个字典，键为 RESULT_COLUMNS 中定义的 9 个列名
        output_path: 输出文件路径（eval/qa_result.csv）
    """
    # 1. 按固定的 9 列顺序构造 DataFrame
    result_df = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)

    # 2. 写入 CSV：encoding="utf-8-sig" 会在文件头写入 BOM；空分数（None）写为空字符串
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep="")
    print(f"评估结果已写入：{output_path}（{len(result_rows)} 行，编码 UTF-8 BOM）")


def main() -> None:
    setup_logging(logging.INFO)
    question_df = step_1_read_questions()
    total = len(question_df)
    print(f"共读取 {total} 条评估问题")

    graph = step_2_build_query_graph()
    ragas_llm = step_5_build_ragas_llm()
    ragas_embeddings = step_6_build_ragas_embeddings()
    metrics = step_7_create_ragas_metrics(ragas_llm, ragas_embeddings)

    samples = []
    result_rows = []

    # 1. 先批量跑 RAG 流程，收集样本
    for index, row in question_df.iterrows():
        question = str(row["question"]).strip()
        ground_truth = str(row["ground_truth"]).strip() if pd.notna(row["ground_truth"]) else ""
        print(f"\n[{index + 1}/{total}] 正在执行 RAG：{question}")

        try:
            state = step_3_invoke_graph_for_answer(graph, question)
            contexts = step_4_extract_context_from_state(state)
            answer = str(state.get("answer") or "").strip()
        except Exception as exc:
            print(f"    错误：RAG 流程失败：{exc}")
            contexts, answer = [], ""

        # 构建 RAGAS 样本
        if answer and contexts:
            samples.append(
                SingleTurnSample(
                    user_input=question,
                    retrieved_contexts=contexts,
                    response=answer,
                    reference=ground_truth,
                )
            )
        else:
            print("    警告：答案或上下文为空，跳过 RAGAS 评估")

        # 占位，稍后填入分数
        result_rows.append({
            "question": question,
            "context": "\n\n".join(contexts),
            "answer": answer,
            "ground_trush": ground_truth,
            **{metric.name: None for metric in metrics}
        })

    # 2. 一次性批量评估所有样本（核心提速点！）
    if samples:
        print(f"\n开始批量 RAGAS 评估，共 {len(samples)} 条样本...")
        dataset = EvaluationDataset(samples=samples)

        # 关键：引入 RunConfig 开启并发（max_workers=4 表示同时跑4个指标任务）
        from ragas import RunConfig
        run_config = RunConfig(max_workers=4, timeout=120)

        try:
            result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config, show_progress=True)
            score_df = result.to_pandas()

            # 把分数回填到 result_rows
            for i, row in score_df.iterrows():
                # 找到对应的 result_rows（按问题匹配）
                for r in result_rows:
                    if r["question"] == row["user_input"]:
                        for metric in metrics:
                            if metric.name in row and pd.notna(row[metric.name]):
                                r[metric.name] = round(float(row[metric.name]), 4)
                        break
        except Exception as exc:
            print(f"批量 RAGAS 评估失败：{exc}")

    # 3. 打印并保存
    print("\n===== 单条问题分数 =====")
    for r in result_rows:
        score_text = "，".join(
            f"{METRIC_COLUMN_MAP.get(name, name)}={value:.4f}"
            if isinstance(value, float) else f"{METRIC_COLUMN_MAP.get(name, name)}=N/A"
            for name, value in r.items() if name in METRIC_COLUMN_MAP.values()
        )
        print(f"{r['question']}: {score_text}")

    step_9_save_result_csv(result_rows, OUTPUT_CSV_PATH)

    print("\n===== 平均分汇总 =====")
    for metric in metrics:
        column = METRIC_COLUMN_MAP.get(metric.name, metric.name)
        values = [r[column] for r in result_rows if isinstance(r.get(column), float)]
        print(
            f"{column}：{sum(values) / len(values):.4f}（有效 {len(values)}/{total} 条）" if values else f"{column}：无有效分数")
    print(f"\n评估完成，结果已保存至：{OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
