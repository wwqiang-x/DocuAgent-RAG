import os
import json
import pandas as pd
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# 清代理
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

EVAL_DIR = Path(__file__).parent
INPUT_CSV = EVAL_DIR / "qa_result.csv"   # 复用已有的 RAG 结果
OUTPUT_CSV = EVAL_DIR / "qa_simple_result.csv"

JUDGE_PROMPT = """你是技术文档问答评估专家。请对下面的问答对逐项打分。

【问题】{question}
【标准答案】{ground_truth}
【RAG生成的答案】{answer}
【检索到的上下文】
{context}

请严格按下面 JSON 格式输出，不要任何额外文字：
{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "context_recall": 0.0,
  "answer_correctness": 0.0
}}

评分标准：
- faithfulness：答案是否完全基于【上下文】，无编造=1.0，有幻觉=0.0~0.5
- answer_relevancy：答案是否切题，答非所问=0.0
- context_recall：上下文是否覆盖了【标准答案】的所有关键信息，全覆盖=1.0
- answer_correctness：答案与【标准答案】的语义一致程度，完全一致=1.0
"""

def evaluate_row(row):
    q = str(row["question"])
    gt = str(row["ground_trush"])
    ans = str(row.get("answer", "") or "")
    ctx = str(row.get("context", "") or "")[:3000]

    if not ans.strip():
        return {"faithfulness": 0, "answer_relevancy": 0, "context_recall": 0, "answer_correctness": 0}

    prompt = JUDGE_PROMPT.format(question=q, ground_truth=gt, answer=ans, context=ctx)
    try:
        resp = client.chat.completions.create(
            model="qwen-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  打分失败：{e}")
        return {"faithfulness": None, "answer_relevancy": None, "context_recall": None, "answer_correctness": None}

def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    # 只取有 answer 和 context 的行
    df = df[df["answer"].notna() & df["context"].notna()].reset_index(drop=True)
    print(f"共 {len(df)} 条有效数据，开始打分...")

    scores = {"faithfulness": [], "answer_relevancy": [], "context_recall": [], "answer_correctness": []}
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {str(row['question'])[:40]}...")
        r = evaluate_row(row)
        for k in scores:
            scores[k].append(r.get(k))
        df.at[i, "Faithfulness"] = r.get("faithfulness")
        df.at[i, "Answer Relevancy"] = r.get("answer_relevancy")
        df.at[i, "Context Recall"] = r.get("context_recall")
        df.at[i, "Answer Correctness"] = r.get("answer_correctness")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\n===== 平均分 =====")
    for k, v in scores.items():
        valid = [x for x in v if isinstance(x, (int, float))]
        if valid:
            print(f"{k}: {sum(valid)/len(valid):.4f}（有效 {len(valid)}/{len(df)} 条）")
        else:
            print(f"{k}: 无有效分数")
    print(f"\n结果已保存至：{OUTPUT_CSV}")

if __name__ == "__main__":
    main()