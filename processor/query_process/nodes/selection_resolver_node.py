"""
Selection Resolver Node
作用：判断用户当前回复是否是在回答助手的候选列表。
     纯 LLM 方案，把完整历史交给 LLM，由 LLM 推断用户的真实意图。

三条路径：
    A. 命中候选     → 锁定商品名 + 改写查询，进入下游 RAG 检索
    B. 否定候选     → 输出引导语，让用户重新描述
    C. 非候选选择   → 交回 item_name_confirmed_node 走正常提取

设计要点：
    - limit=10，覆盖多轮纠错场景
    - 严格校验：最近一条 assistant 消息必须是候选列表
    - 把完整历史交给 LLM，让 LLM 自己找到"用户真正想问的问题"
    - 安全校验：LLM 返回的商品名必须严格在候选列表里
"""
import json
import re
from typing import List, Dict, Tuple

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from processor.query_process.base import BaseNode
from utils.client.ai_clients import AIClients
from utils.mongo_history_util import get_recent_messages


# ==================== LLM 结构化输出 ====================
class SelectionDecision(BaseModel):
    """LLM 对用户回复意图的判断结果"""
    decision: str = Field(
        description=(
            "只能是以下三种值之一："
            "(1) 候选列表中的某个商品名（必须严格等于候选列表里的原文）；"
            "(2) 'UNKNOWN'，表示用户明确否定了所有候选；"
            "(3) 'NOT_SELECTION'，表示用户不是在回答候选，而是在提新问题。"
        )
    )
    rewritten_query: str = Field(
        description=(
            "改写后的完整问题。规则："
            "如果 decision 是具体商品名，结合完整历史找到用户真正想问的原始问题，"
            "把指代词替换成选中的商品名，生成一个语义完整、可直接用于检索的独立问题；"
            "如果 decision 是 UNKNOWN 或 NOT_SELECTION，原样返回用户最新回复。"
        )
    )
    reason: str = Field(description="判断理由，一句话")


# ==================== 提示词 ====================
SYSTEM_PROMPT = """你是多轮对话中的"候选选择解析专家"。

背景：助手刚刚向用户提供了一个候选商品列表，请结合完整历史，判断用户当前的回复意图。

## 判断规则
1. 如果用户明确选择了某个候选（"第一个"、"就它"、"第二个"、"RS-12那个"、"不是第一个，是第二个"），
   返回该候选的**原文**（必须严格等于候选列表里的某一项）。
2. 如果用户明确否定所有候选（"都不是"、"不对"、"没有我要的"），返回 "UNKNOWN"。
3. 如果用户不是在回答候选，而是在提新问题或提新商品（"顺便问一下 XX"、"我想问 L420x"），
   返回 "NOT_SELECTION"。
4. 如果无法判断，返回 "NOT_SELECTION"（保守策略，交给下游 LLM 重新提取）。

## 关键提示
- 用户可能经历过多次候选列表反问，请找到用户**最初想咨询的商品和问题**。
- 如果用户在之前的轮次里纠正过商品名，以用户**最新确认**的商品名为准。

## 注意
- 返回的商品名**必须**严格等于候选列表里的某一项，不能自己编造或改写。
- 只输出 JSON，不要任何额外文字。
"""

USER_PROMPT = """【完整历史对话】（按时间顺序，最后一条是用户的最新回复）
{history_text}

【当前候选列表】（由助手最新一轮提供）
{candidates}

【助手的提问】
{assistant_message}

【用户最新回复】
{user_reply}

## 你的任务
请结合**完整历史对话**，判断用户当前的回复意图：
1. 用户是不是在回答【当前候选列表】？
2. 如果是，用户选中的是哪一个候选？
3. 用户**真正想问的问题**是什么（可能是几轮之前提出的）？请把它改写成一个语义完整、可直接用于检索的独立问题。

## 输出要求
只输出 JSON，不要任何额外文字。
{{
  "decision": "商品名 或 UNKNOWN 或 NOT_SELECTION",
  "rewritten_query": "改写后的问题（完整、独立、可用于检索）",
  "reason": "判断理由"
}}

## 改写示例
示例1（简单场景）：
- 历史：user:"数字万用表怎么测电阻" → assistant:"请问你是在询问以下内容吗[RS-12, RS-05]" → user:"第一个"
- 输出：decision="RS-12数字万用表", rewritten_query="RS-12数字万用表怎么测电阻"

示例2（多轮纠错场景）：
- 历史：
  user:"RS-12数字万用表怎么测量电阻"
  assistant:"RS-12数字万用表测量电阻的方法..."
  user:"那RS-05呢"
  assistant:"请问你是在询问以下内容吗[RS-05数字万用表, RS-12数字万用表]"
  user:"不是刚才打错了，是RS-10"
  assistant:"请问你是在询问以下内容吗[RS-10数字万用表, RS-12数字万用表]"
  user:"对，是第一个"
- 输出：decision="RS-10数字万用表", rewritten_query="RS-10数字万用表怎么测量电阻"

示例3（用户否定）：
- 历史：user:"数字万用表怎么测电阻" → assistant:"请问你是在询问以下内容吗[RS-12, RS-05]" → user:"都不是"
- 输出：decision="UNKNOWN", rewritten_query="数字万用表怎么测电阻"

示例4（用户提新问题）：
- 历史：user:"数字万用表怎么测电阻" → assistant:"请问你是在询问以下内容吗[RS-12, RS-05]" → user:"我想问华为擎云L420x怎么用"
- 输出：decision="NOT_SELECTION", rewritten_query="华为擎云L420x怎么用"
"""


class SelectionResolverNode(BaseNode):
    name = "selection_resolver_node"
    CANDIDATE_PROMPT_MARKER = "请问你是在询问以下内容吗"

    def process(self, state) -> dict:
        original_query = state.get("original_query", "")
        session_id = state.get("session_id", "")

        # 1. 取最近 10 条（约 5 轮对话），保证能覆盖多轮纠错场景
        history = get_recent_messages(session_id=session_id, limit=10)
        state["history"] = history

        # 2. 严格校验：最近一条 assistant 消息必须是候选列表
        if not self._is_last_assistant_candidate_list(history):
            state["selection_resolved"] = False
            self.logger.info("[SelectionResolver] 上一轮不是候选列表，跳过")
            return state

        # 3. 提取候选列表
        last_assistant = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
        assistant_text = last_assistant.get("text", "") if last_assistant else ""
        candidates = self._extract_candidates(assistant_text)
        if not candidates:
            state["selection_resolved"] = False
            self.logger.info("[SelectionResolver] 候选列表为空，跳过")
            return state

        # 4. 调用 LLM 判断
        decision, rewritten_query = self._call_llm(
            original_query, assistant_text, candidates, history
        )

        # 5. 更新 state
        self._update_state(state, decision, rewritten_query)
        return state

    # ==================== LLM 调用 ====================
    def _call_llm(
            self,
            query: str,
            assistant_text: str,
            candidates: List[str],
            history: List[Dict],
    ) -> Tuple[str, str]:
        """LLM 意图识别，返回 (decision, rewritten_query)"""
        try:
            # 1. 用 response_format=False 的 client（不带 json mode 强制约束）
            llm_client = AIClients.get_llm_client(response_format=False)

            history_text = self._format_history(history)

            user_prompt = USER_PROMPT.format(
                history_text=history_text,
                candidates="\n".join(f"- {c}" for c in candidates),
                assistant_message=assistant_text,
                user_reply=query,
            )

            # 2. 直接用 invoke，不用 with_structured_output
            response = llm_client.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])

            raw_content = response.content.strip()
            print(f"[SelectionResolver-LLM-Raw] {raw_content[:300]}")

            # 3. 清理可能出现的 markdown 代码围栏
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
                raw_content = re.sub(r"\s*```$", "", raw_content)

            # 4. 手动解析 JSON
            result = json.loads(raw_content)
            decision = str(result.get("decision", "")).strip()
            rewritten_query = str(result.get("rewritten_query", "")).strip() or query
            reason = str(result.get("reason", "")).strip()

            print(f"[SelectionResolver-LLM] decision={decision}, rewritten_query={rewritten_query}, reason={reason}")

            # 5. 安全校验：非候选名做一次序号映射
            if decision not in candidates and decision not in ("UNKNOWN", "NOT_SELECTION"):
                q_clean = decision.lower().replace(" ", "")
                if "一" in q_clean or q_clean == "1":
                    decision = candidates[0]
                elif "二" in q_clean or q_clean == "2":
                    decision = candidates[1] if len(candidates) > 1 else candidates[0]
                elif "最后" in q_clean:
                    decision = candidates[-1]
                else:
                    self.logger.warning(f"LLM 返回非法商品名：{decision}，降级为 NOT_SELECTION")
                    return "NOT_SELECTION", rewritten_query

            self.logger.info(f"[SelectionResolver] 判定：{decision}（{reason}）")
            return decision, rewritten_query

        except Exception as e:
            self.logger.error(f"[SelectionResolver] LLM 调用失败：{e}")
            return "NOT_SELECTION", query

    # ==================== 更新 state ====================
    def _update_state(self, state, decision: str, rewritten_query: str):
        if decision == "UNKNOWN":
            # 场景 B：用户否定所有候选 → 引导用户重新描述
            state["item_names"] = []
            state["rewritten_query"] = rewritten_query
            state["answer"] = (
                "您提到的商品不在候选列表中，"
                "能否提供更具体的型号？例如 RS-12、HAK 180 等。"
            )
            state["selection_resolved"] = True
            self.logger.info("[SelectionResolver] 用户否定候选，引导重新描述")

        elif decision == "NOT_SELECTION":
            # 场景 C：非候选选择 → 交回下游正常提取
            state["selection_resolved"] = False
            self.logger.info("[SelectionResolver] 非候选选择，交回下游 LLM 提取")

        else:
            # 场景 A：命中候选 → 锁定商品名 + 改写查询
            state["item_names"] = [decision]
            state["rewritten_query"] = rewritten_query
            state["selection_resolved"] = True
            self.logger.info(
                f"[SelectionResolver] 命中：{decision}，改写查询：{rewritten_query}"
            )

    # ==================== 辅助方法 ====================
    def _is_last_assistant_candidate_list(self, history: List[Dict]) -> bool:
        """严格校验：最近一条 assistant 消息必须是候选列表"""
        if not history:
            return False
        last_assistant = next(
            (m for m in reversed(history) if m.get("role") == "assistant"),
            None
        )
        if not last_assistant:
            return False
        return self.CANDIDATE_PROMPT_MARKER in last_assistant.get("text", "")

    def _extract_candidates(self, text: str) -> List[str]:
        """从助手文本中提取 [...] 里的候选列表"""
        match = re.search(r"\[([^\]]+)\]", text)
        if not match:
            return []
        return [c.strip() for c in match.group(1).split(",") if c.strip()]

    def _format_history(self, history: List[Dict]) -> str:
        """把历史格式化成可读文本"""
        if not history:
            return "（无）"
        lines = []
        for msg in history:
            role = "用户" if msg.get("role") == "user" else "助手"
            text = msg.get("text", "")
            lines.append(f"[{role}] {text}")
        return "\n".join(lines)