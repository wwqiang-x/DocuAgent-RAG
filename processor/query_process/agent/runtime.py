"""Agent Runtime。

Runtime 负责串起 Controller、工具、证据管线和答案生成。
Runtime 支持执行步骤回调和答案增量回调。
Runtime 本身仍不写 Mongo，历史保存由 AgentQueryService 负责。
"""

import importlib
from collections.abc import Callable

from processor.query_process.agent.answer_service import AgentAnswerService
from processor.query_process.agent.controller import AgentController
from processor.query_process.agent.decisions import (
    AgentAction,
    AgentDecision,
    AgentStep,
    StopReason,
)
from processor.query_process.agent.quality_gate import (
    EvidenceQualityGate,
    LLMEvidenceAssessor,
)
from processor.query_process.agent.state import AgentState
from processor.query_process.evidence.pipeline import EvidencePipeline
from processor.query_process.state import get_default_state
from processor.query_process.tools.context import ToolContext
from processor.query_process.tools.contracts import (
    ProductResolutionStatus,
    SearchStrategy,
    ToolStatus,
)
from processor.query_process.tools.local_search_tool import (
    LocalSearchRequest,
)
from processor.query_process.tools.registry import ToolRegistry
from processor.query_process.tools.web_search_tool import WebSearchRequest


class AgentRuntime:
    """单 Agent 的离线运行入口。"""

    def __init__(
        self,
        controller: AgentController | None = None,
        quality_gate: EvidenceQualityGate | None = None,
        answer_service: AgentAnswerService | None = None,
        tool_registry: ToolRegistry | None = None,
        evidence_pipeline: EvidencePipeline | None = None,
        enable_selection_preflight: bool = True,
        event_callback: Callable[[AgentStep], None] | None = None,
        delta_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.controller = controller or AgentController()
        self.quality_gate = quality_gate or EvidenceQualityGate(
            assessor=LLMEvidenceAssessor()
        )
        self.answer_service = answer_service or AgentAnswerService()
        self.tool_registry = tool_registry or ToolRegistry()
        self.evidence_pipeline = evidence_pipeline or EvidencePipeline()
        self.enable_selection_preflight = enable_selection_preflight

        # 阶段 C 新增：每一步执行后通知外部，例如推送到 SSE 队列。
        self.event_callback = event_callback

        # 流式答案生成时，每收到一段文本就通知外部。
        self.delta_callback = delta_callback

    def run(
        self,
        query: str,
        session_id: str = "",
        task_id: str = "",
        max_steps: int = 6,
    ) -> AgentState:
        """执行一次完整的 Agent 对话。"""

        state = AgentState(
            task_id=task_id,
            session_id=session_id,
            original_query=query,
            max_steps=max_steps,
        )

        # 空问题直接要求用户补充，不进入检索。
        if not query.strip():
            state.clarification = "请提供需要咨询的问题。"
            state.stop_reason = StopReason.ASKED_USER
            return state

        # 先复用现有 SelectionResolver，处理上一轮候选商品回复。
        if self.enable_selection_preflight:
            self._run_selection_preflight(state)

        # 用户否定候选时已经终止，不再进入检索。
        if state.stop_reason is not None:
            return state

        while state.current_step < state.max_steps:
            # Controller 只决定下一步，不直接执行工具。
            decision = self.controller.decide(state)

            if decision.action == AgentAction.ANSWER:
                self._record_terminal_step(
                    state,
                    decision.action,
                    decision.reason,
                )
                state.answer = self.answer_service.generate(
                    state,
                    delta_callback=self.delta_callback,
                )
                state.stop_reason = StopReason.ANSWERED
                break

            if decision.action == AgentAction.ASK_USER:
                self._record_terminal_step(
                    state,
                    decision.action,
                    decision.reason,
                )
                state.clarification = (
                    decision.clarification
                    or state.clarification
                    or "请补充具体商品型号或问题信息。"
                )
                state.stop_reason = StopReason.ASKED_USER
                break

            if decision.action == AgentAction.CANNOT_ANSWER:
                self._record_terminal_step(
                    state,
                    decision.action,
                    decision.reason,
                )
                state.answer = "根据现有资料无法回答"
                state.stop_reason = StopReason.CANNOT_ANSWER
                break

            # 执行普通工具动作。
            self._execute_decision(state, decision)

        # 循环结束但没有终止原因时，说明达到最大步数。
        if state.stop_reason is None:
            state.answer = "根据现有资料无法回答"
            state.stop_reason = StopReason.MAX_STEPS

        return state

    def _run_selection_preflight(self, state: AgentState) -> None:
        """复用现有 SelectionResolver 处理多轮候选。

        当前阶段分支可能从 main 创建，未必包含 SelectionResolverNode。
        因此这里使用惰性导入：存在则复用，不存在则跳过。
        """

        try:
            module = importlib.import_module(
                "processor.query_process.nodes.selection_resolver_node"
            )
            resolver_class = getattr(module, "SelectionResolverNode")
        except (ImportError, AttributeError):
            # 当前分支没有该节点时保持基础检索流程，不阻断 Agent 运行。
            state.selection_resolved = False
            return

        legacy_state = get_default_state()
        legacy_state["session_id"] = state.session_id
        legacy_state["task_id"] = state.task_id
        legacy_state["original_query"] = state.original_query
        legacy_state["is_stream"] = False

        # 直接调用 process，跳过进度记录和 SSE。
        legacy_state = resolver_class().process(legacy_state)

        state.history = legacy_state.get("history") or []
        state.selection_resolved = bool(
            legacy_state.get("selection_resolved")
        )
        state.item_names = list(legacy_state.get("item_names") or [])
        state.rewritten_query = str(
            legacy_state.get("rewritten_query") or ""
        )

        # SelectionResolver 否定候选时会写入 answer。
        answer = str(legacy_state.get("answer") or "").strip()
        if answer:
            state.clarification = answer
            state.stop_reason = StopReason.ASKED_USER

        # 已从上一轮候选中锁定商品，无需再次识别。
        if state.item_names:
            state.product_resolved = True

    def _execute_decision(
        self,
        state: AgentState,
        decision: AgentDecision,
    ) -> None:
        """执行 Controller 返回的非终止动作。"""

        if decision.action == AgentAction.RESOLVE_PRODUCT:
            self._execute_resolve_product(state, decision)
            return

        if decision.action == AgentAction.SEARCH_LOCAL:
            self._execute_local_search(state, decision)
            return

        if decision.action == AgentAction.SEARCH_WEB:
            self._execute_web_search(state, decision)
            return

        raise ValueError(f"不支持的动作: {decision.action}")

    def _execute_resolve_product(
        self,
        state: AgentState,
        decision: AgentDecision,
    ) -> None:
        """执行商品识别工具。"""

        tool = self.tool_registry.get("resolve_product")
        result = tool.run(
            ToolContext(
                task_id=state.task_id,
                session_id=state.session_id,
                original_query=state.original_query,
            )
        )

        state.mark_action_used(AgentAction.RESOLVE_PRODUCT.value)
        state.product_resolved = True

        if result.status == ToolStatus.ERROR:
            state.last_error = result.error
        elif result.output is not None:
            state.item_names = result.output.item_names
            state.candidates = result.output.candidates
            state.rewritten_query = (
                result.output.rewritten_query
                or state.rewritten_query
                or state.original_query
            )
            state.clarification = result.output.clarification

            if result.output.resolution == ProductResolutionStatus.CONFIRMED:
                state.product_resolved = True

        self._append_step(
            state=state,
            action=decision.action,
            reason=decision.reason,
            tool_name=result.tool_name,
            tool_status=result.status.value,
            duration_ms=result.duration_ms,
            error=result.error,
        )

    def _execute_local_search(
        self,
        state: AgentState,
        decision: AgentDecision,
    ) -> None:
        """执行基础检索或 HyDE 检索。"""

        strategy = decision.strategy or SearchStrategy.BASIC
        tool = self.tool_registry.get("search_local")

        result = tool.run(
            ToolContext(
                task_id=state.task_id,
                session_id=state.session_id,
                original_query=state.original_query,
                rewritten_query=decision.query or state.rewritten_query,
                item_names=decision.item_names or state.item_names,
            ),
            LocalSearchRequest(
                query=decision.query or state.rewritten_query,
                item_names=decision.item_names or state.item_names,
                strategy=strategy,
            ),
        )

        state.mark_action_used(
            f"{AgentAction.SEARCH_LOCAL.value}:{strategy.value}"
        )

        if result.status == ToolStatus.ERROR:
            state.last_error = result.error
        elif result.output is not None:
            if strategy == SearchStrategy.HYDE:
                state.hyde_search = result.output
            else:
                state.basic_search = result.output

        self._append_step(
            state=state,
            action=decision.action,
            reason=decision.reason,
            query=decision.query or state.rewritten_query,
            strategy=strategy.value,
            tool_name=result.tool_name,
            tool_status=result.status.value,
            evidence_count=(
                len(result.output.evidences)
                if result.output is not None
                else 0
            ),
            duration_ms=result.duration_ms,
            error=result.error,
        )

        self._refresh_evidence(state)

    def _execute_web_search(
        self,
        state: AgentState,
        decision: AgentDecision,
    ) -> None:
        """执行网络搜索。"""

        tool = self.tool_registry.get("search_web")
        result = tool.run(
            ToolContext(
                task_id=state.task_id,
                session_id=state.session_id,
                original_query=state.original_query,
                rewritten_query=decision.query or state.rewritten_query,
                item_names=state.item_names,
            ),
            WebSearchRequest(query=decision.query or state.rewritten_query),
        )

        state.mark_action_used(AgentAction.SEARCH_WEB.value)

        if result.status == ToolStatus.ERROR:
            state.last_error = result.error
        elif result.output is not None:
            state.web_search = result.output

        self._append_step(
            state=state,
            action=decision.action,
            reason=decision.reason,
            query=decision.query or state.rewritten_query,
            tool_name=result.tool_name,
            tool_status=result.status.value,
            evidence_count=(
                len(result.output.evidences)
                if result.output is not None
                else 0
            ),
            duration_ms=result.duration_ms,
            error=result.error,
        )

        self._refresh_evidence(state)

    def _refresh_evidence(self, state: AgentState) -> None:
        """执行 RRF、Reranker 和证据质量评估。"""

        # 融合基础检索和 HyDE 两个本地分支。
        if state.basic_search is not None or state.hyde_search is not None:
            state.fused_evidence = self.evidence_pipeline.fuse_local(
                basic=state.basic_search,
                hyde=state.hyde_search,
            )
        else:
            state.fused_evidence = []

        # 网络证据单独取出，随后和本地证据一起重排序。
        web_evidences = (
            state.web_search.evidences
            if state.web_search is not None
            else []
        )

        state.reranked_evidence = self.evidence_pipeline.rerank(
            query=state.rewritten_query or state.original_query,
            local_evidence=state.fused_evidence,
            web_evidence=web_evidences,
        )

        # 每次检索后都重新评估证据质量。
        state.assessment = self.quality_gate.assess(state)

    def _append_step(
        self,
        state: AgentState,
        action: AgentAction,
        reason: str,
        query: str = "",
        strategy: str | None = None,
        tool_name: str | None = None,
        tool_status: str | None = None,
        evidence_count: int = 0,
        duration_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        """记录一次工具或决策执行步骤。"""

        state.current_step += 1
        step = AgentStep(
            step_no=state.current_step,
            action=action,
            reason=reason,
            query=query,
            strategy=strategy,
            tool_name=tool_name,
            tool_status=tool_status,
            evidence_count=evidence_count,
            duration_ms=duration_ms,
            error=error,
        )
        state.steps.append(step)

        # 外部回调异常不能影响 Agent 主流程。
        if self.event_callback is not None:
            try:
                self.event_callback(step)
            except Exception:
                pass

    def _record_terminal_step(
        self,
        state: AgentState,
        action: AgentAction,
        reason: str,
    ) -> None:
        """记录 answer、ask_user 或 cannot_answer 终止动作。"""

        self._append_step(
            state=state,
            action=action,
            reason=reason,
        )
