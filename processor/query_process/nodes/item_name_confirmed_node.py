import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from processor.query_process.config import get_config
from pymilvus import AnnSearchRequest
from processor.query_process.base import BaseNode,T
from processor.query_process.state import QueryGraphState, get_default_state
from prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors
from utils.milvus_util import execute_hybrid_search_query, create_hybrid_search_requests
from utils.mongo_history_util import get_recent_messages


class ItemNameConfirmedNode(BaseNode):

    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self.extractor = _ItemNameExtractor()
        self.aligner = _ItemNameAligner()

    def process(self,state:QueryGraphState)->QueryGraphState:
        #1.验证参数
        original_query = state.get("original_query")
        #2.根据session_id查询历史记录（MongoDB）
        session_id = state.get("session_id")
        history = get_recent_messages(
            session_id=session_id,
            limit = 8,
        )
        state["history"] = history
        #格式化历史记录
        nick_name = {
            "user":"用户",
            "assistant":"助手"
        }
        formatted_lines = []
        for message in history:
            role = message.get("role")
            text = message.get("text")
            formatted_line = f"{nick_name.get(role)}:{text}"
            formatted_lines.append(formatted_line)
        history_context = "\n".join(formatted_lines)
        #3.原始查询与历史上下文封装提示词，调用llm生成商品名和更新问题
        item_names, rewritten_query = self.extractor.extract_item_name(original_query, history_context)
        print(rewritten_query)
        print(item_names)
        #4.将llm生成的商品名与数据库中的对齐
        if item_names:
            confirmed, options = self.aligner.search_and_align(item_names)
        else:
            confirmed = []
            options = []

        #5.决策处理
        #  confirmed直接进入喜爱一个节点
        #  options 询问用户 等待用户确认
        #  都没有设置state的answer
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            # 如果只有一个候选，直接确认，无需询问用户
            if len(options) == 1:
                state["item_names"] = options
                state["rewritten_query"] = rewritten_query
            else:
                state["answer"] = f"我不确定您指的是什么，请问你是在询问以下内容吗:\n[{', '.join(options)}]"
        else:
            state["item_names"] = []
            state["rewritten_query"] = rewritten_query
        return state

class _ItemNameExtractor:
    #商品名提取
    def extract_item_name(self,original_query:str,history_context):
        item_names = []
        rewritten_query = ""
        #调用llm 构建提示词
        #1.构造提示词
        system_prompt = "你是一位商品名提取专家，请从用户的问题以及历史对话中提取相关的商品名以及改写原始查询"
        history_text = history_context if history_context else "暂无上下文"
        user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context,
            query=original_query,
        )

        #2.调用大模型
        llm_client = AIClients.get_llm_client()
        llm_response = llm_client.invoke([
            SystemMessage(content = system_prompt),
            HumanMessage(content = user_prompt),
        ])
        llm_result = llm_response.content
        #3.解析结果
        item_names,rewritten_query = self._clean_and_parse_llm_result(llm_result)
        rewritten_query = rewritten_query if rewritten_query.strip() else original_query

        return item_names,rewritten_query


    def _clean_and_parse_llm_result(self,llm_result):
        #清洗数据
        item_names = []
        rewritten_query = ""
        json_obj = json.loads(llm_result)
        raw_item_names = json_obj.get("item_names",[])
        if isinstance(raw_item_names, list):
            item_names = [name.strip() for name in raw_item_names]
        else:
            item_names = []

        raw_rewritten_query = json_obj.get("rewritten_query","")
        if isinstance(raw_rewritten_query, str):
            rewritten_query = raw_rewritten_query.strip()
        else:
            rewritten_query = ""

        return item_names,rewritten_query


class _ItemNameAligner:
    #对齐商品名
    def search_and_align(self,item_names):
        # confirmed 高信
        # options 中信
        #根据llm生产的item_name先进行向量化再进行混合检索
        search_results = self._search_vector(item_names)

        json_str = json.dumps(search_results,ensure_ascii=False,indent=4)
        #得到搜索结果的得分
        #商品名对齐
        #confirmed直接用 options等待用户确认
        confirmed,options = self._item_score_name_align(search_results)
        #得到俩个以上的商品名，分数差异化过滤
        if len(confirmed) > 1:
            confirmed = self._item_score_name_filter(confirmed,search_results)
        return confirmed, options

    def _search_vector(self,item_names):
        search_results = []

        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logging.error(f"milvus连接失败{e}")
            return []
        try:
            bge_m3 = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logging.error(f"bge-m3连接失败{e}")
            return []
        #对所有传过来的item_name向量化
        item_names_vector =  generate_bge_m3_hybrid_vectors(bge_m3, item_names)
        #item_names = ["a","b"]
        # {
        #     "dense":[[...],[...]]
        #     "sparse":[{...}, {...}]
        # }
        #遍历每个商品获取稀疏和稠密向量
        for index,item_name in enumerate(item_names):
            dense_vector = item_names_vector["dense"][index]
            sparse_vector = item_names_vector["sparse"][index]

            #创建多个AnnSearchRequest实例
            requests = create_hybrid_search_requests(dense_vector,sparse_vector)
            #执行混合搜索
            res = execute_hybrid_search_query(
                milvus_client = milvus_client,
                collection_name = get_config().item_name_collection,
                search_requests = requests,
            )
            #处理结果
            if res:
                hybird_hits = res[0]
                current_item_name_result = []
                for hit in hybird_hits:
                    score = hit.distance
                    name = hit.entity.get("item_name")
                    current_item_name_result.append({
                        "item_name": name,
                        "score": score,
                    })
            #保存结果
            search_results.append({
                "extracted_name": item_name,
                "matches":current_item_name_result
            })
        return search_results


    def _item_score_name_align(self,search_results):
        config = get_config()
        confirmed = []
        options = []

        for search_result in search_results:
            extracted_name = search_result.get("extracted_name")
            matches = search_result.get("matches")
            #将matchs中的结果按从高到低排序
            matches_sorted = sorted(matches, key=lambda x: x["score"], reverse=True)
            #获取评分高于高置信阈值的结果
            high = [match for match in matches_sorted if match.get("score") >= config.item_name_high_confidence ]
            if high:
                exact_hit = next((h for h in high if str(h["item_name"]) == extracted_name),None)
                if exact_hit:
                    if exact_hit["item_name"] not in confirmed:
                        confirmed.append(exact_hit["item_name"])
                elif len(high)==1:
                    if high[0]["item_name"] not in confirmed:
                        confirmed.append(high[0]["item_name"])
                else:
                    if high[0]['score'] - high[1]['score'] > config.item_name_score_gap:
                        # 有多个高命中，且第一个score如果远大于第二个，则直接将最高分的作为精确匹配项
                        if high[0]['item_name'] not in confirmed:
                            confirmed.append(high[0]['item_name'])
                    else:
                        for h in high[:config.item_name_max_options]:
                            picked = h.get('item_name')
                            if picked not in options and picked not in confirmed:
                                options.append(picked)

            else:
                mid = [match for match in matches_sorted if match.get("score") >= 0.4 ]
                if mid:
                    for m in mid[:config.item_name_max_options]:
                        options.append(m.get('item_name'))

        return confirmed, options

    def _item_score_name_filter(self,confirmed,search_results):
        # 1. 构建 商品名 → 最高分数 的映射
        # 例如：{"RS-12万用表": 0.95, "数字电压表": 0.88}
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get('matches', [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)
            # 2. 防御性检查：如果没有收集到任何分数，直接返回原始 confirmed
        if not item_name_score:
            return confirmed

            # 3. 取出分数值最大的作为基准
        max_score = max(item_name_score.values())
        final_confirmed = []
        for name in confirmed:
            name_score = item_name_score.get(name,0)
            if max_score - name_score > get_config().item_name_score_gap:
                continue
            else:
                final_confirmed.append(name)
        return final_confirmed


if __name__ == '__main__':
    node = ItemNameConfirmedNode()
    state = get_default_state()
    state["original_query"] = "RS-12数字万用表如何测量电阻"

    state = node.process(state)
    json_str = json.dumps(state,ensure_ascii=False,indent=4)
    print(json_str)

























