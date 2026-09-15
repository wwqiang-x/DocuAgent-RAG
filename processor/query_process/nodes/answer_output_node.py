import json

from processor.query_process.base import BaseNode
from processor.query_process.state import QueryGraphState
from prompts.query_prompt import ANSWER_PROMPT
from utils.client.ai_clients import AIClients
from utils.mongo_history_util import save_chat_message
from utils.sse_util import create_sse_queue, get_sse_queue, push_sse_event, SSEEvent
from utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):

    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        #获取state中的数据
        answer = state.get("answer")
        is_stream = state.get("is_stream",True)
        task_id = state.get("task_id")
        #判断answer
        if answer:
            #直接回答
            self._push_answer(task_id,is_stream,answer)
        else:
            #构造prompt
            prompt:str = self._build_prompt(state)
            self._generate_answer(task_id,is_stream,prompt,state)

        #用户问题（原始）
        #将历史记录保存到mongodb
        self._save_history(state)
        return state


    def _push_answer(self,task_id,is_stream,answer):
        #推送答案到task_util或sse_util
        if is_stream:
            queue = get_sse_queue(task_id)
            if not queue:
                create_sse_queue(task_id)
            push_sse_event(
                task_id = task_id,
                event = SSEEvent.FINAL,
                data = {"answer":answer}
            )
        else:
            set_task_result(
                task_id = task_id,
                key = "answer",
                value = answer
            )

    def _build_prompt(self, state):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        reranked_docs = state.get("reranked_docs")
        history_list = state.get("history")

        format_context = self._format_context(reranked_docs, self.config.max_context_chars)
        usage_chars = len(format_context)
        format_history = self._format_history(history_list,usage_chars)
        prompt = ANSWER_PROMPT.format(
            context = format_context,
            history = format_history,
            item_names = item_names,
            question = rewritten_query,
        )
        return prompt.strip()

    def _format_context(self, reranked_docs, max_context_chars):
        formatted_lines = []
        used_chars = 0
        for index, doc in enumerate(reranked_docs):
            content = doc.get("content")
            chunk_id = doc.get("chunk_id")
            title = doc.get("title")
            source = doc.get("source")
            url = doc.get("url")
            score = doc.get("score")
            line_1 = f"文档{index+1}"
            line_2 = f"chunk_id= {chunk_id},title = {title},source = {source},url = {url},score = {score}"
            line_3 = f"{content}"
            doc_line = f"{line_1}\n{line_2}\n{line_3}"

            if used_chars + len(doc_line) + 2 < max_context_chars:
                formatted_lines.append(doc_line)
                used_chars += len(doc_line) + 2
            else:
                break
        return "\n\n".join(formatted_lines)

    def _format_history(self, history_list, usage_chars):
        history_max_chars = self.config.max_context_chars - usage_chars
        if history_max_chars <= 0:
            return ""

        nick_names = {
            "user":"用户",
            "assistant":"助手"
        }
        formatted_lines = []
        used_chars = 0
        for history in history_list:
            role = history.get("role")
            text = history.get("text")
            line = f"{nick_names.get(role)}:{text}"
            if used_chars + len(line) + 1 <= history_max_chars:
                formatted_lines.append(line)
                used_chars += len(line) + 1
            else:
                break
        return "\n".join(formatted_lines)

    def _generate_answer(self, task_id, is_stream, prompt,state):
        #创建失败的响应
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as e:
            self.logger.error(f"LLM模型客户端创建失败：{e}")
            state["answer"] = "很抱歉，获取答案失败"
            self._push_answer(task_id,is_stream,state["answer"])
            return
        #根据is_stream判断调用放式
        if is_stream:
            final_answer = self._llm_stream(llm_client, task_id, prompt)
            state["answer"] = final_answer
        else:
            final_answer = self._llm_invoke(llm_client, task_id, prompt)
            state["answer"] = final_answer

    def _llm_stream(self, llm_client, task_id, prompt):
        #大模型相应数据流式输出
        llm_result = ""
        try:
            for section in llm_client.stream(prompt):
                delta_data = getattr(section, "content", "")
                if delta_data:
                    push_sse_event(
                        task_id=task_id,
                        event=SSEEvent.DELTA,
                        data={"answer": delta_data}
                    )
                    llm_result += delta_data
            # 调用完成推送数据
            push_sse_event(
                task_id=task_id,
                event=SSEEvent.FINAL,
                data={"answer": llm_result}
            )
        except Exception as e:
            return "很抱歉，获取答案失败"
        return llm_result

    def _llm_invoke(self, llm_client, task_id, prompt):
        llm_resp = llm_client.invoke(prompt)
        llm_result = getattr(llm_resp, "content", "")
        if not llm_result:
            llm_result = "很抱歉，获取答案失败"
        set_task_result(
            task_id = task_id,
            key = "answer",
            value = llm_result
        )

        return llm_result

    def _save_history(self, state):
        try:
            query = state.get("original_query")
            save_chat_message(
                session_id=state.get("session_id"),
                role="user",
                text=query,
                rewritten_query=state.get("rewritten_query"),
                item_names=state.get("item_names")
            )
            answer = state.get("answer")
            save_chat_message(
                session_id=state.get("session_id"),
                role="assistant",
                text=answer,
                rewritten_query=state.get("rewritten_query"),
                item_names=state.get("item_names")
            )
        except Exception as e:
            self.logging.error(f"保存历史记录失败:{e}")

if __name__ == "__main__":
    state =  {
        "session_id": "",
        "task_id": "1",
        "message_id": "",
        "original_query": "RS-12数字万用表如何测量电阻",
        "reranked_docs": [
        {
            "chunk_id": 468568991990268392,
            "content": "万用表RS-12的使用\n\n## 电阻测量\n\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![使用万用表测量电阻](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/dfbcdd205c8748df2005169dfc3c1b55f16dfe3a15024197c9d1a6b0064a9d6e.jpg)\n\n\n## 短路蜂鸣测试\n\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n\n3. 把表笔与被测物体相接触。\n\n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n",
            "title": "万用表RS-12的使用",
            "url": "",
            "source": "local",
            "score": 6.758012294769287
        },
        {
            "chunk_id": "",
            "content": "怎么用数字万用表测量电阻 1. 插表笔:将黑表笔插入数字万用表的COM(公共)接口,红表笔插入V/Ω(电压/电阻)接口;2. 选量程:将功能选择旋钮拨至欧姆档(标注为“Ω”)的合适量程(如测量未知电阻时,可先选择较大量程,再根据测量结果调整至更精确的量程);3. 校零(自动量程万用表无需此步骤):若使用手动量程万用表,将两表笔金属端短接,调整调零旋钮(若有)使显示屏示数为0Ω;4. 测电阻:将待测电阻从电路中断开(避免电路干扰或带电损坏仪表),用两表笔分别接触电阻的两个引脚;5. 读结果:观察显示屏数值,结合所选量程的单位(如“kΩ”“MΩ”)读取电阻值(如“20kΩ”量程下显示“10.5”,表示电阻为10.5kΩ)。",
            "title": "怎么用数字万用表测量电阻 ",
            "url": "https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1728213610231814564&fr=search",
            "source": "web",
            "score": 5.555666446685791
        },
        {
            "chunk_id": "",
            "content": "万用表测电阻数字表示什么?例如显示数字为0,是表示没有电阻?还是电阻无限大,不通电? 万用表测电阻数字表示什么?例如显示数字为0,是表示没有电阻?还是电阻无限大,不通电? 万用表测电阻数字表示什么? 例如显示数字为0,是表示没有电阻?还是电阻无限大,不通电? 如果显示为20,是否说电阻为20?还是表示通电值为20? 知乎用户 【溪城说电路,好用不迷糊。】 各位看官好,今天不说电路图,咱好好唠唠,数字万用表如何测电阻? 说到数字万用表,相信大家都不陌生。 有人会问,测电阻这么简单,还要讨论吗? 别急,看完这篇文章,相信你会有新收获。 准备工作: 步骤1:检查万用表 开机,检查无电池亏电报警。 将红表笔插入V/Ω插孔,黑表笔插入COM插孔,选择任意“欧姆档”。 短接试验:将两支表笔金属探针相碰短接,此时,屏幕应该显示:“0” Ω,或者“000” 开路试验:将两支表笔金属探分开,此时,屏幕应该显示:“1” ,或者 “0L” 做完这2项测试,效果如上述描述一样,说明万用表是好的。 步骤2:预估电阻阻值,选择适宜量程。 根据电阻标记的阻值,选择适宜的欧姆档量程。 被测电阻值,小于 200Ω时,用200Ω挡测量; 被测电阻值,在 200Ω — 2KΩ之间,用2kΩ挡测量; 被测电阻值,在 2kΩ — 2MΩ之间, 用2MΩ挡测量; 被测电阻值,大于2MΩ时,用20MΩ挡测量。 如果,选择的量程小于电阻值,则测量时,会显示“1” ,或者 “0L”,需要更换大一档的测量档位。 步骤3:测电阻。 断电测电阻,这是前提,否则测不准,还有可能伤表。 把两支表笔,并联在被测量电阻的两引脚上,此时测得的电阻值,应与该电阻的标称阻值相符合。 测量电阻时,手不能同时触及被测电阻的两端,以免人体电阻的并联作用,影响测量结果,尤其测量高阻值时,要注意这一点。 因为电阻有误差等级,实测阻值与标称阻值之间允许有士5%、士10%和士20%的误差。 若超出误差范围,则说明被测电阻已经变值。 若测得的结果为0,则说明被测电阻已短路烧坏,需要更换 若测得的电阻值为“1” ,或者 “0L”,则说明该电阻已断路损坏,需要更换 步骤4:严谨的读数。 在做仪表检查,短接试验时,应记录读数。 假设短接试验,读数为:0.3Ω,实际测量为:5.0Ω, 则电阻的实际阻值为:5.0— 0.3 = 4.7Ω。 此读法,适用于,200Ω以下的小阻值电阻读数。 一般来说,表笔短接试验,读数都在0Ω左右。 测量只有几欧姆的电阻时,应当刮去电阻引脚的氧化层。",
            "title": "万用表测电阻数字表示什么?例如显示数字为0,是表示没有电阻?还是电阻无限大,不通电?",
            "url": "https://www.zhihu.com/question/580960365/answer/3001352962",
            "source": "web",
            "score": 5.370693683624268
        },
        {
            "chunk_id": "",
            "content": "简述数字万用表测量电阻方法和步骤。 将表笔插进“COM”和“VΩ”孔中,把旋钮打旋到“Ω”中所需的量程(注意:表盘上的数值均为最大量程),用表笔接在电阻两端金属部位,测量中可以用手接触电阻,但不要把手同时接触电阻两端,这样会影响测量精确度的(人体是电阻很大但是有限大的导体)。读数时,要保持表笔和电阻有良好的接触,数值可以直接从显示屏上读取,若显示为“1.”,则表明量程太小,那么就要加大量程后再测量工业电器。注意单位:在“200”档时单位是“Ω”,在“2K”到“200K“档时单位为“KΩ ”,“2M”以上的单位是“MΩ”。图1.11(b)为测量电阻方法示意图。",
            "title": "简述数字万用表测量电阻方法和步骤。",
            "url": "https://easylearn.baidu.com/edu-page/tiangong/bgkdetail?id=8436047f168884868762d633&fr=search",
            "source": "web",
            "score": 5.253030776977539
        },
        {
            "chunk_id": "",
            "content": "如何正确使用数字万用表测量电压、电阻? 使用前准备:阅读万用表说明书,熟悉电源开关、量程开关、插孔作用;检查仪表及表笔是否损坏(如表笔裸露、液晶显示异常等);将电源开关置于“ON”位置。直流/交流电压测量:红表笔插入“V/Ω”插孔,黑表笔插入“COM”插孔;功能量程选择开关置于DCV(直流电压)或ACV(交流电压)合适量程(未知电压范围时先选高量程再逐步调低);测试笔并联在被测信号源两端,记录读数及红表笔极性。电阻测量:红表笔插入“V/Ω”插孔,黑表笔插入“COM”插孔;功能量程选择开关置于“Ω”量程;若电阻在电路中需先断开电源,将测试笔跨接在被测电阻两端;若显示“1”说明超出量程,应选择更高档位。测量后处理:关掉仪表电源(“OFF”),长时间不用时取出电池;拔出并收好测试表笔。",
            "title": "如何正确使用数字万用表测量电压、电阻?",
            "url": "https://easylearn.baidu.com/edu-page/tiangong/questiondetail?id=1828370570798890731&fr=search",
            "source": "web",
            "score": 4.93678617477417
        }
    ],
    "prompt": "",
    "answer": "",
    "item_names": [
        "RS PRO RS-12 数字万用表"
    ],
    "rewritten_query": "RS-12数字万用表如何测量电阻？",
    "history": [],
    "is_stream": False
}
    node = AnswerOutputNode()
    state = node.process(state)
    json_str = json.dumps(state,indent=4,ensure_ascii=False)
    print(json_str)










































