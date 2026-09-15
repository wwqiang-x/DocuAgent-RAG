import base64
import logging
import os
import re
import json
from pathlib import Path
from typing import Tuple,List,Dict
from collections import deque
import time
import urllib.parse
import minio
from click import prompt

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import StateFieldError, ImageProcessingError
from processor.import_process.state import ImportGraphState
from prompts.import_prompt import IMG_SUMMARY_PROMPT
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients


class MdImgNode(BaseNode):
    name = "md_img_node"

    def process(self,state: ImportGraphState) -> ImportGraphState:
        md_content, md_path_obj,md_img_path_obj = self._get_md_content_and_path(state)
        #过滤图片
        img_info_list = self._scan_and_filter_images(md_img_path_obj, md_content)
        #调用vlm为每个图片生成摘要
        image_summaries = self._generate_image_summaries(img_info_list)
        #将图片信息传到minioadmin中，获取在minio中的路径
        new_md_content = self._upload_img_and_update_md(md_path_obj.stem,md_content,img_info_list,image_summaries)
        #更新state
        state["md_content"] = new_md_content
        return state

    def _get_md_content_and_path(self,state:ImportGraphState)->Tuple[str,Path, Path]:
        #获取state参数判断文件是否存在。获取md.content和md路径和图片
        #拿到md文档路径
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(
                node_name = self.name,
                field_name = "md_path",
                message = "md_path不存在"
            )
        #读取文档content
        md_path_obj = Path(md_path)
        try:
            with open(md_path_obj, "r", encoding="utf-8") as f:
                md_content = f.read()
        except IOError as e:
            raise ImageProcessingError(
                node_name=self.name,
                message="文档读取内容失败"
            )
        #获取文档的路径
        md_img_path_obj = md_path_obj.parent / "images"
        return md_content, md_path_obj,md_img_path_obj

    # List[Tuple[str,str,Tuple[str,str,str]]]返回图片名字 图片路径 图片描述（标题，上文，下文）
    def _scan_and_filter_images(self,md_img_path_obj:Path,md_content:str) -> List[Tuple[str,str,Tuple[str,str,str]]]:
        # 遍历images这个目录拿到图片名称 上下文 过滤格式
        #1. 遍历所有图片 os.listdir遍历成一个列表结构
        img_info_list = []
        for img_name in os.listdir(md_img_path_obj):
            #2. 过滤格式 os.path.splitext(img_name)[1]获取格式 将文件分割成list(d\:.....\text,.jpg)
            ext = os.path.splitext(img_name)[1]
            if ext not in self.config.image_extensions:
                continue
            #3. 获取图片路径
            img_path = str(md_img_path_obj / img_name)
            #4 .提取这个图片的上下文描述
            img_context = self._extract_img_context(img_path,md_content)
            #5. 把内容传递给列表
            img_info_list.append((img_name,img_path,img_context))

        return img_info_list

    def _extract_img_context(self,img_name,md_content)->Tuple[str,str,str]:
        #提取图片标题、图片上文、图片下文
        #获取文档的每一行行号
        #使其图片名字 = 正则名字 提取文件所在行
        #向上遍历行号 找到标题 = 正则 获取图片标题
        #提取（标题+1,图片）获取上文
        #向下遍历len（md文档）找到下一个标题
        # 提取（图片+1,标题）获取下文
        #返回图片标题、图片上文、图片下文

        #图片出现多次的处理
        context_list = []

        #1.获取每行内容
        md_lines = md_content.split("\n")
        #定义一下图片的正则表达
        img_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
        for index,line in enumerate(md_lines):
            if not img_pattern.search(line):
                continue
            #获取到图片的行号
            img_index = index
            #2. 截取标题和上文
            #定义标题的正则
            title_pattern = re.compile(r'^#{1,6}\s+')
            #从图片上一行找到md文档的最上部分
            pre_title_index = -1
            pre_title_content = ""
            for i in range(img_index-1,-1,-1):
                if title_pattern.search(md_lines[i]):
                    pre_title_index = i
                    # 截取标题
                    pre_title_content = md_lines[i]
                    break
            #截取上文
            pre_context = "\n".join(md_lines[pre_title_index + 1:img_index])
            final_pre_context = self._extract_context_with_limit("pre_context",200,"up")

            for j in range(img_index+1,len(md_lines)):
                if title_pattern.search(md_lines[j]):
                    post_title_index = j
                    break
            #截取下文
            post_context = "\n".join(md_lines[img_index + 1:post_title_index])
            final_post_context = self._extract_context_with_limit(post_context, 200, "down")
            #构建元组加入（标题，上文，下文）
            context_list.append((pre_title_content,final_pre_context,final_post_context))

        #返回上下文列表
        if len(context_list) == 0:
            return ("", "", "")

        return context_list[0]

    def _extract_context_with_limit(self,context,max_chars,direction)->str:
        #context: 上下文
        #max_chars: 最大文字量
        #direction: up上文 down下文

        #用来接收行 为转成段做准备
        current_para = []
        # 存储段落
        para_list = []

        #用context_lines接收上（下）文的内容
        context_lines = context.split("\n")

        #遍历context_lines
        for line in context_lines:
            #去掉这一行左右的空格
            stripped_line = line.strip()
            if stripped_line:
                #如果当行不为空，判断是否为图片
                if re.match(r"^!\[.*?\]\(.*?\)$", stripped_line):
                    #如果是图片判断是否为空集
                    if current_para:
                        p_str = "\n".join(current_para)
                        para_list.append(p_str)
                        current_para = []
                else:
                   current_para.append(stripped_line)
            else:
                if current_para:
                    p_str = "\n".join(current_para)
                    para_list.append(p_str)
                    current_para = []
        if current_para:
            p_str = "/n".join(current_para)
            para_list.append(p_str)

        #现在拿到段落列表para_list了
        #编写上下文max_chars为最大长度
        #超出部分段落全部不要

        selected_para_list = []

        if direction == "up":
            para_list.reverse()

        if len(para_list) > 0:
            selected_para_list = para_list[0]
            total_chars = len(para_list[0])

            for p in para_list[1:]:
                total_chars += len(p)
                if total_chars + len(p) > max_chars:
                    break
                else:
                    selected_para_list.append(p)
                    total_chars += len(p)

        if direction == "up":
            selected_para_list.reverse()

        return "\n\n".join(selected_para_list)

    def _generate_image_summaries(self,img_info_list)->Dict[str, str]:
        #调用一个vlm==千问VLM
        image_summaries = {}
        #1、创建客户端
        vlm_client = AIClients.get_vlm_client()
        #2、遍历图片信息列表
        request_timestamps: Deque[float] = deque()
        for img_name, img_path, img_context in img_info_list:
            # self.enforce_rate_limit(request_timestamps,8,60)
            summary = self._get_img_summary(vlm_client,img_name,img_path,img_context)
            image_summaries[img_name] = summary
        #返回图片摘要
        return image_summaries

    def _get_img_summary(self,vlm_client,img_name,img_path,img_context):
        #调用vlm
        #1、构造提示词
        title, pre_context, post_context = img_context
        img_context = f"[图片上文]:{pre_context}\n[图片下文]:{post_context}"
        prompt = IMG_SUMMARY_PROMPT.format(
            title_content=title,
            img_context=img_context,
        )

        #2、读取图片数据
        try:
            with open(img_path,"rb") as f:
                img_data = f.read()
                img_data_str = base64.b64encode(img_data).decode()
        except IOError as e:
            raise ImageProcessingError(f"图片文件读取失败：{e}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data_str}"
                        },
                    }
                ]
            }
        ]

        completion = vlm_client.chat.completions.create(
            model=self.config.vl_model,
            messages=messages
        )

        return completion.choices[0].message.content

    def enforce_rate_limit(self,
            request_timestamps: deque,  # 存储请求时间戳
            max_requests: int,  # 最大请求数
            window_seconds: int = 60  # 时间窗口（秒）
    ):
        current_time = time.time()

        # 移除窗口外的旧时间戳
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        # 如果达到上限，等待
        if len(request_timestamps) >= max_requests:
            sleep_time = window_seconds - (current_time - request_timestamps[0])
            if sleep_time > 0:
                self.logger.info(f"{self.name}is rate limited, sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)

        # 记录本次请求
        request_timestamps.append(time.time())

    def _upload_img_and_update_md(self,document_name , md_content, img_info_list, image_summaries):
        #将图片上传到Minio
        minio_client = StorageClients.get_minio_client()
        #遍历信息tupian
        urls = {}
        for img_name, img_path, img_context in img_info_list:
            minio_client.fput_object(
                bucket_name = self.config.minio_bucket,
                object_name = f"{document_name}/{img_name}",
                file_path = img_path
            )
            original_url = f"{self.config.get_minio_base_url()}/{self.config.minio_bucket}/{document_name}/{img_name}"
            # 关键：全量编码，把中文、空格全变成 %XX
            encoded_url = urllib.parse.quote(original_url, safe='')
            # 关键：套上你的代理接口
            remote_url = f"http://127.0.0.1:8001/api/proxy/image?url={encoded_url}"
            urls[img_name] = remote_url

        #summary摘要 remote_url地址
        new_content = md_content
        for img_name, summary in image_summaries.items():
            remote_url = urls[img_name]
            replace_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")
            new_content = replace_pattern.sub(f"![{summary}]({remote_url})", new_content)


        return new_content
if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    state = {
    "task_id": "",
    "is_pdf_read_enabled": True,
    "is_md_read_enabled": False,
    "file_dir": "D:\\knowledge_base\\processor\\import_process\\improcess_files",
    "import_file_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
    "pdf_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
    "md_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md",
    "file_title": "万用表RS-12的使用",
    "md_content": "",
    "chunks": [],
    "item_name": ""
}

    node = MdImgNode()
    state = node(state)
    json_str = json.dumps(state,indent = 4,ensure_ascii=False)
    print(json_str)

# {
#     "task_id": "",
#     "is_pdf_read_enabled": true,
#     "is_md_read_enabled": false,
#     "file_dir": "D:\\knowledge_base\\processor\\import_process\\improcess_files",
#     "import_file_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
#     "pdf_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用.pdf",
#     "md_path": "D:\\knowledge_base\\processor\\import_process\\improcess_files\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md",
#     "file_title": "万用表RS-12的使用",
#     "md_content": "![RS PRO品牌标志](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/d329d008ba12d6f5eed073b52a378a6829cb4c1baef85b0d77934fa902bbb7fd.jpg)\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表\n\n![中文语言标识](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/81735c16d6175e1dd624407b3448d8bf2039d8e22b1f9cc3941e533706a070a9.jpg)\n\nCE\n\n![多功能数字万用表](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/f179d9399297a15b5d4e764602734c25302eec0b528b231f0e455ca9c76dce0b.jpg)\n\n## 安全手册\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>输入量程</td></tr><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>最大输入</td></tr><tr><td rowspan=1 colspan=1>交/直流电压</td><td rowspan=1 colspan=1>直流/交流电压600V</td></tr><tr><td rowspan=1 colspan=1>直流/交流电压</td><td rowspan=1 colspan=1>直流/交流电压600V, 200Vrms 用于200mV量程</td></tr><tr><td rowspan=1 colspan=1>mA直流</td><td rowspan=1 colspan=1>200mA 250V快速熔断保险丝</td></tr><tr><td rowspan=1 colspan=1>A DC</td><td rowspan=1 colspan=1>10A 250V 快速熔断保险丝(最多每15分钟，需时30秒)</td></tr><tr><td rowspan=1 colspan=1>电阻,短路测试</td><td rowspan=1 colspan=1>250Vrms, 最多15秒</td></tr></table>\n\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n\n7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n\n## 安全标识\n\n![警告标志](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/6ed1c2d2c192fe7422f77d0eb13133a4f4b01ea3e738ce2017bc5df75185e1a0.jpg)\n\n表明此操作须参照说明书进行。\n\nWARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n\nCAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n\n![最大液面标记](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/4033d9da04e1ffceb8382efdf1c281a8fbddf7ebe238b3aa131ff2f0c43fbeb0.jpg)\n\n请勿连接到500VAC或VDC的电路上。\n\n![闪电符号](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/2219ca75130874ec766983013be86c7afd233a4a1b3a5188990261156f1f2cc5.jpg)\n\n表明此端口可能出现危险电压。\n\n![双层边框正方形图案](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/6b7cc68242e90cec872c128c48320467c4ea101b9973420a5a2a2c46c1a7d489.jpg)\n\n双绝缘保护。\n\n## 控制与端口\n\n1.LCD液晶显示\n\n2.功能选择转盘\n\n3.10A端口\n\n4.COM端口\n\n5.正极端口\n\n6.数据保持按键\n\n7.背光按键\n\n![数字万用表外观及接口标识示意图](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/d6946861c4592804bd8d7e75b58029565712d4dc58f855e374bf0fcf370c91dd.jpg)\n\n## 功能符号指示\n\n•))) 蜂鸣指示\n\n![二极管符号](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/e92ffd955b1ca1fd14290da681a763771c958cbcf0a73a332107f471f96c29b2.jpg)\n\n二极管测试指示\n\nµ micro (电流范围)\n\nm milli ( 电压/电流范围)\n\nk kilo (电阻范围)\n\nVDC 直流电压\n\nVAC 交流电流\n\nADC 直流电流\n\nBAT 电池电量不足指示\n\n## 规格\n\n<table><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 colspan=1>量程</td><td rowspan=1 colspan=1>分辨率</td><td rowspan=1 colspan=1>精确度</td></tr><tr><td rowspan=5 colspan=1>直流电压</td><td rowspan=1 colspan=1>200mV</td><td rowspan=1 colspan=1>0.1mV</td><td rowspan=3 colspan=1>± (0.5% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000mV</td><td rowspan=1 colspan=1>1mV</td></tr><tr><td rowspan=1 colspan=1>20V</td><td rowspan=1 colspan=1>0.01V</td></tr><tr><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=2 colspan=1>交流电压</td><td rowspan=1 colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (1.2% reading + 10 digits50/60Hz)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=4 colspan=1>直流电流</td><td rowspan=1 colspan=1>2000μA</td><td rowspan=1 colspan=1>1μA</td><td rowspan=2 colspan=1>± (1.0% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>20mA</td><td rowspan=1 colspan=1>10μA</td></tr><tr><td rowspan=1 colspan=1>200mA</td><td rowspan=1 colspan=1>100μA</td><td rowspan=1 colspan=1>± (1.2% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>10A</td><td rowspan=1 colspan=1>10mA</td><td rowspan=1 colspan=1>± (2.0% reading + 2 digits)</td></tr><tr><td rowspan=5 colspan=1>电阻</td><td rowspan=1 colspan=1>200Ω</td><td rowspan=1 colspan=1>0.1Ω</td><td rowspan=4 colspan=1>± (0.8% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>2000Ω</td><td rowspan=1 colspan=1>1Ω</td></tr><tr><td rowspan=1 colspan=1>20kΩ</td><td rowspan=1 colspan=1>0.01kΩ</td></tr><tr><td rowspan=1 colspan=1>200kΩ</td><td rowspan=1 colspan=1>0.1kΩ</td></tr><tr><td rowspan=1 colspan=1>2000kΩ</td><td rowspan=1 colspan=1>1kΩ</td><td rowspan=1 colspan=1>± (1.0% reading + 2 digits)</td></tr><tr><td rowspan=2 colspan=1>电池</td><td rowspan=1 colspan=1>9V</td><td rowspan=1 colspan=1>10mV</td><td rowspan=2 colspan=1>± (1.0% reading + 2 digits)</td></tr><tr><td rowspan=1 colspan=1>1.5V</td><td rowspan=1 colspan=1>1mV</td></tr></table>\n\n注意: 精确度规格由两种因素组成。  \n● (% reading) –测量电路的精确度。  \n● (+ digits) –数位转换器条码的精确度。  \n注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。\n\n## 技术指标说明\n\n二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n\n短路蜂鸣测试 若电阻小于30时产生蜂鸣\n\n电池测试电流 9V (6mA)；1.5V (100mA)\n\n输入阻抗 >1MΩ\n\n交流电压频宽 45Hz～450Hz\n\nDCA电压跌路测试 200mV\n\n显示 3 ½ 数位，2000位液晶显示，1.1”数位\n\n超量程提示 以“1”表示\n\n极性 自动(正极无显示);负极显示(-)\n\n测量率 正常情况下每秒2次\n\n低电池提示 电池电压不足时，显示BAT符号\n\n电池 一粒9V (NEDA 1604) 电池\n\n保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n\n操作环境 32°F～122°F (0°C～50°C)\n\n储存温度 -4°F～140°F (-20°C～60°C)\n\n相对湿度 <70% 操作, <80% 储存\n\n室内使用,最高海拔 7000英尺(2000米)\n\n重量 255g\n\n尺寸 150mm x 70mm x 48mm\n\n安全认证 室内使用，符合过电压类别II\n\n污染级别 2\n\n## 电池安装\n\n警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 把表笔与仪表断开。\n\n2. 用螺丝刀拧开电池后盖上的螺母。\n\n3. 正确安装电池，正负极应一致。\n\n4. 盖上电池后盖并拧紧螺丝钉。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n\n## 操作指导\n\n## 数值保持按键HOLD\n\n保持键允许仪表固定测量值以供参考：\n\n1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n\n2. 再次按下“DATA HOLD”键 切换至正常操作\n\n## 背光灯键（BACKLIGHT）\n\n1. 按下背光灯键开启背光灯。\n\n2. 再次按背光灯键关闭背光灯。\n\n警告：小心触电，高压电流十分危险，应小心操作。\n\n1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n\n2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n\n注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n\n## 测量非接触交流电压\n\n警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n\n1. 让其探头靠近或插入火线的输出插座孔时。\n\n2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n\n注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n\n注意: 此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n\n## 直流电压测量\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n\n## 交流电压测量\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![使用万用表测量电压](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/84c37b209829d15820d5bbe76bbc98e1bf9eddc58bd9c983fc710cb2747d341b.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n\n## 直流电流测量\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![数字万用表测量电压](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/8eb1e59b1e3f5e200f6d947da47dcd767fe061b91e72b1ba5325869677dcdad2.jpg)\n\n1. 将黑色表笔插入负极COM端口。\n\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n\n6. 接通电源。\n\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n![电压表测量负载两端电压](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/950918b0a12b239de83d45093fa5e6258bfa3d33848f927ecf0993f3210bf3d9.jpg)\n\n## 电阻测量\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![使用万用表测量电阻](http://192.168.10.100:9000/knowledge-base/万用表RS-12的使用/dfbcdd205c8748df2005169dfc3c1b55f16dfe3a15024197c9d1a6b0064a9d6e.jpg)\n\n## 短路蜂鸣测试\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n\n3. 把表笔与被测物体相接触。\n\n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n\n## 二极管测试\n\n1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n\n2. 将功能转盘置于 位置。\n\n3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 0V，开路时会在两种极性上显示“1”符号。\n\n## 电池测试\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V 电池：</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 colspan=1>7.2 至 8.2V</td><td rowspan=1 colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V 电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 colspan=1>1.22 至 1.35V</td><td rowspan=1 colspan=1>&lt;1.22V</td></tr></table>\n\n## 更换电池\n\n警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n\n2. 按下面的步骤安装电池。\n\n3. 妥善处理废电池。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n## 更换保险丝\n\n警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n\n1. 把表笔与仪表及其它被测物断开。\n\n2. 用螺丝刀拧开保险丝门上的螺母。\n\n3. 轻轻取出废旧的保险丝。\n\n4. 装入新的保险丝。\n\n5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V 快速熔断保险丝用于10A的量程。\n\n6. 盖回后盖，拧紧螺钉。\n\n警告: 为防触电，在保险盖盖紧前请勿操作仪表。",
#     "chunks": [],
#     "item_name": ""
# }











































