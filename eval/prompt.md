##生成黄金问题集的提示词
请帮我生成基于ragas评估的黄金问题集
【参考文档】
请参考eval/hak180产品安全手册_new.md 来生成黄金问题集
【产出要求】
1. 保存到目录eval下,名称：qs.csv
2. 要求列头：question, ground_truth
3. 编码集：UTF-8 BOM，支持excel打开
4. 生成的每个问题请以：hak180烫金机为开头
5. 生成10对问答

## 生成评估要求
请帮我基于ragas生成一个python语言的评估测试程序
【程序输入要求】
    1.要求从eval下读取 qs.csv，改文件包含两列：问题和参考答案
【程序过程要求】
    1.rag流程入口在 processor/query_process/main_graph.py程序中
    2.从state中的history获取上下文
    3.ragas要使用到的llm在utils/client/ai_clients.py中 embding模型在utils/embedding_util.py中获取
    4.要求ragas生成5个指标：Faithfulness， Answer Relevancy， Context Precision， Context Recall， Answer Correctness
【程序输出要求】
    1.将评估结果保存在目录eval中qa_result.csv文件下，编码为UTF-8 BOM格式
    2，qa_result.csv一共有9列：question，context，answer，ground_trush，Faithfulness， Answer Relevancy， Context Precision， Context Recall， Answer Correctness
【代码要求】
    1.所有方法增加中文注释
    2。核心步骤的函数名请使用：step_1,step_2...作为前缀，来明确步骤
    3.程序保存在wval目录下eval.py文件中