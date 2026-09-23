"""Agent 检索策略规则。"""

import re


# 这类问题通常依赖实时或外部信息，不能仅凭本地说明书直接回答。
_WEB_REQUIRED_PATTERN = re.compile(
    r"("
    r"市场价|价格|售价|报价|多少钱|打折|折扣|优惠|"
    r"库存|现货|购买|下单|官网|电商|"
    r"最新|实时|当前|今天|新闻|发布|上市"
    r")",
    re.IGNORECASE,
)


def requires_web_search(
    original_query: str,
    rewritten_query: str = "",
) -> bool:
    """判断问题是否必须补充网络搜索。"""
    query = f"{original_query} {rewritten_query}".strip()
    return bool(_WEB_REQUIRED_PATTERN.search(query))
