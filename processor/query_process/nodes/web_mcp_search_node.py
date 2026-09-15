import json
import logging
import asyncio
import httpx2

from processor.query_process.base import BaseNode
from processor.query_process.state import QueryGraphState


class WebMcpSearchNode(BaseNode):
    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 从state中获取 rewritten_query, item_names
        rewritten_query = state.get("rewritten_query", "")
        # 2. 调用mcp进行网络搜索（注意：这里不再使用 agents 库，而是直接用 httpx2 发送原始请求）
        mcp_search_result = asyncio.run(self._call_mcp(rewritten_query))
        # 3. 装入state
        return {"web_search_docs": mcp_search_result}

    async def _call_mcp(self, rewritten_query):
        mcp_search_result = []
        url = self.config.mcp_dashscope_base_url
        headers = {
            "Authorization": f"Bearer {self.config.mcp_dashscope_api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx2.AsyncClient(timeout=30.0) as client:
                # 1. 初始化 MCP 会话（这是兼容阿里云的正确握手方式）
                init_payload = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "knowledge_base", "version": "1.0"}
                    },
                    "id": 1
                }
                resp = await client.post(url, headers=headers, json=init_payload)
                if resp.status_code != 200:
                    print(f"MCP 初始化失败: {resp.status_code} {resp.text}")
                    return mcp_search_result

                # 获取 Session ID (如果有)
                session_id = resp.headers.get("mcp-session-id")

                # 2. 调用搜索工具
                call_payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "bailian_web_search",
                        "arguments": {
                            "query": rewritten_query,
                            "count": 5
                        }
                    },
                    "id": 2
                }

                call_headers = dict(headers)
                if session_id:
                    call_headers["mcp-session-id"] = session_id

                resp2 = await client.post(url, headers=call_headers, json=call_payload)
                if resp2.status_code != 200:
                    print(f"MCP 调用工具失败: {resp2.status_code} {resp2.text}")
                    return mcp_search_result

                # 3. 解析结果
                result_json = resp2.json()
                content = result_json["result"]["content"]
                json_str = content[0]["text"]
                obj = json.loads(json_str)
                pages = obj.get("pages", [])

                for page in pages:
                    title = page.get("title", "")
                    url = page.get("url", "")
                    snippet = page.get("snippet", "")
                    mcp_search_result.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    })
        except Exception as e:
            print(f"MCP 搜索调用异常: {e}")

        return mcp_search_result


if __name__ == "__main__":
    node = WebMcpSearchNode()
    state = {
        "rewritten_query": "华为擎云W585 台式计算机和华为 B3-243H 显示器的参数",
        "item_names": [
            "华为擎云W585 台式计算机",
            "华为 B3-243H 显示器"
        ]
    }

    state = node.process(state)

    jsaon_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(jsaon_str)