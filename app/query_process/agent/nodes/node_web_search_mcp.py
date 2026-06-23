import asyncio
import json
import sys
import threading
from app.core.logger import logger
from fastmcp import Client
from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task, add_done_task

DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

# ============================
# MCP 专用持久化事件循环
# fastmcp.Client 的 SSE 传输需要稳定的事件循环，
# 不能每次 asyncio.run() 创建新循环，否则会话会中断。
# ============================
_mcp_loop = None
_mcp_loop_thread = None
_mcp_loop_lock = threading.Lock()


def _get_mcp_loop():
    """获取或创建 MCP 专用的持久化事件循环"""
    global _mcp_loop, _mcp_loop_thread
    with _mcp_loop_lock:
        if _mcp_loop is None or _mcp_loop.is_closed():
            _mcp_loop = asyncio.new_event_loop()
            _mcp_loop_thread = threading.Thread(target=_mcp_loop.run_forever, daemon=True)
            _mcp_loop_thread.start()
    return _mcp_loop


def _run_async(coro):
    """在 MCP 专用事件循环中执行异步代码"""
    loop = _get_mcp_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


async def mcp_call_streamable(query: str, count: int = 5):
    """调用百炼 MCP 网络搜索工具"""
    client = Client(
        DASHSCOPE_BASE_URL_STREAMABLE,
        auth=f"Bearer {DASHSCOPE_API_KEY}",
        timeout=15
    )
    async with client:
        result = await client.call_tool(
            "bailian_web_search",
            {"query": query, "count": count}
        )
        return result


def mcp_search(query: str, count: int = 5) -> list:
    """同步调用 MCP 搜索，返回网页结果列表"""
    result = _run_async(mcp_call_streamable(query, count))
    # fastmcp.Client.call_tool 返回 list[TextContent]
    if result and len(result) > 0:
        text = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(text).get("pages", [])
    return []


def node_web_search_mcp(state):
    """节点功能：调用外部搜索引擎补充信息"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print("---node-web-search-mcp处理---")

    query = state.get("rewritten_query")
    try:
        web_documents = mcp_search(query)
    except Exception as e:
        logger.warning(f"MCP web search 调用失败，跳过网络搜索: {e}")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
        return {"web_search_docs": []}

    logger.info(f"mcp搜索的结果为:{web_documents}")
    print("---node-web-search-mcp处理结束---")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {"web_search_docs": web_documents}


from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id": "mcp_01",
        "rewritten_query": "HAK 180 烫金机的功率",
        "is_stream": True
    }
    result_state = node_web_search_mcp(test_state)
    search_results = result_state.get('web_search_docs', [])
    print(f"搜索结果数量: {len(search_results)}")
    for r in search_results[:3]:
        print(f"  - {r.get('title', '')} ({r.get('hostname', '')})")
