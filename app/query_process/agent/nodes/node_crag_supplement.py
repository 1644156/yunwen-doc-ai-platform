import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def node_crag_supplement(state):
    """
    节点功能：CRAG 补充搜索
    当检索质量中等（ambiguous）时，触发网络搜索补充信息。
    复用 node_web_search_mcp 的逻辑。
    """
    print("---node_crag_supplement 补充搜索---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 直接复用 web_search_mcp 节点的逻辑
    from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
    result = node_web_search_mcp(state)

    web_docs = result.get("web_search_docs", [])
    logger.info(f"CRAG 补充搜索完成，网络结果数量: {len(web_docs)}")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print(f"---node_crag_supplement 完成: {len(web_docs)} 条---")

    return {"web_search_docs": web_docs}