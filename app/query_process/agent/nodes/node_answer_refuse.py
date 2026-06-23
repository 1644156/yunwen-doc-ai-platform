import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.core.logger import logger
from app.clients.mongo_history_utils import save_chat_message

REFUSAL_TEMPLATE = """抱歉，针对您的问题"{query}"，我在知识库中没有找到能可靠回答的资料。

可能原因：
- 该问题不在我的知识范围内
- 您的问题可能需要更具体的描述

建议：
1. 尝试用其他关键词重新描述
2. 确认商品名称是否正确
3. 联系人工客服获取专业解答"""


def node_answer_refuse(state):
    """
    节点功能：拒答
    当 CRAG 判定所有检索路径均失败时，返回拒答模板。
    """
    print("---node_answer_refuse 拒答---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    query = state.get("rewritten_query") or state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    # 生成拒答内容
    answer = REFUSAL_TEMPLATE.format(query=query)

    # 输出
    if is_stream:
        push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        push_to_session(state["session_id"], SSEEvent.FINAL, {
            "answer": answer,
            "status": "refused",
        })
    else:
        set_task_result(state["session_id"], "answer", answer)

    # 存储历史
    session_id = state.get("session_id")
    item_names = state.get("item_names", [])
    if answer:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            item_names=item_names,
            rewritten_query=query
        )

    logger.info(f"拒答完成: {answer[:50]}...")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---node_answer_refuse 完成---")

    return {"answer": answer}