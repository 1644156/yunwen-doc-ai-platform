import sys
import json
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task


def node_query_routing(state):
    """
    节点功能：查询意图分类路由
    将查询分为 factual/reasoning/temporal/ambiguous 四类，
    后续根据类型选择不同的检索策略。
    """
    print("---node_query_routing 查询路由分类---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    rewritten_query = state.get("rewritten_query") or state.get("original_query", "")
    item_names = state.get("item_names", [])
    item_names_str = ", ".join(item_names) if item_names else "未知"

    # 加载路由分类 prompt
    prompt = load_prompt("query_routing",
                         item_names=item_names_str,
                         rewritten_query=rewritten_query)

    # 调用 LLM 进行分类
    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke(prompt)
        raw_content = response.content.strip()

        # 去除可能的 markdown 代码块标记
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[-1]
        if raw_content.endswith("```"):
            raw_content = raw_content.rsplit("```", 1)[0]
        raw_content = raw_content.strip()

        data = json.loads(raw_content)
        query_type = data.get("query_type", "ambiguous").lower()
        confidence = float(data.get("confidence", 0.5))
        reason = data.get("reason", "")

        # 校验 query_type 合法性
        valid_types = {"factual", "reasoning", "temporal", "ambiguous"}
        if query_type not in valid_types:
            logger.warning(f"LLM 返回了无效的 query_type: {query_type}，降级为 ambiguous")
            query_type = "ambiguous"
            confidence = 0.3

        logger.info(f"查询分类完成: type={query_type}, confidence={confidence}, reason={reason}")

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"查询分类解析失败，降级为 ambiguous: {e}")
        query_type = "ambiguous"
        confidence = 0.0

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print(f"---node_query_routing 完成: {query_type}---")

    return {
        "query_type": query_type,
        "routing_confidence": confidence,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_state = {
        "session_id": "test_routing_001",
        "original_query": "H3C路由器和华为路由器哪个好",
        "rewritten_query": "H3C路由器和华为路由器的对比分析",
        "item_names": ["H3C路由器", "华为路由器"],
        "is_stream": False,
    }
    result = node_query_routing(test_state)
    print(f"分类结果: {result}")