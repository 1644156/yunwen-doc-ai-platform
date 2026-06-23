import sys
import json
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task


def node_query_rewrite(state):
    """
    节点功能：CRAG 查询改写
    当检索质量低时，改写查询以提高检索效果。
    递增 crag_retry_count，更新 rewritten_query。
    """
    print("---node_query_rewrite 查询改写---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    rewritten_query = state.get("rewritten_query") or state.get("original_query", "")
    item_names = state.get("item_names", [])
    item_names_str = ", ".join(item_names) if item_names else "未知"
    retry_count = state.get("crag_retry_count", 0)

    # 加载改写 prompt
    prompt = load_prompt("crag_query_rewrite",
                         rewritten_query=rewritten_query,
                         item_names=item_names_str)

    # 调用 LLM 改写
    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke(prompt)
        raw_content = response.content.strip()

        # 去除 markdown 代码块标记
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[-1]
        if raw_content.endswith("```"):
            raw_content = raw_content.rsplit("```", 1)[0]
        raw_content = raw_content.strip()

        data = json.loads(raw_content)
        new_query = data.get("rewritten_query", rewritten_query)
        logger.info(f"查询改写: '{rewritten_query}' -> '{new_query}'")

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"查询改写解析失败，使用原查询: {e}")
        new_query = rewritten_query

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print(f"---node_query_rewrite 完成 (retry={retry_count + 1})---")

    return {
        "rewritten_query": new_query,
        "crag_retry_count": retry_count + 1,
    }