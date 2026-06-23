import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

# CRAG 决策阈值（根据评测调优：放宽 correct 阈值，收紧 incorrect 阈值）
CRAG_CORRECT_THRESHOLD = 0.5
CRAG_INCORRECT_THRESHOLD = 0.2


def node_crag(state):
    """
    节点功能：CRAG 检索质量评估
    评估重排序后的文档质量，决定后续处理路径：
    - correct (score >= 0.7): 直接生成答案
    - ambiguous (0.3 <= score < 0.7): 补充搜索后生成
    - incorrect (score < 0.3): 改写查询重试
    - all_fail: 拒答
    """
    print("---node_crag 检索质量评估---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    reranked_docs = state.get("reranked_docs", [])

    # 无检索结果 -> all_fail
    if not reranked_docs:
        logger.info("CRAG: 无检索结果，决策为 all_fail")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"crag_decision": "all_fail", "crag_score": 0.0}

    # 过滤空内容 chunk（只有标题没有正文的 chunk 不应参与质量评估）
    MIN_CONTENT_LENGTH = 20
    valid_docs = [doc for doc in reranked_docs if len(doc.get("text", "").strip()) >= MIN_CONTENT_LENGTH]

    if not valid_docs:
        logger.info("CRAG: 所有 chunk 内容为空，决策为 all_fail")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"crag_decision": "all_fail", "crag_score": 0.0}

    # 取 top-3 的 rerank 分数计算平均分（仅计算有实际内容的 chunk）
    top_scores = []
    for doc in valid_docs[:3]:
        score = doc.get("score", 0.0)
        if score is not None:
            top_scores.append(float(score))

    if not top_scores:
        avg_score = 0.0
    else:
        avg_score = sum(top_scores) / len(top_scores)

    # 决策逻辑
    if avg_score >= CRAG_CORRECT_THRESHOLD:
        decision = "correct"
    elif avg_score >= CRAG_INCORRECT_THRESHOLD:
        decision = "ambiguous"
    else:
        decision = "incorrect"

    logger.info(f"CRAG 决策: {decision}, avg_score={avg_score:.3f}, top_scores={top_scores}")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print(f"---node_crag 完成: {decision} (score={avg_score:.3f})---")

    return {
        "crag_decision": decision,
        "crag_score": avg_score,
    }