from typing_extensions import TypedDict
from typing import List
import copy

class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档
    bm25_chunks: list  # BM25 稀疏检索回来的切片
    graph_rag_chunks: list  # 知识图谱检索回来的切片

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记

    # 路由字段
    query_type: str  # 查询类型: factual | reasoning | temporal | ambiguous
    routing_confidence: float  # 路由置信度 0.0-1.0

    # CRAG 字段
    crag_decision: str  # CRAG 决策: correct | ambiguous | incorrect | all_fail
    crag_score: float  # CRAG 评估分数 0.0-1.0
    crag_retry_count: int  # CRAG 重试次数

    # 功能开关
    config: dict  # {"enable_routing": True, "enable_crag": True}


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "bm25_chunks": [],
    "graph_rag_chunks": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "query_type": "",
    "routing_confidence": 0.0,
    "crag_decision": "",
    "crag_score": 0.0,
    "crag_retry_count": 0,
    "config": {"enable_routing": True, "enable_crag": True},
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


# ========================
# ✅ 状态复制函数（你要的）
# ========================
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 测试
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
        is_stream=False
    )
    print("初始化状态：", state)

    # 复制状态
    new_state = copy_query_state(
        state,
        original_query="修改后的问题"
    )
    print("复制后的状态：", new_state)