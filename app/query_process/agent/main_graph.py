from langgraph.graph import StateGraph, END

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_answer_refuse import node_answer_refuse
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.nodes.node_query_routing import node_query_routing
from app.query_process.agent.nodes.node_search_bm25 import node_search_bm25
from app.query_process.agent.nodes.node_search_graph_rag import node_search_graph_rag
from app.query_process.agent.nodes.node_crag import node_crag
from app.query_process.agent.nodes.node_query_rewrite import node_query_rewrite
from app.query_process.agent.nodes.node_crag_re_retrieve import node_crag_re_retrieve
from app.query_process.agent.nodes.node_crag_supplement import node_crag_supplement
from app.query_process.agent.state import QueryGraphState

builder = StateGraph(QueryGraphState)

# ========================
# 注册所有节点
# ========================
# 原有节点
builder.add_node("node_item_name_confirm", node_item_name_confirm)
builder.add_node("node_search_embedding", node_search_embedding)
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_rrf", node_rrf)
builder.add_node("node_rerank", node_rerank)
builder.add_node("node_answer_output", node_answer_output)

# 路由节点
builder.add_node("node_query_routing", node_query_routing)
builder.add_node("node_search_bm25", node_search_bm25)
builder.add_node("node_search_graph_rag", node_search_graph_rag)

# CRAG 节点
builder.add_node("node_crag", node_crag)
builder.add_node("node_query_rewrite", node_query_rewrite)
builder.add_node("node_crag_re_retrieve", node_crag_re_retrieve)
builder.add_node("node_crag_supplement", node_crag_supplement)
builder.add_node("node_answer_refuse", node_answer_refuse)

# CRAG 补充路径和重试路径使用独立的 rrf/rerank 实例（同函数不同名）
builder.add_node("supplement_rrf", node_rrf)
builder.add_node("supplement_rerank", node_rerank)
builder.add_node("retry_rrf", node_rrf)
builder.add_node("retry_rerank", node_rerank)

# ========================
# 入口
# ========================
builder.set_entry_point("node_item_name_confirm")


# ========================
# 条件边 1: item_name_confirm 之后
# ========================
def route_after_node_item_name_confirm(state: QueryGraphState):
    if state.get('answer'):
        return "node_answer_output"
    config = state.get("config", {})
    if config.get("enable_routing", True):
        return "node_query_routing"
    else:
        # 不启用路由，走原有三路并行
        return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"


builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_node_item_name_confirm,
    {
        "node_answer_output": "node_answer_output",
        "node_query_routing": "node_query_routing",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp",
    }
)


# ========================
# 条件边 2: query_routing 之后（根据查询类型选择检索策略）
# ========================
def route_after_query_routing(state: QueryGraphState):
    query_type = state.get("query_type", "ambiguous")
    if query_type == "factual":
        return ["node_search_bm25", "node_web_search_mcp"]
    elif query_type == "reasoning":
        return ["node_search_embedding", "node_search_graph_rag"]
    elif query_type == "temporal":
        return ["node_search_embedding", "node_web_search_mcp"]
    else:  # ambiguous
        return ["node_search_embedding_hyde", "node_search_embedding", "node_web_search_mcp"]


builder.add_conditional_edges(
    "node_query_routing",
    route_after_query_routing,
    {
        "node_search_bm25": "node_search_bm25",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_search_graph_rag": "node_search_graph_rag",
        "node_web_search_mcp": "node_web_search_mcp",
    }
)

# ========================
# 主路径: 所有检索节点 -> RRF -> Rerank
# ========================
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")
builder.add_edge("node_search_bm25", "node_rrf")
builder.add_edge("node_search_graph_rag", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")


# ========================
# 条件边 3: rerank 之后（CRAG 评估 或 直接生成）
# ========================
def route_after_rerank(state: QueryGraphState):
    config = state.get("config", {})
    if config.get("enable_crag", True):
        return "node_crag"
    else:
        return "node_answer_output"


builder.add_conditional_edges(
    "node_rerank",
    route_after_rerank,
    {
        "node_crag": "node_crag",
        "node_answer_output": "node_answer_output",
    }
)


# ========================
# 条件边 4: CRAG 决策之后
# ========================
def route_after_crag(state: QueryGraphState):
    decision = state.get("crag_decision", "correct")
    retry_count = state.get("crag_retry_count", 0)

    if decision == "correct":
        return "node_answer_output"
    elif decision == "ambiguous":
        return "node_crag_supplement"
    elif decision == "incorrect" and retry_count < 1:  # 最多重试 1 次，避免超时
        return "node_query_rewrite"
    else:  # all_fail 或重试耗尽
        return "node_answer_refuse"


builder.add_conditional_edges(
    "node_crag",
    route_after_crag,
    {
        "node_answer_output": "node_answer_output",
        "node_crag_supplement": "node_crag_supplement",
        "node_query_rewrite": "node_query_rewrite",
        "node_answer_refuse": "node_answer_refuse",
    }
)

# ========================
# CRAG 补充路径: supplement -> supplement_rrf -> supplement_rerank -> answer_output
# （独立的 rrf/rerank 实例，不经过 CRAG 再评估）
# ========================
builder.add_edge("node_crag_supplement", "supplement_rrf")
builder.add_edge("supplement_rrf", "supplement_rerank")
builder.add_edge("supplement_rerank", "node_answer_output")

# ========================
# CRAG 重试路径: query_rewrite -> crag_re_retrieve -> retry_rrf -> retry_rerank -> crag
# （独立的 rrf/rerank 实例，重新进入 CRAG 评估）
# ========================
builder.add_edge("node_query_rewrite", "node_crag_re_retrieve")
builder.add_edge("node_crag_re_retrieve", "retry_rrf")
builder.add_edge("retry_rrf", "retry_rerank")
builder.add_edge("retry_rerank", "node_crag")

# ========================
# 终点
# ========================
builder.add_edge("node_answer_output", END)
builder.add_edge("node_answer_refuse", END)

# ========================
# 编译图
# ========================
query_app = builder.compile()