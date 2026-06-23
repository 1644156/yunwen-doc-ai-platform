import sys
from app.conf.milvus_config import milvus_config
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.core.logger import logger


def node_crag_re_retrieve(state):
    """
    节点功能：CRAG 重检索
    使用改写后的查询重新执行向量检索，
    覆盖 embedding_chunks 后流入 RRF。
    """
    print("---node_crag_re_retrieve 重检索---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names", [])

    # 生成向量
    embeddings = generate_embeddings([rewritten_query])

    # 构建过滤条件
    item_name_str = ', '.join(f'"{item}"' for item in item_names)
    expr = f"item_name in [{item_name_str}]" if item_names else None

    # 混合检索
    hybrid_search_requests = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=expr
    )

    milvus_client = get_milvus_client()
    embedding_chunks = []
    if milvus_client:
        try:
            resp = hybrid_search(
                client=milvus_client,
                collection_name=milvus_config.chunks_collection,
                reqs=hybrid_search_requests,
                ranker_weights=(0.9, 0.1),
                norm_score=True,
                limit=5,
                output_fields=["chunk_id", "content", "file_title", "title", "parent_title", "item_name"]
            )
            embedding_chunks = resp[0] if resp else []
        except Exception as e:
            logger.error(f"CRAG 重检索失败: {e}")

    logger.info(f"CRAG 重检索完成，结果数量: {len(embedding_chunks)}")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print(f"---node_crag_re_retrieve 完成: {len(embedding_chunks)} 条---")

    # 覆盖 embedding_chunks，清空其他检索源避免旧数据干扰
    return {
        "embedding_chunks": embedding_chunks,
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
        "bm25_chunks": [],
        "graph_rag_chunks": [],
    }