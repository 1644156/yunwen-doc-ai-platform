import sys
from pymilvus import AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client
from app.core.logger import logger


def node_search_bm25(state):
    """
    节点功能：BM25 稀疏检索
    使用 BGE-M3 的稀疏向量进行纯词汇匹配检索，
    适用于事实型查询（短句、明确实体）。
    """
    print("---node_search_bm25 稀疏检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names", [])

    # 生成稀疏向量
    embeddings = generate_embeddings([rewritten_query])
    sparse_vector = embeddings['sparse'][0]

    # 构建 item_name 过滤条件
    item_name_str = ', '.join(f'"{item}"' for item in item_names)
    expr = f"item_name in [{item_name_str}]" if item_names else None

    # 构建稀疏检索请求（仅使用 sparse_vector 字段）
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        expr=expr,
        limit=5
    )

    # 执行检索
    milvus_client = get_milvus_client()
    if milvus_client is None:
        logger.error("Milvus 客户端不可用")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"bm25_chunks": []}

    try:
        resp = milvus_client.hybrid_search(
            collection_name=milvus_config.chunks_collection,
            reqs=[sparse_req],
            ranker=WeightedRanker(1.0, norm_score=True),
            limit=5,
            output_fields=["chunk_id", "content", "file_title", "title", "parent_title", "item_name"]
        )
        bm25_chunks = resp[0] if resp else []
    except Exception as e:
        logger.error(f"BM25 稀疏检索失败: {e}")
        bm25_chunks = []

    logger.info(f"BM25 稀疏检索完成，结果数量: {len(bm25_chunks)}")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---node_search_bm25 稀疏检索 处理结束---")

    return {"bm25_chunks": bm25_chunks}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_state = {
        "session_id": "test_bm25_001",
        "rewritten_query": "HAK 180 烫金机功率",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }
    result = node_search_bm25(test_state)
    chunks = result.get("bm25_chunks", [])
    print(f"检索到 {len(chunks)} 条结果")
    for i, c in enumerate(chunks[:3], 1):
        content = (c.get("entity", {}).get("content", ""))[:50]
        print(f"  {i}. {content}...")