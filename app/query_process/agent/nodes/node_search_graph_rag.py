import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def node_search_graph_rag(state):
    """
    节点功能：知识图谱检索（GraphRAG）
    通过 Neo4j 查询实体关系，补充结构化知识。
    Neo4j 不可用时优雅降级返回空列表。
    """
    print("---node_search_graph_rag 知识图谱检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query", "")
    graph_rag_chunks = []

    try:
        from app.clients.neo4j_utils import get_neo4j_driver, search_entity_relations
        driver = get_neo4j_driver()

        for item_name in item_names:
            relations = search_entity_relations(driver, item_name, limit=5)
            for i, rel in enumerate(relations):
                # 将知识图谱关系格式化为与向量检索相同的 chunk 结构
                content = f"实体「{rel.get('source', '')}」通过关系「{rel.get('relation', '')}」关联到「{rel.get('target', '')}」"
                if rel.get("description"):
                    content += f"。说明：{rel['description']}"

                graph_rag_chunks.append({
                    "chunk_id": f"kg_{item_name}_{i}",
                    "id": f"kg_{item_name}_{i}",
                    "distance": 0.5,  # 默认中等相关度
                    "entity": {
                        "chunk_id": f"kg_{item_name}_{i}",
                        "content": content,
                        "file_title": "知识图谱",
                        "title": f"{item_name} - 关系查询",
                        "parent_title": "",
                        "item_name": item_name,
                    }
                })

        logger.info(f"知识图谱检索完成，结果数量: {len(graph_rag_chunks)}")

    except ImportError as e:
        logger.warning(f"Neo4j 依赖不可用，跳过知识图谱检索: {e}")
    except Exception as e:
        logger.warning(f"知识图谱检索失败（Neo4j 可能未启动），跳过: {e}")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print(f"---node_search_graph_rag 完成，结果: {len(graph_rag_chunks)}---")

    return {"graph_rag_chunks": graph_rag_chunks}