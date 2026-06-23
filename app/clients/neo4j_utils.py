import os
from neo4j import GraphDatabase

_neo4j_driver = None
def get_neo4j_driver() -> GraphDatabase:
    """
    获取 Neo4j 驱动实例
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    return _neo4j_driver


def search_entity_relations(driver, entity_name: str, limit: int = 5) -> list:
    """
    查询 Neo4j 中与给定实体名称相关的关系

    :param driver: Neo4j 驱动实例
    :param entity_name: 实体名称（如产品名）
    :param limit: 返回结果数量限制
    :return: 关系列表 [{"source": str, "relation": str, "target": str, "description": str}]
    """
    query = """
    MATCH (n)-[r]-(m)
    WHERE n.name CONTAINS $name
    RETURN n.name AS source, type(r) AS relation, m.name AS target,
           r.description AS description
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, name=entity_name, limit=limit)
        return [dict(record) for record in result]