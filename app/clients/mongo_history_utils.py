import os
from typing import List, Dict, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from dotenv import load_dotenv
from app.core.logger import logger
from app.utils.circuit_breaker import mongodb_circuit_breaker

load_dotenv()


class HistoryMongoTool:
    """MongoDB 历史对话记录读写工具类"""

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])
            logger.info(f"MongoDB 连接成功: {self.db_name}")
        except Exception as e:
            logger.error(f"MongoDB 连接失败: {e}")
            raise


# 单例
_history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool


# 模块加载时预初始化
try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    logger.warning(f"MongoDB 预初始化失败，将在首次调用时重试: {e}")


def clear_history(session_id: str) -> int:
    """清空指定会话的所有历史对话记录（带熔断保护）"""
    def _clear():
        mongo_tool = get_history_mongo_tool()
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        return result.deleted_count

    try:
        count = mongodb_circuit_breaker.call(_clear)
        logger.info(f"已删除 {count} 条记录 (session={session_id})")
        return count
    except Exception as e:
        logger.error(f"清空历史记录失败: {e}")
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None
) -> str:
    """
    写入/更新单条会话记录（带熔断保护）
    :param message_id: 有值则更新，无值则新增
    :return: 记录 ID
    """
    def _save():
        ts = datetime.now().timestamp()
        document = {
            "session_id": session_id,
            "role": role,
            "text": text,
            "rewritten_query": rewritten_query or "",
            "item_names": item_names,
            "image_urls": image_urls,
            "ts": ts
        }

        mongo_tool = get_history_mongo_tool()
        if message_id:
            mongo_tool.chat_message.update_one(
                {"_id": ObjectId(message_id)},
                {"$set": document}
            )
            return message_id
        else:
            result = mongo_tool.chat_message.insert_one(document)
            return str(result.inserted_id)

    try:
        return mongodb_circuit_breaker.call(_save)
    except Exception as e:
        logger.error(f"保存聊天记录失败: {e}")
        return ""


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    批量更新历史记录的商品名称（带熔断保护）
    仅更新 item_names 为空/不存在/None 的记录，避免覆盖已有数据
    """
    def _update():
        mongo_tool = get_history_mongo_tool()
        object_ids = [ObjectId(i) for i in ids]
        result = mongo_tool.chat_message.update_many(
            {
                "_id": {"$in": object_ids},
                "$or": [
                    {"item_names": {"$exists": False}},
                    {"item_names": []},
                    {"item_names": None}
                ]
            },
            {"$set": {"item_names": item_names}}
        )
        return result.modified_count

    try:
        count = mongodb_circuit_breaker.call(_update)
        logger.info(f"已更新 {count} 条记录的 item_names")
        return count
    except Exception as e:
        logger.error(f"更新 item_names 失败: {e}")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """查询指定会话的最近 N 条记录，按时间正序（带熔断保护）"""
    def _query():
        mongo_tool = get_history_mongo_tool()
        cursor = mongo_tool.chat_message.find(
            {"session_id": session_id}
        ).sort("ts", ASCENDING).limit(limit)
        return list(cursor)

    try:
        return mongodb_circuit_breaker.call(_query)
    except Exception as e:
        logger.error(f"查询历史记录失败: {e}")
        return []


if __name__ == "__main__":
    sid = "test_session"
    save_chat_message(sid, "user", "你好")
    save_chat_message(sid, "assistant", "你好！有什么可以帮您？")
    messages = get_recent_messages(sid, limit=5)
    print(f"查询到 {len(messages)} 条记录")
    for m in messages:
        print(f"  [{m['role']}] {m['text']}")
