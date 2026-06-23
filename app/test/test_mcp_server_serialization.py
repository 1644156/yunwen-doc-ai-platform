from bson import ObjectId

from app.mcp_server import _history_message_to_dict


def test_history_message_to_dict_is_json_safe():
    message = {
        "_id": ObjectId("64f1a1f1a1f1a1f1a1f1a1f1"),
        "session_id": "session-1",
        "role": "assistant",
        "text": "回答",
        "rewritten_query": "重写问题",
        "item_names": ["HAK 180"],
        "image_urls": ["http://example.com/a.jpg"],
        "ts": 123.45,
        "internal": {"should": "not leak"},
    }

    result = _history_message_to_dict(message)

    assert result == {
        "id": "64f1a1f1a1f1a1f1a1f1a1f1",
        "session_id": "session-1",
        "role": "assistant",
        "text": "回答",
        "rewritten_query": "重写问题",
        "item_names": ["HAK 180"],
        "image_urls": ["http://example.com/a.jpg"],
        "ts": 123.45,
    }
