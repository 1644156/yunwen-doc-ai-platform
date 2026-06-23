import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from app.core.logger import logger

QUERY_TIMEOUT_SECONDS = 60.0
HISTORY_TIMEOUT_SECONDS = 5.0
DEFAULT_HISTORY_LIMIT = 10
MAX_HISTORY_LIMIT = 100


def _json_safe(value: Any) -> Any:
    """Convert values returned by databases/models into JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _history_message_to_dict(message: dict[str, Any]) -> dict[str, Any]:
    """Return the public, JSON-safe subset of a MongoDB chat message."""
    return {
        "id": str(message.get("_id", "")),
        "session_id": str(message.get("session_id", "")),
        "role": str(message.get("role", "")),
        "text": str(message.get("text", "")),
        "rewritten_query": str(message.get("rewritten_query", "")),
        "item_names": _json_safe(message.get("item_names") or []),
        "image_urls": _json_safe(message.get("image_urls") or []),
        "ts": _json_safe(message.get("ts")),
    }


def _extract_fallback_citations(result: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """Build citation-like metadata from reranked docs when citations are absent."""
    citations: list[dict[str, Any]] = []
    for index, doc in enumerate(result.get("reranked_docs", [])[:limit], start=1):
        if not isinstance(doc, dict):
            continue
        text = doc.get("text", "") or ""
        citations.append(
            {
                "index": index,
                "type": doc.get("source", ""),
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "score": doc.get("score", 0.0),
                "chunk_id": doc.get("chunk_id", ""),
                "preview": text[:100] + "..." if len(text) > 100 else text,
            }
        )
    return citations


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Preload heavy models and connections before serving MCP requests."""
    logger.info("yunwen MCP service starting; preloading models and connections")

    try:
        from app.lm.embedding_utils import get_bge_m3_ef

        get_bge_m3_ef()
    except Exception as exc:
        logger.warning(f"Embedding model preload failed: {exc}")

    try:
        from app.lm.reranker_utils import get_reranker_model

        get_reranker_model()
    except Exception as exc:
        logger.warning(f"Reranker model preload failed: {exc}")

    try:
        from app.clients.milvus_utils import get_milvus_client

        get_milvus_client()
    except Exception as exc:
        logger.warning(f"Milvus preload failed: {exc}")

    logger.info("yunwen MCP service ready")
    yield {}


mcp = FastMCP("yunwen_mcp", lifespan=lifespan)


@mcp.tool(
    name="yunwen_query_knowledge_base",
    annotations={
        "title": "Query yunwen Knowledge Base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def yunwen_query_knowledge_base(
    query: str,
    session_id: str = "",
    item_names: list[str] | None = None,
    enable_routing: bool = True,
    enable_crag: bool = True,
) -> dict[str, Any]:
    """Query the yunwen LangGraph RAG knowledge base.

    Args:
        query: User question to answer.
        session_id: Conversation id. A UUID is generated when empty.
        item_names: Reserved product names to pass into state. The current graph
            still runs item-name confirmation, so this does not guarantee a skip.
        enable_routing: Whether to enable adaptive query routing.
        enable_crag: Whether to enable CRAG correction/retrieval enhancement.

    Returns:
        A JSON-safe dict with success, answer, session_id, citations,
        image_urls, query_type, and crag_decision.
    """
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "query must not be empty"}

    from app.query_process.agent.main_graph import query_app
    from app.query_process.agent.state import create_query_default_state

    session_id = session_id or str(uuid.uuid4())
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        item_names=item_names or [],
        is_stream=False,
        config={"enable_routing": enable_routing, "enable_crag": enable_crag},
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(query_app.invoke, state),
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        citations = result.get("citations") or _extract_fallback_citations(result)
        return _json_safe(
            {
                "success": True,
                "answer": result.get("answer", ""),
                "session_id": session_id,
                "citations": citations,
                "image_urls": result.get("image_urls", []),
                "query_type": result.get("query_type", ""),
                "crag_decision": result.get("crag_decision", ""),
            }
        )
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"query timed out after {int(QUERY_TIMEOUT_SECONDS)} seconds",
            "session_id": session_id,
        }
    except Exception as exc:
        logger.exception(f"yunwen_query_knowledge_base failed: {exc}")
        return {"success": False, "error": str(exc), "session_id": session_id}


@mcp.tool(
    name="yunwen_get_chat_history",
    annotations={
        "title": "Get yunwen Chat History",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def yunwen_get_chat_history(
    session_id: str,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> dict[str, Any]:
    """Return recent JSON-safe chat messages for a yunwen session."""
    session_id = (session_id or "").strip()
    if not session_id:
        return {"success": False, "error": "session_id must not be empty"}

    limit = max(1, min(int(limit), MAX_HISTORY_LIMIT))
    from app.clients.mongo_history_utils import get_recent_messages

    try:
        messages = await asyncio.wait_for(
            asyncio.to_thread(get_recent_messages, session_id, limit),
            timeout=HISTORY_TIMEOUT_SECONDS,
        )
        items = [_history_message_to_dict(message) for message in messages]
        return {
            "success": True,
            "session_id": session_id,
            "count": len(items),
            "limit": limit,
            "messages": items,
        }
    except asyncio.TimeoutError:
        return {"success": False, "error": "chat history query timed out"}
    except Exception as exc:
        logger.exception(f"yunwen_get_chat_history failed: {exc}")
        return {"success": False, "error": str(exc)}


@mcp.tool(
    name="yunwen_clear_chat_history",
    annotations={
        "title": "Clear yunwen Chat History",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def yunwen_clear_chat_history(session_id: str) -> dict[str, Any]:
    """Clear all chat messages for a yunwen session."""
    session_id = (session_id or "").strip()
    if not session_id:
        return {"success": False, "error": "session_id must not be empty"}

    from app.clients.mongo_history_utils import clear_history

    try:
        deleted_count = await asyncio.wait_for(
            asyncio.to_thread(clear_history, session_id),
            timeout=HISTORY_TIMEOUT_SECONDS,
        )
        return {
            "success": True,
            "session_id": session_id,
            "deleted_count": deleted_count,
        }
    except asyncio.TimeoutError:
        return {"success": False, "error": "clear chat history timed out"}
    except Exception as exc:
        logger.exception(f"yunwen_clear_chat_history failed: {exc}")
        return {"success": False, "error": str(exc)}


@mcp.tool(
    name="yunwen_get_service_status",
    annotations={
        "title": "Get yunwen MCP Service Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def yunwen_get_service_status() -> dict[str, Any]:
    """Return health information for the yunwen MCP service dependencies."""
    components: dict[str, str] = {"mcp_server": "ok"}

    try:
        from app.lm.embedding_utils import get_bge_m3_ef

        components["embedding_model"] = "loaded" if get_bge_m3_ef() else "not_loaded"
    except Exception as exc:
        components["embedding_model"] = f"error: {exc}"

    try:
        from app.lm.reranker_utils import get_reranker_model

        components["reranker_model"] = "loaded" if get_reranker_model() else "not_loaded"
    except Exception as exc:
        components["reranker_model"] = f"error: {exc}"

    try:
        from app.clients.milvus_utils import get_milvus_client

        components["milvus"] = "connected" if get_milvus_client() else "disconnected"
    except Exception as exc:
        components["milvus"] = f"error: {exc}"

    try:
        from app.clients.mongo_history_utils import get_history_mongo_tool

        components["mongodb"] = "connected" if get_history_mongo_tool() else "disconnected"
    except Exception as exc:
        components["mongodb"] = f"error: {exc}"

    ok_values = {"ok", "loaded", "connected"}
    return {
        "success": all(value in ok_values for value in components.values()),
        "components": components,
    }
