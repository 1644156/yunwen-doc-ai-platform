"""
查询路由分类器测试
"""
import json
import pytest
from unittest.mock import MagicMock, patch


def make_mock_llm_response(content: str):
    """创建 mock LLM 响应"""
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


class TestQueryRouting:
    """测试 node_query_routing 节点"""

    @patch("app.query_process.agent.nodes.node_query_routing.get_llm_client")
    @patch("app.query_process.agent.nodes.node_query_routing.load_prompt")
    def test_factual_classification(self, mock_load_prompt, mock_get_llm):
        mock_load_prompt.return_value = "test prompt"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = make_mock_llm_response(
            json.dumps({"query_type": "factual", "confidence": 0.9, "reason": "明确事实询问"})
        )
        mock_get_llm.return_value = mock_llm

        from app.query_process.agent.nodes.node_query_routing import node_query_routing
        state = {
            "session_id": "test_001",
            "rewritten_query": "HAK 180 的功率是多少",
            "item_names": ["HAK 180"],
            "is_stream": False,
        }
        result = node_query_routing(state)
        assert result["query_type"] == "factual"
        assert result["routing_confidence"] == 0.9

    @patch("app.query_process.agent.nodes.node_query_routing.get_llm_client")
    @patch("app.query_process.agent.nodes.node_query_routing.load_prompt")
    def test_reasoning_classification(self, mock_load_prompt, mock_get_llm):
        mock_load_prompt.return_value = "test prompt"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = make_mock_llm_response(
            json.dumps({"query_type": "reasoning", "confidence": 0.85, "reason": "对比分析"})
        )
        mock_get_llm.return_value = mock_llm

        from app.query_process.agent.nodes.node_query_routing import node_query_routing
        state = {
            "session_id": "test_002",
            "rewritten_query": "H3C和华为路由器哪个好",
            "item_names": ["H3C路由器"],
            "is_stream": False,
        }
        result = node_query_routing(state)
        assert result["query_type"] == "reasoning"

    @patch("app.query_process.agent.nodes.node_query_routing.get_llm_client")
    @patch("app.query_process.agent.nodes.node_query_routing.load_prompt")
    def test_invalid_type_defaults_to_ambiguous(self, mock_load_prompt, mock_get_llm):
        mock_load_prompt.return_value = "test prompt"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = make_mock_llm_response(
            json.dumps({"query_type": "invalid_type", "confidence": 0.8, "reason": "test"})
        )
        mock_get_llm.return_value = mock_llm

        from app.query_process.agent.nodes.node_query_routing import node_query_routing
        state = {
            "session_id": "test_003",
            "rewritten_query": "test query",
            "item_names": [],
            "is_stream": False,
        }
        result = node_query_routing(state)
        assert result["query_type"] == "ambiguous"

    @patch("app.query_process.agent.nodes.node_query_routing.get_llm_client")
    @patch("app.query_process.agent.nodes.node_query_routing.load_prompt")
    def test_parse_error_defaults_to_ambiguous(self, mock_load_prompt, mock_get_llm):
        mock_load_prompt.return_value = "test prompt"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = make_mock_llm_response("not valid json")
        mock_get_llm.return_value = mock_llm

        from app.query_process.agent.nodes.node_query_routing import node_query_routing
        state = {
            "session_id": "test_004",
            "rewritten_query": "test",
            "item_names": [],
            "is_stream": False,
        }
        result = node_query_routing(state)
        assert result["query_type"] == "ambiguous"
        assert result["routing_confidence"] == 0.0


class TestRRFDynamicSources:
    """测试 RRF 动态源列表"""

    def test_rrf_with_all_sources(self):
        from app.query_process.agent.nodes.node_rrf import step_3_reciprocal_rank_fusion

        source1 = [{"id": "c1", "entity": {"chunk_id": "c1"}}, {"id": "c2", "entity": {"chunk_id": "c2"}}]
        source2 = [{"id": "c1", "entity": {"chunk_id": "c1"}}, {"id": "c3", "entity": {"chunk_id": "c3"}}]
        source3 = [{"id": "c4", "entity": {"chunk_id": "c4"}}]

        result = step_3_reciprocal_rank_fusion([
            (source1, 1.0),
            (source2, 1.0),
            (source3, 0.8),
        ], top_k=5)

        # c1 应该排第一（出现在两个源中）
        assert len(result) >= 1
        assert result[0].get("id") == "c1" or result[0].get("entity", {}).get("chunk_id") == "c1"

    def test_rrf_with_single_source(self):
        from app.query_process.agent.nodes.node_rrf import step_3_reciprocal_rank_fusion

        source = [{"id": "c1", "entity": {"chunk_id": "c1"}}, {"id": "c2", "entity": {"chunk_id": "c2"}}]
        result = step_3_reciprocal_rank_fusion([(source, 1.0)], top_k=5)
        assert len(result) == 2

    def test_rrf_with_empty_sources(self):
        from app.query_process.agent.nodes.node_rrf import step_3_reciprocal_rank_fusion
        result = step_3_reciprocal_rank_fusion([], top_k=5)
        assert result == []


class TestCRAGDecision:
    """测试 CRAG 决策逻辑"""

    def test_correct_decision(self):
        from app.query_process.agent.nodes.node_crag import node_crag
        state = {
            "session_id": "test_crag_001",
            "reranked_docs": [
                {"score": 0.9, "text": "high quality doc"},
                {"score": 0.8, "text": "another good doc"},
            ],
            "is_stream": False,
        }
        result = node_crag(state)
        assert result["crag_decision"] == "correct"
        assert result["crag_score"] >= 0.7

    def test_ambiguous_decision(self):
        from app.query_process.agent.nodes.node_crag import node_crag
        state = {
            "session_id": "test_crag_002",
            "reranked_docs": [
                {"score": 0.5, "text": "medium quality"},
                {"score": 0.4, "text": "somewhat relevant"},
            ],
            "is_stream": False,
        }
        result = node_crag(state)
        assert result["crag_decision"] == "ambiguous"

    def test_incorrect_decision(self):
        from app.query_process.agent.nodes.node_crag import node_crag
        state = {
            "session_id": "test_crag_003",
            "reranked_docs": [
                {"score": 0.2, "text": "low quality"},
                {"score": 0.1, "text": "irrelevant"},
            ],
            "is_stream": False,
        }
        result = node_crag(state)
        assert result["crag_decision"] == "incorrect"

    def test_all_fail_on_empty(self):
        from app.query_process.agent.nodes.node_crag import node_crag
        state = {
            "session_id": "test_crag_004",
            "reranked_docs": [],
            "is_stream": False,
        }
        result = node_crag(state)
        assert result["crag_decision"] == "all_fail"