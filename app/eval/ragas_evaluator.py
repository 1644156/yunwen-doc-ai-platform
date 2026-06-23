"""
评测指标计算
包含 RAGAS 指标（需要 LLM）和简化指标（无需 LLM）。
"""
import math
import re
from typing import List, Dict
from app.core.logger import logger

METRIC_NAMES = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


def _tokenize(text: str) -> List[str]:
    """简单分词：按空格和标点分割，转小写"""
    tokens = re.findall(r'[\w一-鿿]+', text.lower())
    return tokens


def _compute_token_overlap(answer_tokens: set, gt_tokens: set) -> float:
    """计算 token 级别的 F1 分数"""
    if not answer_tokens or not gt_tokens:
        return 0.0
    common = answer_tokens & gt_tokens
    if not common:
        return 0.0
    precision = len(common) / len(answer_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_simple_metrics(predictions: List[Dict], golden_set: List[Dict]) -> Dict[str, float]:
    """
    计算简化指标（无需 LLM 调用）：
    - answer_f1: 答案与 ground_truth 的 token F1
    - answer_coverage: ground_truth 关键词在答案中的覆盖率
    - context_relevance: 上下文与问题的 token 重叠度
    - answer_length_avg: 平均答案长度
    - success_rate: 有效答案率
    """
    if not predictions or not golden_set:
        return {}

    valid_pairs = []
    for p, g in zip(predictions, golden_set):
        answer = p.get("answer", "").strip()
        gt = g.get("ground_truth", "").strip()
        if answer and gt and len(answer) > 5:
            valid_pairs.append((p, g))

    if not valid_pairs:
        return {"answer_f1": 0.0, "answer_coverage": 0.0, "context_relevance": 0.0,
                "answer_length_avg": 0.0, "success_rate": 0.0}

    f1_scores = []
    coverage_scores = []
    context_scores = []
    answer_lengths = []

    for p, g in valid_pairs:
        answer = p.get("answer", "")
        gt = g.get("ground_truth", "")
        question = p.get("question", "")
        contexts = p.get("contexts", [])

        # Answer F1
        ans_tokens = set(_tokenize(answer))
        gt_tokens = set(_tokenize(gt))
        f1 = _compute_token_overlap(ans_tokens, gt_tokens)
        f1_scores.append(f1)

        # Answer Coverage (GT 关键词在答案中的覆盖率)
        if gt_tokens:
            covered = len(ans_tokens & gt_tokens) / len(gt_tokens)
            coverage_scores.append(covered)

        # Context Relevance (上下文与问题的重叠度)
        if contexts:
            ctx_text = " ".join(contexts)
            ctx_tokens = set(_tokenize(ctx_text))
            q_tokens = set(_tokenize(question))
            ctx_rel = _compute_token_overlap(ctx_tokens, q_tokens)
            context_scores.append(ctx_rel)

        answer_lengths.append(len(answer))

    return {
        "answer_f1": sum(f1_scores) / len(f1_scores),
        "answer_coverage": sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0,
        "context_relevance": sum(context_scores) / len(context_scores) if context_scores else 0.0,
        "answer_length_avg": sum(answer_lengths) / len(answer_lengths),
        "success_rate": len(valid_pairs) / len(predictions) if predictions else 0.0,
    }


def _build_ragas_llm():
    """构建 RAGAS 使用的 LLM（复用项目 DashScope 配置）"""
    from ragas.llms import LangchainLLMWrapper
    from app.lm.lm_utils import get_llm_client
    llm = get_llm_client(json_mode=False)
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """构建 RAGAS 使用的 Embedding（复用项目 BGE-M3）"""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings
    from app.lm.embedding_utils import generate_embeddings

    class BGEEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return generate_embeddings(texts)["dense"]

        def embed_query(self, text: str) -> List[float]:
            return generate_embeddings([text])["dense"][0]

    return LangchainEmbeddingsWrapper(BGEEmbeddings())


def evaluate_run(predictions: List[Dict], golden_set: List[Dict],
                 use_ragas: bool = False) -> Dict[str, float]:
    """
    计算评测指标

    :param use_ragas: 是否使用 RAGAS（需要 LLM 额度）。默认使用简化指标。
    """
    # 始终计算简化指标
    simple_metrics = compute_simple_metrics(predictions, golden_set)

    if not use_ragas:
        logger.info(f"简化指标: {simple_metrics}")
        return simple_metrics

    # 尝试 RAGAS
    try:
        ragas_metrics = _evaluate_ragas(predictions, golden_set)
        # 合并简化指标和 RAGAS 指标
        simple_metrics.update(ragas_metrics)
        return simple_metrics
    except Exception as e:
        logger.warning(f"RAGAS 评估失败，使用简化指标: {e}")
        return simple_metrics


def _evaluate_ragas(predictions: List[Dict], golden_set: List[Dict]) -> Dict[str, float]:
    """RAGAS 评估（需要 LLM 额度）"""
    from ragas import evaluate
    from ragas.metrics import (
        context_precision, context_recall, faithfulness,
        answer_relevancy, answer_correctness
    )
    from datasets import Dataset

    valid_pairs = []
    for p, g in zip(predictions, golden_set):
        answer = p.get("answer", "").strip()
        gt = g.get("ground_truth", "").strip()
        if answer and gt and len(answer) > 5:
            valid_pairs.append((p, g))

    if not valid_pairs:
        logger.warning("RAGAS: 无有效预测结果")
        return {}

    data = {
        "question": [p["question"] for p, g in valid_pairs],
        "answer": [p.get("answer", "") for p, g in valid_pairs],
        "contexts": [p.get("contexts", [""]) for p, g in valid_pairs],
        "ground_truth": [g.get("ground_truth", "") for p, g in valid_pairs],
    }
    dataset = Dataset.from_dict(data)

    # 使用项目自身的 LLM 和 Embedding
    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = result.to_pandas()
    ragas_dict = {}
    for m in METRIC_NAMES:
        if m in df.columns:
            val = df[m].dropna()
            ragas_dict[m] = float(val.mean()) if len(val) > 0 else 0.0

    logger.info(f"RAGAS 评测完成 ({len(valid_pairs)} 条): {ragas_dict}")
    return ragas_dict