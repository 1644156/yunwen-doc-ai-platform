"""
评测脚本：运行 4 组实验对比 + RAGAS 指标
"""
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.core.logger import logger


def run_evaluation(max_cases: int = 0, ragas_sample: int = 20, use_ragas: bool = False):
    """
    :param max_cases: 限制每个实验的最大 case 数（0=全部）
    :param ragas_sample: RAGAS 评估的采样数（减少 LLM 调用）
    """
    # 加载 Golden Set
    golden_path = PROJECT_ROOT / "data" / "golden_set.jsonl"
    golden_set = []
    with open(golden_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                golden_set.append(json.loads(line))

    if max_cases > 0:
        golden_set = golden_set[:max_cases]

    print(f"Golden Set: {len(golden_set)} 条")
    print(f"RAGAS 采样数: {min(ragas_sample, len(golden_set))}")

    # 定义实验配置
    experiments = {
        "baseline": {"enable_routing": False, "enable_crag": False},
        "with_routing": {"enable_routing": True, "enable_crag": False},
        "with_crag": {"enable_routing": False, "enable_crag": True},
        "full": {"enable_routing": True, "enable_crag": True},
    }

    all_results = {}

    for exp_name, config in experiments.items():
        print(f"\n{'='*60}")
        print(f"实验: {exp_name} (config={config})")
        print(f"{'='*60}")

        predictions = []
        total_latency = 0
        success_count = 0
        fail_count = 0

        for i, case in enumerate(golden_set):
            question = case["question"]
            item_names = case.get("item_names", [])
            case_id = case.get("case_id", f"CASE-{i}")

            state = create_query_default_state(
                session_id=f"eval_{exp_name}_{i}",
                original_query=question,
                rewritten_query=question,
                item_names=item_names,
                is_stream=False,
                config=config,
            )

            start_time = time.time()
            try:
                result = query_app.invoke(state)
                latency = (time.time() - start_time) * 1000
                total_latency += latency

                answer = result.get("answer", "")
                reranked_docs = result.get("reranked_docs", [])
                contexts = [doc.get("text", "") for doc in reranked_docs]
                query_type = result.get("query_type", "")
                crag_decision = result.get("crag_decision", "")

                predictions.append({
                    "case_id": case_id,
                    "question": question,
                    "ground_truth": case.get("ground_truth", ""),
                    "answer": answer,
                    "contexts": contexts,
                    "query_type_used": query_type,
                    "crag_decision": crag_decision,
                    "latency_ms": latency,
                    "success": True,
                })
                success_count += 1
                status = "OK"
                if query_type:
                    status += f" [type={query_type}]"
                if crag_decision:
                    status += f" [crag={crag_decision}]"

            except Exception as e:
                latency = (time.time() - start_time) * 1000
                total_latency += latency
                fail_count += 1
                predictions.append({
                    "case_id": case_id,
                    "question": question,
                    "ground_truth": case.get("ground_truth", ""),
                    "answer": "",
                    "contexts": [],
                    "latency_ms": latency,
                    "success": False,
                    "error": str(e)[:200],
                })
                status = f"FAIL: {str(e)[:80]}"

            print(f"  [{i+1:3d}/{len(golden_set)}] {status} ({latency:.0f}ms)")

        # 计算基础统计
        avg_latency = total_latency / len(golden_set) if golden_set else 0
        success_rate = success_count / len(golden_set) if golden_set else 0

        answer_quality = sum(1 for p in predictions
                            if p["success"] and p["answer"] and len(p["answer"]) > 20)
        quality_rate = answer_quality / len(golden_set) if golden_set else 0

        refuse_count = sum(1 for p in predictions
                          if p["success"] and ("抱歉" in p.get("answer", "") or "没有找到" in p.get("answer", "")))

        stats = {
            "success_count": success_count,
            "fail_count": fail_count,
            "success_rate": success_rate,
            "quality_rate": quality_rate,
            "refuse_count": refuse_count,
            "avg_latency_ms": avg_latency,
            "total_latency_ms": total_latency,
        }

        # 计算评测指标
        print(f"\n  计算评测指标...")
        ragas_metrics = {}
        try:
            from app.eval.ragas_evaluator import evaluate_run
            # 采样：取前 ragas_sample 条有效预测
            valid_predictions = [p for p in predictions if p["success"] and p["answer"] and len(p["answer"]) > 10]
            valid_golden = [golden_set[i] for i, p in enumerate(predictions)
                           if p["success"] and p["answer"] and len(p["answer"]) > 10]

            sample_n = min(ragas_sample, len(valid_predictions))
            if sample_n > 0:
                # use_ragas=True 时使用 RAGAS（需要 LLM 额度），否则使用简化指标
                ragas_metrics = evaluate_run(valid_predictions[:sample_n], valid_golden[:sample_n], use_ragas=use_ragas)
                print(f"  评测指标 ({sample_n} 条采样):")
                for k, v in ragas_metrics.items():
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"  评测: 无有效预测，跳过")
        except Exception as e:
            print(f"  评测指标计算失败: {e}")

        stats["ragas"] = ragas_metrics

        all_results[exp_name] = {
            "config": config,
            "stats": stats,
            "predictions": predictions,
        }

        print(f"\n  成功率: {success_rate:.1%} ({success_count}/{len(golden_set)})")
        print(f"  有效答案率: {quality_rate:.1%}")
        print(f"  拒答次数: {refuse_count}")
        print(f"  平均延迟: {avg_latency:.0f}ms")

    # ========================
    # 保存结果
    # ========================
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "reports" / f"eval_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, result in all_results.items():
        with open(output_dir / f"{exp_name}_predictions.jsonl", "w", encoding="utf-8") as f:
            for p in result["predictions"]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 保存统计（含 RAGAS）
    summary = {}
    for exp_name, result in all_results.items():
        summary[exp_name] = result["stats"]

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ========================
    # 打印对比表
    # ========================
    print(f"\n{'='*100}")
    print(f"{'实验对比结果':^100}")
    print(f"{'='*100}")

    # 基础指标表
    print(f"\n{'--- 基础指标 ---':^100}")
    print(f"{'实验名':<20} {'成功率':>10} {'有效答案率':>12} {'拒答':>8} {'平均延迟(ms)':>14}")
    print(f"{'-'*70}")
    for exp_name, stats in summary.items():
        print(f"{exp_name:<20} {stats['success_rate']:>9.1%} {stats['quality_rate']:>11.1%} {stats['refuse_count']:>7d} {stats['avg_latency_ms']:>13.0f}")

    # 评测指标表
    metrics_available = any(stats.get("ragas") for stats in summary.values())
    if metrics_available:
        print(f"\n{'--- 评测指标 ---':^100}")
        metric_keys = ["answer_f1", "answer_coverage", "context_relevance", "answer_length_avg", "success_rate"]
        header = f"{'实验名':<20}" + "".join(f"{k:>18}" for k in metric_keys)
        print(header)
        print(f"{'-'*110}")
        for exp_name, stats in summary.items():
            m = stats.get("ragas", {})
            row = f"{exp_name:<20}" + "".join(f"{m.get(k, 0):>17.4f}" for k in metric_keys)
            print(row)

    print(f"\n结果保存在: {output_dir}")

    # ========================
    # 生成 Markdown 报告
    # ========================
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# RAG 评测报告\n\n")
        f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Golden Set**: {len(golden_set)} 条\n")
        f.write(f"**RAGAS 采样数**: {min(ragas_sample, len(golden_set))}\n\n")

        # 基础指标
        f.write(f"## 1. 基础指标对比\n\n")
        f.write(f"| 实验 | 成功率 | 有效答案率 | 拒答 | 平均延迟(ms) |\n")
        f.write(f"|---|---|---|---|---|\n")
        for exp_name, stats in summary.items():
            f.write(f"| {exp_name} | {stats['success_rate']:.1%} | {stats['quality_rate']:.1%} | {stats['refuse_count']} | {stats['avg_latency_ms']:.0f} |\n")

        # 评测指标
        if metrics_available:
            f.write(f"\n## 2. 评测指标对比\n\n")
            metric_keys = ["answer_f1", "answer_coverage", "context_relevance", "answer_length_avg", "success_rate"]
            f.write(f"| 实验 | " + " | ".join(metric_keys) + " |\n")
            f.write(f"|---" * (len(metric_keys) + 1) + "|\n")
            for exp_name, stats in summary.items():
                m = stats.get("ragas", {})
                vals = [f"{m.get(k, 0):.4f}" for k in metric_keys]
                f.write(f"| {exp_name} | " + " | ".join(vals) + " |\n")

        # 配置信息
        f.write(f"\n## 3. 实验配置\n\n")
        for exp_name, result in all_results.items():
            f.write(f"### {exp_name}\n\n")
            f.write(f"- 配置: `{json.dumps(result['config'])}`\n")
            f.write(f"- 成功: {result['stats']['success_count']}/{len(golden_set)}\n")
            f.write(f"- 失败: {result['stats']['fail_count']}\n\n")

    print(f"报告: {report_path}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=0, help="限制每个实验的最大 case 数（0=全部）")
    parser.add_argument("--ragas-sample", type=int, default=20, help="RAGAS 评估的采样数")
    parser.add_argument("--use-ragas", action="store_true", help="使用 RAGAS 指标（需要 LLM 额度）")
    args = parser.parse_args()
    run_evaluation(max_cases=args.max_cases, ragas_sample=args.ragas_sample, use_ragas=args.use_ragas)
