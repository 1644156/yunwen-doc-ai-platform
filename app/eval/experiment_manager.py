"""
实验管理器
加载 YAML 配置，批量运行 pipeline，收集结果并生成报告。
"""
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from app.core.logger import logger


class ExperimentManager:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载实验 YAML 配置"""
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("experiment", data)

    def load_golden_set(self, path: str) -> List[Dict]:
        """加载 Golden Set"""
        cases = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        return cases

    def run_single_pipeline(self, query: str, item_names: List[str],
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单次查询 pipeline

        :param query: 查询文本
        :param item_names: 已知的产品名（跳过 item_name_confirm）
        :param config: pipeline 配置 {"enable_routing": bool, "enable_crag": bool}
        :return: {"answer": str, "contexts": [str], "reranked_docs": list}
        """
        from app.query_process.agent.main_graph import query_app
        from app.query_process.agent.state import create_query_default_state

        state = create_query_default_state(
            session_id=f"eval_{int(time.time() * 1000)}",
            original_query=query,
            rewritten_query=query,
            item_names=item_names,
            is_stream=False,
            config=config,
        )

        try:
            result = query_app.invoke(state)
            # 提取上下文
            reranked_docs = result.get("reranked_docs", [])
            contexts = [doc.get("text", "") for doc in reranked_docs]

            return {
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "reranked_docs": reranked_docs,
            }
        except Exception as e:
            logger.error(f"Pipeline 执行失败: {e}")
            return {"answer": "", "contexts": [], "reranked_docs": []}

    def run_experiment(self, config_path: str) -> Dict[str, Any]:
        """
        运行完整实验

        :param config_path: YAML 配置文件路径
        :return: 实验结果字典
        """
        config = self.load_config(config_path)
        golden_set_path = config.get("golden_set_path", "data/golden_set.jsonl")
        runs = config.get("runs", [])
        exp_name = config.get("name", "unnamed")

        golden_set = self.load_golden_set(golden_set_path)
        logger.info(f"实验 '{exp_name}' 加载 {len(golden_set)} 条 Golden Set")

        all_results = {}

        for run in runs:
            run_name = run.get("name", "unnamed")
            pipeline_config = run.get("config", {})
            logger.info(f"执行 run: {run_name}, config: {pipeline_config}")

            predictions = []
            for i, case in enumerate(golden_set):
                result = self.run_single_pipeline(
                    query=case["question"],
                    item_names=case.get("item_names", []),
                    config=pipeline_config,
                )
                predictions.append({
                    "case_id": case.get("case_id", f"CASE-{i:03d}"),
                    "question": case["question"],
                    "ground_truth": case.get("ground_truth", ""),
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                })
                if (i + 1) % 10 == 0:
                    logger.info(f"  进度: {i + 1}/{len(golden_set)}")

            all_results[run_name] = predictions

        # 保存结果
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.reports_dir / f"{exp_name}_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for run_name, predictions in all_results.items():
            with open(output_dir / f"{run_name}_predictions.jsonl", "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # 计算 RAGAS 指标
        from app.eval.ragas_evaluator import evaluate_run
        metrics_results = {}
        for run_name, predictions in all_results.items():
            metrics = evaluate_run(predictions, golden_set)
            metrics_results[run_name] = metrics

        # 保存指标
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_results, f, indent=2, ensure_ascii=False)

        # 生成报告
        from app.eval.report_generator import ReportGenerator
        gen = ReportGenerator()
        report_md = gen.generate_markdown(exp_name, metrics_results, config)
        with open(output_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        logger.info(f"实验完成，结果保存在: {output_dir}")
        return {"output_dir": str(output_dir), "metrics": metrics_results}


def main():
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) < 3:
        print("用法: python -m app.eval.experiment_manager --config <yaml_path>")
        sys.exit(1)

    config_path = sys.argv[2]
    manager = ExperimentManager()
    result = manager.run_experiment(config_path)
    print(f"实验完成: {result['output_dir']}")
    for run, metrics in result["metrics"].items():
        print(f"\n{run}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()