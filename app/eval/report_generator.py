"""
报告生成器
生成 Markdown 和 HTML 格式的实验报告。
"""
from typing import Dict, Any
from datetime import datetime


class ReportGenerator:
    def generate_markdown(self, experiment_name: str,
                          metrics_results: Dict[str, Dict[str, float]],
                          config: Dict[str, Any]) -> str:
        """生成 Markdown 报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        description = config.get("description", "")

        md = f"""# 实验报告：{experiment_name}

**时间**：{timestamp}
**描述**：{description}

## 1. 总体指标对比

| 指标 | {" | ".join(metrics_results.keys())} |
|---|{"|---" * len(metrics_results)}|
"""
        # 指标行
        metric_names = set()
        for m in metrics_results.values():
            metric_names.update(m.keys())

        for metric in sorted(metric_names):
            values = []
            for run_name in metrics_results:
                val = metrics_results[run_name].get(metric, 0.0)
                values.append(f"{val:.4f}")
            md += f"| {metric} | {' | '.join(values)} |\n"

        # 各 run 的详细指标
        md += "\n## 2. 各 Run 详细指标\n\n"
        for run_name, metrics in metrics_results.items():
            md += f"### {run_name}\n\n"
            md += "| 指标 | 值 |\n|---|---|\n"
            for k, v in sorted(metrics.items()):
                md += f"| {k} | {v:.4f} |\n"
            md += "\n"

        # 配置信息
        md += "\n## 3. 实验配置\n\n"
        md += "```yaml\n"
        import yaml
        md += yaml.dump(config, allow_unicode=True, default_flow_style=False)
        md += "\n```\n"

        return md

    def generate_html(self, experiment_name: str,
                      metrics_results: Dict[str, Dict[str, float]],
                      config: Dict[str, Any]) -> str:
        """生成 HTML 报告"""
        md = self.generate_markdown(experiment_name, metrics_results, config)
        try:
            import markdown
            return markdown.markdown(md, extensions=["tables"])
        except ImportError:
            return f"<pre>{md}</pre>"