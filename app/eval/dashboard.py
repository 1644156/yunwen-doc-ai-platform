"""
RAG 评测看板
Streamlit 应用，提供实验对比、Case 下钻、历史趋势三个页面。
"""
import json
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="RAG 评测看板", layout="wide")
st.title("RAG 评测看板")

# 侧边栏导航
page = st.sidebar.radio("导航", ["实验对比", "Case 下钻", "历史趋势"])

reports_dir = Path("reports")


def load_experiment_names():
    if not reports_dir.exists():
        return []
    return sorted([d.name for d in reports_dir.iterdir() if d.is_dir()])


def load_metrics(exp_name: str):
    path = reports_dir / exp_name / "metrics.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_predictions(exp_name: str, run_name: str):
    path = reports_dir / exp_name / f"{run_name}_predictions.jsonl"
    cases = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


# ========================
# 页面 1: 实验对比
# ========================
if page == "实验对比":
    st.header("实验对比")

    experiments = load_experiment_names()
    if not experiments:
        st.info("暂无实验结果。请先运行实验。")
    else:
        selected = st.multiselect("选择要对比的实验", experiments,
                                  default=experiments[-2:] if len(experiments) >= 2 else experiments[-1:])

        if selected:
            all_metrics = {}
            for exp_name in selected:
                metrics = load_metrics(exp_name)
                all_metrics[exp_name] = metrics

            # 雷达图
            st.subheader("RAGAS 指标雷达图")
            metric_names = ["context_precision", "context_recall", "faithfulness",
                            "answer_relevancy", "answer_correctness"]

            fig = go.Figure()
            for exp_name, run_metrics in all_metrics.items():
                for run_name, metrics in run_metrics.items():
                    values = [metrics.get(m, 0) for m in metric_names]
                    values.append(values[0])  # 闭合
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metric_names + [metric_names[0]],
                        fill="toself",
                        name=f"{exp_name}/{run_name}"
                    ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)

            # 指标表格
            st.subheader("详细指标表")
            rows = []
            for exp_name, run_metrics in all_metrics.items():
                for run_name, metrics in run_metrics.items():
                    row = {"实验": exp_name, "Run": run_name}
                    row.update(metrics)
                    rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows))


# ========================
# 页面 2: Case 下钻
# ========================
elif page == "Case 下钻":
    st.header("单 Case 下钻")

    experiments = load_experiment_names()
    if not experiments:
        st.info("暂无实验结果。")
    else:
        selected_exp = st.selectbox("选择实验", experiments)
        metrics = load_metrics(selected_exp)
        run_names = list(metrics.keys()) if metrics else []

        if run_names:
            selected_run = st.selectbox("选择 Run", run_names)
            cases = load_predictions(selected_exp, selected_run)

            if cases:
                # 排序
                sort_by = st.selectbox("排序方式", ["case_id", "answer (非空优先)"])
                if sort_by == "answer (非空优先)":
                    cases = sorted(cases, key=lambda c: (0 if c.get("answer") else 1))

                case_ids = [c.get("case_id", f"CASE-{i}") for i, c in enumerate(cases)]
                selected_idx = st.selectbox("选择 Case", range(len(cases)),
                                            format_fn=lambda i: case_ids[i])
                case = cases[selected_idx]

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("问题")
                    st.info(case.get("question", ""))
                    st.subheader("Ground Truth")
                    st.success(case.get("ground_truth", ""))
                    st.subheader("系统答案")
                    st.warning(case.get("answer", ""))

                with col2:
                    st.subheader("召回的上下文")
                    contexts = case.get("contexts", [])
                    for i, ctx in enumerate(contexts, 1):
                        with st.expander(f"Context {i}"):
                            st.write(ctx[:500] if ctx else "(空)")


# ========================
# 页面 3: 历史趋势
# ========================
elif page == "历史趋势":
    st.header("历史趋势")

    experiments = load_experiment_names()
    if len(experiments) < 2:
        st.info("至少需要 2 个实验才能显示趋势。")
    else:
        # 收集所有实验的指标
        trend_data = []
        for exp_name in experiments:
            metrics = load_metrics(exp_name)
            for run_name, m in metrics.items():
                row = {"实验": exp_name, "Run": run_name}
                row.update(m)
                trend_data.append(row)

        if trend_data:
            df = pd.DataFrame(trend_data)
            metric_cols = [c for c in df.columns if c not in ("实验", "Run")]

            selected_metric = st.selectbox("选择指标", metric_cols)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"{r['实验']}/{r['Run']}" for _, r in df.iterrows()],
                y=df[selected_metric],
                name=selected_metric
            ))
            fig.update_layout(
                title=f"{selected_metric} 趋势",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df)