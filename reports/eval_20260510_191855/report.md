# RAG 评测报告

**时间**: 2026-05-10 19:18:55
**Golden Set**: 82 条
**RAGAS 采样数**: 20

## 1. 基础指标对比

| 实验 | 成功率 | 有效答案率 | 拒答 | 平均延迟(ms) |
|---|---|---|---|---|
| baseline | 100.0% | 98.8% | 0 | 5787 |
| with_routing | 100.0% | 98.8% | 0 | 4202 |
| with_crag | 100.0% | 98.8% | 2 | 5526 |
| full | 100.0% | 97.6% | 2 | 4005 |

## 2. 评测指标对比

| 实验 | answer_f1 | answer_coverage | context_relevance | answer_length_avg | success_rate |
|---|---|---|---|---|---|
| baseline | 0.1016 | 0.2026 | 0.0104 | 398.6000 | 1.0000 |
| with_routing | 0.0968 | 0.1987 | 0.0072 | 399.3500 | 1.0000 |
| with_crag | 0.0981 | 0.2299 | 0.0504 | 423.2000 | 1.0000 |
| full | 0.0726 | 0.1732 | 0.0272 | 461.2000 | 1.0000 |

## 3. 实验配置

### baseline

- 配置: `{"enable_routing": false, "enable_crag": false}`
- 成功: 82/82
- 失败: 0

### with_routing

- 配置: `{"enable_routing": true, "enable_crag": false}`
- 成功: 82/82
- 失败: 0

### with_crag

- 配置: `{"enable_routing": false, "enable_crag": true}`
- 成功: 82/82
- 失败: 0

### full

- 配置: `{"enable_routing": true, "enable_crag": true}`
- 成功: 82/82
- 失败: 0

