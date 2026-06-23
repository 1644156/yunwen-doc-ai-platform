# 云问：企业文档智能问答平台

云问是一个面向企业知识库场景的 RAG 工程化项目。它把文档导入、混合检索、质量校验、引用溯源、多轮问答和 MCP 工具化调用串成一个可演示的后端系统，用来解决企业产品手册、设备说明书、技术文档难查、难追溯、难复用的问题。

这个项目更适合作为 AI 应用后端作品集来阅读：重点不是单个 Prompt，而是一个 RAG 系统从文档入库、检索增强、回答生成到外部 Agent 调用的完整工程链路。

## 如果只看 3 分钟

- **项目定位**：企业文档智能问答平台，支持 PDF / Markdown 文档导入、知识库检索、引用溯源和 MCP 调用。
- **技术关键词**：FastAPI、LangGraph、Milvus、MongoDB、MinIO、BGE-M3、Reranker、CRAG、FastMCP、DashScope / Qwen。
- **我重点实现的能力**：文档入库流程、多路检索与 Rerank、CRAG 质量判断、SSE 问答接口、MCP 工具封装、离线评测。
- **可展示结果**：仓库内保留了文档入库、问答引用、图片引用展示截图，README 下方可直接查看。

## 项目亮点

| 能力 | 说明 |
| --- | --- |
| 文档入库 | 支持 PDF 转 Markdown、图片对象存储、标题层级切分、文档主体识别和向量入库。 |
| 混合检索 | 支持 dense、sparse / BM25、HyDE、Web Search MCP 等多路召回，并用 RRF 融合。 |
| 结果质量控制 | 使用 Reranker 精排，并通过 CRAG 判断上下文质量，低相关时补检、改写或拒答。 |
| 引用溯源 | 回答保留文本引用和图片引用，便于核对答案来源。 |
| 多轮问答 | 使用 MongoDB 保存会话历史，辅助问题改写和主体确认。 |
| 工具化集成 | 通过 FastMCP 暴露查询、历史读取、历史清理、状态检查等工具，可被外部 Agent 调用。 |
| 评测意识 | 内置离线评测脚本和评测报告，用于验证回答质量、链路稳定性和延迟表现。 |

## Demo

### 文档入库工作台

![文档入库工作台](docs/assets/Snipaste_2026-06-23_09-01-01.png)

### 文档问答与引用溯源

![文档问答与引用溯源](docs/assets/Snipaste_2026-06-23_09-02-15.png)

### 图片引用展示

![图片引用展示](docs/assets/Snipaste_2026-06-23_09-02-44.png)

## 系统架构

```text
PDF / Markdown
    |
    v
FastAPI 文档导入服务
    |
    v
LangGraph 入库流程
    |-- PDF 转 Markdown
    |-- 图片处理 -> MinIO
    |-- 标题 / 段落切分
    |-- 文档主体识别
    |-- BGE-M3 向量化
    v
Milvus: 文档切片 / 主体索引

用户问题
    |
    v
FastAPI 问答服务 + SSE
    |
    v
LangGraph 查询流程
    |-- MongoDB 历史读取
    |-- 主体确认 / 查询改写
    |-- dense / sparse / HyDE / Web Search MCP
    |-- RRF 融合
    |-- Rerank 精排
    |-- CRAG 质量判断
    v
带引用的答案 / 补检 / 拒答

外部 Agent
    |
    v
FastMCP: yunwen_query_knowledge_base
```

## 核心模块

| 模块 | 关键路径 | 作用 |
| --- | --- | --- |
| 文档入库主图 | `app/import_process/agent/main_graph.py` | 编排 PDF 解析、图片处理、切分、主体识别、Embedding 和入库。 |
| 查询主图 | `app/query_process/agent/main_graph.py` | 编排多轮历史、查询改写、召回、融合、重排、质量判断和答案生成。 |
| Milvus 客户端 | `app/clients/milvus_utils.py` | 封装 dense / sparse 检索和 collection 操作。 |
| MinIO 客户端 | `app/clients/minio_utils.py` | 保存文档解析出的图片对象，支持图片引用展示。 |
| MongoDB 历史 | `app/clients/mongo_history_utils.py` | 保存多轮对话历史。 |
| CRAG 节点 | `app/query_process/agent/nodes/node_crag.py` | 判断检索上下文是否足以回答问题。 |
| MCP 服务 | `app/mcp_server.py` | 将知识库能力封装成可被外部系统调用的工具。 |
| 离线评测 | `scripts/run_eval.py` | 基于 golden set 跑端到端评测。 |

## 技术栈

- Python 3.12+
- FastAPI
- LangGraph / LangChain
- FastMCP
- Milvus
- MongoDB
- MinIO
- MinerU
- BGE-M3 / BGE Reranker
- DashScope / Qwen
- pytest

## 快速启动

### 1. 安装依赖

```bash
cd yunwen
uv sync
```

或：

```bash
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

按本地环境填写：

```text
OPENAI_API_KEY=...
MILVUS_URL=http://localhost:19530
MONGO_URL=mongodb://admin:admin123@localhost:27017
MINIO_ENDPOINT=localhost:9000
```

### 3. 启动依赖服务

```bash
cd docker
docker compose up -d
```

默认依赖：

```text
Milvus: 19530
MinIO: 9000 / 9001
MongoDB: 27017
```

### 4. 启动后端服务

```bash
python -m app.import_process.api.import_server
python -m app.query_process.api.query_server
python run_mcp_server.py
```

常用入口：

```text
文档入库：http://127.0.0.1:8000/import
文档问答：http://127.0.0.1:8001/chat.html
健康检查：http://127.0.0.1:8001/health
MCP 服务：http://127.0.0.1:9100/mcp
```

## 评测与测试

离线评测脚本：

```bash
python scripts/run_eval.py
```

已保留一份评测输出：

```text
reports/eval_20260510_191855/report.md
reports/eval_20260510_191855/summary.json
```

自动化测试：

```bash
pytest app/test -q
```

说明：完整端到端运行依赖本地模型、向量库、对象存储、MongoDB 和外部 LLM API。缺少这些依赖时，部分端到端测试需要先完成环境配置。

## 与云枢项目的关系

云问负责文档知识库能力，云枢负责电商客服 Agent 能力。两个项目组合后，云枢可以通过 `YunwenMcpRetriever` 调用云问的 `yunwen_query_knowledge_base` 工具，把文档问答能力接入到客服 Agent 中。

## 公开仓库说明

公开版本保留核心代码、示例配置、评测报告和演示截图，不包含：

- `.env` 和真实 API Key。
- 本地模型权重。
- 大量原始 PDF / zip 文档。
- 运行日志、解析输出和数据库数据。

当前项目是工程化原型，适合用于展示 AI 应用后端、RAG 系统设计、检索增强生成、MCP 工具化和测试评测能力。生产化还需要补充权限、审计、监控告警、任务队列、缓存、高并发压测和真实用户反馈闭环。
