# 云问文档智能平台

云问是一个面向企业产品手册、设备说明书和技术文档的 RAG 问答原型。它支持 PDF/Markdown 文档导入、图片对象存储、Milvus 混合检索、Rerank、CRAG 质量校验、引用溯源和 MCP 工具化调用。

> 项目定位：AI 应用后端工程化原型，重点展示文档入库、检索增强生成、服务化封装和评测意识。它不是生产环境直接上线版本。

## Demo

本仓库保留核心代码、启动配置、评测报告、示例配置和演示截图。

### 文档入库工作台

![文档入库工作台](docs/assets/Snipaste_2026-06-23_09-01-01.png)

### 文档问答与引用溯源

![文档问答与引用溯源](docs/assets/Snipaste_2026-06-23_09-02-15.png)

### 图片引用展示

![图片引用展示](docs/assets/Snipaste_2026-06-23_09-02-44.png)

## 解决的问题

企业内部常见的产品手册、设备说明书和技术规范文档数量多、格式杂，人工查找效率低。普通关键词检索对自然语言问题支持有限，简单 RAG demo 又容易停留在“切分、向量检索、拼 prompt、生成答案”的单链路。

云问把文档问答拆成更完整的工程链路：

- 导入时处理 PDF/Markdown、图片、标题层级、文档主体和向量入库。
- 查询时进行主体确认、查询路由、多路召回、RRF 融合和 Rerank。
- 生成前通过 CRAG 判断上下文质量，低相关时补检、改写或拒答。
- 使用 MongoDB 保存多轮会话历史，辅助问题改写和主体确认。
- 使用 MinIO 保存文档解析出的图片，让回答可以保留图片引用。
- 通过 FastMCP 暴露工具接口，可被客服 Agent 等外部系统调用。

## 架构概览

```text
PDF / Markdown
    |
    v
导入服务 FastAPI
    |
    v
LangGraph 导入链路
    |-- PDF 转 Markdown
    |-- 图片处理 -> MinIO
    |-- 标题/段落切分
    |-- 文档主体识别
    |-- BGE-M3 向量化
    v
Milvus: kb_chunks / kb_item_names

用户问题
    |
    v
查询服务 FastAPI + SSE
    |
    v
LangGraph 查询链路
    |-- MongoDB 历史读取
    |-- 主体确认 / 查询改写
    |-- 查询路由
    |-- dense / sparse / HyDE / Web Search MCP
    |-- RRF + Rerank
    |-- CRAG 质量判断
    v
带引用的答案 / 补检 / 拒答

外部 Agent
    |
    v
FastMCP: yunwen_query_knowledge_base
```

## 核心能力

### 文档导入链路

入口：`app/import_process/agent/main_graph.py`

```text
node_entry
-> node_pdf_to_md
-> node_md_img
-> node_document_split
-> node_item_name_recognition
-> node_bge_embedding
-> node_import_milvus
```

职责：

- `node_pdf_to_md`：将 PDF 解析成 Markdown。
- `node_md_img`：处理 Markdown 图片并上传到 MinIO。
- `node_document_split`：按标题层级和段落结构切分。
- `node_item_name_recognition`：识别文档主体，方便查询过滤。
- `node_bge_embedding`：生成 dense/sparse 向量。
- `node_import_milvus`：创建 collection、清理旧数据、写入切片。

### RAG 查询链路

入口：`app/query_process/agent/main_graph.py`

```text
主体确认
-> 查询路由
-> 多路召回
-> RRF 融合
-> Rerank 精排
-> CRAG 质量判断
-> 答案生成 / 补检 / 改写 / 拒答
```

已实现能力：

- Milvus dense 向量检索。
- Milvus sparse/BM25 检索。
- HyDE 假设答案检索。
- Web Search MCP 补充检索。
- RRF 多路结果融合。
- BGE Reranker 精排。
- CRAG 检索质量判断。
- 低置信拒答和引用溯源。

### 数据组件

| 组件 | 用途 |
| --- | --- |
| Milvus | 存储 `kb_chunks` 和 `kb_item_names`，支持 dense/sparse 混合检索 |
| MinIO | 存储文档解析出的图片对象，回答中保留图片 URL |
| MongoDB | 保存 `chat_message` 会话历史，支持多轮问题改写和主体确认 |
| DashScope/Qwen | LLM、视觉模型和生成能力 |
| BGE-M3 / BGE Reranker | 向量化和重排序 |

### MCP 服务

入口：`app/mcp_server.py`，启动脚本：`run_mcp_server.py`

已暴露工具：

- `yunwen_query_knowledge_base`：查询知识库。
- `yunwen_get_chat_history`：获取会话历史。
- `yunwen_clear_chat_history`：清理会话历史。
- `yunwen_get_service_status`：查看服务状态。

默认地址：

```text
http://127.0.0.1:9100/mcp
```

## 技术栈

- Python 3.12+
- FastAPI
- LangGraph
- LangChain
- FastMCP
- Milvus
- MongoDB
- MinIO
- MinerU
- BGE-M3
- BGE Reranker
- DashScope/Qwen
- pytest

## 项目结构

```text
yunwen/
├─ app/
│  ├─ import_process/       # 文档导入流程
│  ├─ query_process/        # RAG 查询流程
│  ├─ eval/                 # 评测模块
│  ├─ clients/              # Milvus、MongoDB、MinIO 客户端
│  ├─ lm/                   # LLM、Embedding、Rerank 封装
│  ├─ utils/                # SSE、限流、熔断等工具
│  └─ mcp_server.py         # MCP 服务入口
├─ doc/                     # 本地示例文档目录，公开仓库仅保留说明
├─ docker/                  # Milvus、MongoDB、MinIO 依赖服务
├─ prompts/                 # 提示词模板
├─ reports/                 # 离线评测报告
├─ scripts/                 # 评测和辅助脚本
├─ .env.example             # 示例配置
├─ run_mcp_server.py
└─ pyproject.toml
```

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

依赖服务：

```text
Milvus: 19530
MinIO: 9000 / 9001
MongoDB: 27017
```

### 4. 启动后端服务

文档导入服务：

```bash
python -m app.import_process.api.import_server
```

文档问答服务：

```bash
python -m app.query_process.api.query_server
```

MCP 服务：

```bash
python run_mcp_server.py
```

常用页面：

```text
文档入库：http://127.0.0.1:8000/import
文档问答：http://127.0.0.1:8001/chat.html
健康检查：http://127.0.0.1:8001/health
MCP 服务：http://127.0.0.1:9100/mcp
```

## 评测结果

评测脚本：`scripts/run_eval.py`

评测报告：

```text
reports/eval_20260510_191855/report.md
reports/eval_20260510_191855/summary.json
```

基于 82 条自建 golden set，完整链路结果：

| 指标 | 结果 |
| --- | --- |
| 问题数量 | 82 |
| 成功率 | 100% |
| 质量通过率 | 约 97.6% |
| 平均延迟 | 约 4.0s |
| 拒答次数 | 2 |

这些指标用于说明原型链路稳定性和回答质量，不等同于生产线上真实指标。

## 测试

```bash
pytest app/test -q
```

如果本地缺少完整模型、向量库或 API Key，部分端到端测试需要先完成环境配置。

## 关键代码路径

- 导入主图：`app/import_process/agent/main_graph.py`
- 查询主图：`app/query_process/agent/main_graph.py`
- Milvus 客户端：`app/clients/milvus_utils.py`
- MinIO 客户端：`app/clients/minio_utils.py`
- MongoDB 历史：`app/clients/mongo_history_utils.py`
- CRAG 节点：`app/query_process/agent/nodes/node_crag.py`
- RRF 节点：`app/query_process/agent/nodes/node_rrf.py`
- Rerank 节点：`app/query_process/agent/nodes/node_rerank.py`
- MCP 服务：`app/mcp_server.py`

## 安全与公开说明

公开仓库不包含：

- `.env` 和真实 API Key。
- 本地模型权重。
- 大量原始 PDF/zip 文档。
- 运行日志、解析输出和数据库数据。

需要运行时，请根据 `.env.example` 自行配置本地服务和密钥。

## 项目边界

当前项目仍是工程化原型：

- 尚未接入真实企业权限、审计、监控和高并发压测。
- 文档数据和评测集主要用于本地演示与离线验证。
- 完整运行依赖模型、向量库、对象存储、MongoDB 和外部 LLM API。
- 生产化还需要补充权限控制、任务队列、缓存、监控告警和真实反馈闭环。
