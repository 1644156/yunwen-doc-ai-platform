# 熔断器使用指南

## 概述

熔断器（Circuit Breaker）是一种保护机制，用于防止外部服务故障时的级联失败。当外部服务（如LLM、Milvus、MongoDB）连续失败时，熔断器会自动"打开"，停止调用该服务，避免资源浪费和雪崩效应。

## 状态机

```
CLOSED (正常) ──失败次数超过阈值──→ OPEN (熔断)
      ↑                                │
      │                                │ 等待恢复超时
      │                                ↓
      └──────调用成功────── HALF_OPEN (半开)
                                │
                                └──调用失败──→ OPEN (熔断)
```

## 已集成的服务

| 服务 | 熔断器实例 | 失败阈值 | 恢复超时 |
|------|-----------|---------|---------|
| LLM服务 | `llm_circuit_breaker` | 5次 | 60秒 |
| Milvus服务 | `milvus_circuit_breaker` | 3次 | 30秒 |
| MongoDB服务 | `mongodb_circuit_breaker` | 3次 | 30秒 |
| MinerU服务 | `mineru_circuit_breaker` | 3次 | 60秒 |
| MCP搜索 | `mcp_circuit_breaker` | 5次 | 30秒 |

## 使用方式

### 方式1：使用预定义的包装函数（推荐）

```python
from app.lm.lm_utils import invoke_llm, stream_llm

# 同步调用（自动带熔断保护）
answer = invoke_llm("什么是RAG?")

# 流式调用（自动带熔断保护）
for chunk in stream_llm("什么是RAG?"):
    print(chunk)
```

### 方式2：使用装饰器

```python
from app.utils.circuit_breaker import llm_circuit_breaker

@llm_circuit_breaker.protect
def my_llm_call(prompt: str) -> str:
    client = get_llm_client()
    return client.invoke(prompt).content
```

### 方式3：使用call方法

```python
from app.utils.circuit_breaker import milvus_circuit_breaker

def search_milvus(query_vector):
    client = get_milvus_client()
    return client.search(...)

# 通过熔断器调用
results = milvus_circuit_breaker.call(search_milvus, query_vector)
```

### 方式4：创建自定义熔断器

```python
from app.utils.circuit_breaker import CircuitBreaker

# 创建自定义熔断器
my_breaker = CircuitBreaker(
    name="my_service",
    failure_threshold=3,      # 连续失败3次触发熔断
    recovery_timeout=30.0,    # 30秒后尝试恢复
    fallback=lambda: "默认值"  # 熔断时的降级返回
)

# 使用熔断器
result = my_breaker.call(risky_function)
```

## 降级策略

当熔断器打开时，可以通过以下方式处理：

### 1. 设置降级函数

```python
breaker = CircuitBreaker(
    name="llm_service",
    failure_threshold=5,
    recovery_timeout=60.0,
    fallback=lambda prompt: "抱歉，服务暂时不可用，请稍后再试。"
)
```

### 2. 捕获异常

```python
from app.utils.circuit_breaker import CircuitOpenError

try:
    result = breaker.call(risky_function)
except CircuitOpenError:
    # 熔断器打开，执行降级逻辑
    result = "服务暂时不可用"
```

## 监控和日志

熔断器会自动记录状态变化日志：

```
[熔断器:llm_service] 状态转换 CLOSED -> OPEN (连续失败5次，阈值5)
[熔断器:llm_service] 调用被拒绝 (熔断中)
[熔断器:llm_service] 状态转换 OPEN -> HALF_OPEN
[熔断器:llm_service] 半开状态，尝试恢复调用
[熔断器:llm_service] 状态转换 -> CLOSED (恢复正常)
```

## 最佳实践

1. **合理设置阈值**：根据服务特性设置合适的失败阈值
2. **设置降级函数**：为关键服务提供降级方案
3. **监控熔断状态**：通过日志监控熔断器状态变化
4. **定期测试恢复**：确保熔断器能正常恢复

## 注意事项

- 熔断器是线程安全的
- 熔断器状态不会持久化，重启后会重置
- 半开状态只允许一次尝试调用
- 降级函数不应该抛出异常
