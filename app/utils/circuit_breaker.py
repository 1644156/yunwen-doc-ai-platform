"""
熔断器工具类
实现外部服务调用的熔断保护，防止级联故障。

状态机：
    CLOSED (正常) → 失败次数超过阈值 → OPEN (熔断)
    OPEN (熔断) → 等待恢复超时 → HALF_OPEN (半开)
    HALF_OPEN (半开) → 调用成功 → CLOSED (正常)
    HALF_OPEN (半开) → 调用失败 → OPEN (熔断)
"""
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from app.core.logger import logger


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"        # 正常状态，允许调用
    OPEN = "open"            # 熔断状态，拒绝调用
    HALF_OPEN = "half_open"  # 半开状态，尝试恢复


class CircuitBreaker:
    """
    通用熔断器

    使用示例：
        breaker = CircuitBreaker(
            name="llm_service",
            failure_threshold=5,
            recovery_timeout=60
        )

        # 方式1：装饰器
        @breaker.protect
        def call_llm(prompt):
            return llm.invoke(prompt)

        # 方式2：上下文管理器
        with breaker:
            result = llm.invoke(prompt)

        # 方式3：手动调用
        result = breaker.call(llm.invoke, prompt)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        fallback: Optional[Callable] = None
    ):
        """
        初始化熔断器

        Args:
            name: 熔断器名称（用于日志）
            failure_threshold: 触发熔断的连续失败次数
            recovery_timeout: 熔断恢复等待时间（秒）
            fallback: 降级函数，熔断时调用
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """获取当前状态（自动检查是否应该转换为HALF_OPEN）"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否超过恢复时间
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"[熔断器:{self.name}] 状态转换 OPEN -> HALF_OPEN")
            return self._state

    def _on_success(self):
        """调用成功时的处理"""
        with self._lock:
            self._failure_count = 0
            if self._state != CircuitState.CLOSED:
                self._state = CircuitState.CLOSED
                logger.info(f"[熔断器:{self.name}] 状态转换 -> CLOSED (恢复正常)")

    def _on_failure(self, error: Exception):
        """调用失败时的处理"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态失败，重新熔断
                self._state = CircuitState.OPEN
                logger.warning(f"[熔断器:{self.name}] 状态转换 HALF_OPEN -> OPEN (恢复失败: {error})")
            elif self._failure_count >= self.failure_threshold:
                # 达到失败阈值，触发熔断
                self._state = CircuitState.OPEN
                logger.error(
                    f"[熔断器:{self.name}] 状态转换 CLOSED -> OPEN "
                    f"(连续失败{self._failure_count}次，阈值{self.failure_threshold})"
                )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器调用函数

        Args:
            func: 要调用的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值，或降级函数返回值

        Raises:
            CircuitOpenError: 熔断器打开且无降级函数时抛出
            Exception: 原始函数异常（熔断器未打开时）
        """
        current_state = self.state

        # 熔断状态：直接拒绝调用
        if current_state == CircuitState.OPEN:
            logger.warning(f"[熔断器:{self.name}] 调用被拒绝 (熔断中)")
            if self.fallback:
                return self.fallback(*args, **kwargs)
            raise CircuitOpenError(f"熔断器 {self.name} 已打开，拒绝调用")

        # 半开状态：允许尝试调用
        if current_state == CircuitState.HALF_OPEN:
            logger.info(f"[熔断器:{self.name}] 半开状态，尝试恢复调用")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def protect(self, func: Callable) -> Callable:
        """
        装饰器方式保护函数

        使用示例：
            @breaker.protect
            def my_function():
                ...
        """
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def __enter__(self):
        """上下文管理器入口"""
        current_state = self.state
        if current_state == CircuitState.OPEN:
            if self.fallback:
                self._fallback_result = self.fallback()
                return self
            raise CircuitOpenError(f"熔断器 {self.name} 已打开")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type is not None:
            self._on_failure(exc_val)
        else:
            self._on_success()
        return False  # 不抑制异常

    def reset(self):
        """手动重置熔断器"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
            logger.info(f"[熔断器:{self.name}] 手动重置为 CLOSED")


class CircuitOpenError(Exception):
    """熔断器打开异常"""
    pass


# ==================== 预定义的熔断器实例 ====================

# LLM服务熔断器：连续失败5次，60秒后尝试恢复
llm_circuit_breaker = CircuitBreaker(
    name="llm_service",
    failure_threshold=5,
    recovery_timeout=60.0
)

# Milvus服务熔断器：连续失败3次，30秒后尝试恢复
milvus_circuit_breaker = CircuitBreaker(
    name="milvus_service",
    failure_threshold=3,
    recovery_timeout=30.0
)

# MongoDB服务熔断器：连续失败3次，30秒后尝试恢复
mongodb_circuit_breaker = CircuitBreaker(
    name="mongodb_service",
    failure_threshold=3,
    recovery_timeout=30.0
)

# MinerU服务熔断器：连续失败3次，60秒后尝试恢复
mineru_circuit_breaker = CircuitBreaker(
    name="mineru_service",
    failure_threshold=3,
    recovery_timeout=60.0
)

# MCP网络搜索熔断器：连续失败5次，30秒后尝试恢复
mcp_circuit_breaker = CircuitBreaker(
    name="mcp_search",
    failure_threshold=5,
    recovery_timeout=30.0
)
