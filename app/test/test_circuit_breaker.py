"""
熔断器单元测试
"""
import time
import pytest
from app.utils.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError


class TestCircuitBreaker:
    """测试熔断器核心功能"""

    def test_initial_state_is_closed(self):
        """初始状态应该是CLOSED"""
        breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)
        assert breaker.state == CircuitState.CLOSED

    def test_success_keeps_closed(self):
        """成功调用应该保持CLOSED状态"""
        breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)

        def success_func():
            return "ok"

        result = breaker.call(success_func)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    def test_failure_increments_count(self):
        """失败调用应该增加失败计数"""
        breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)

        def failing_func():
            raise ValueError("test error")

        # 第1次失败
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitState.CLOSED

        # 第2次失败
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitState.CLOSED

    def test_threshold_trips_to_open(self):
        """达到失败阈值应该触发熔断"""
        breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)

        def failing_func():
            raise ValueError("test error")

        # 连续失败3次
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        """熔断状态应该拒绝调用"""
        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=1.0)

        def failing_func():
            raise ValueError("test error")

        # 触发熔断
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # 熔断后应该拒绝调用
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "ok")

    def test_open_uses_fallback(self):
        """熔断状态应该使用降级函数"""
        def fallback():
            return "fallback_result"

        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=1.0, fallback=fallback)

        def failing_func():
            raise ValueError("test error")

        # 触发熔断
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # 熔断后应该返回降级结果
        result = breaker.call(lambda: "ok")
        assert result == "fallback_result"

    def test_recovery_timeout_to_half_open(self):
        """超过恢复时间应该转为HALF_OPEN"""
        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)

        def failing_func():
            raise ValueError("test error")

        # 触发熔断
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitState.OPEN

        # 等待恢复时间
        time.sleep(0.2)

        # 应该转为HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        """HALF_OPEN状态成功应该关闭熔断器"""
        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)

        def failing_func():
            raise ValueError("test error")

        # 触发熔断
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # 等待恢复时间
        time.sleep(0.2)

        # HALF_OPEN状态成功
        result = breaker.call(lambda: "ok")
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """HALF_OPEN状态失败应该重新打开熔断器"""
        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)

        def failing_func():
            raise ValueError("test error")

        # 触发熔断
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # 等待恢复时间
        time.sleep(0.2)

        # HALF_OPEN状态失败
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    def test_reset(self):
        """手动重置应该恢复正常"""
        breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=1.0)

        def failing_func():
            raise ValueError("test error")

        # 触发熔断
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        assert breaker.state == CircuitState.OPEN

        # 手动重置
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    def test_protect_decorator(self):
        """装饰器方式应该正常工作"""
        breaker = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)

        @breaker.protect
        def my_func():
            return "decorated"

        result = my_func()
        assert result == "decorated"
        assert breaker.state == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
