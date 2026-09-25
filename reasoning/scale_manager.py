import threading
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Deque, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from config import settings

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class LoadSheddingStrategy(str, Enum):
    REJECT_NEW = "reject_new"
    DROP_OLDEST = "drop_oldest"
    DROP_LOWEST_PRIORITY = "drop_lowest_priority"
    QUEUE_WITH_TIMEOUT = "queue_with_timeout"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class RateLimitConfig:
    requests_per_window: int
    window_seconds: float
    burst_allowance: int = 0


@dataclass
class TenantQuotaConfig:
    rate_limit: RateLimitConfig
    max_concurrent: int = 10
    max_queue_size: int = 100
    priority: int = 1


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3


@dataclass
class SlidingWindowEntry:
    timestamp: float
    count: int = 1


@dataclass
class CircuitBreaker:
    name: str
    config: CircuitBreakerConfig
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_success(self) -> None:
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.last_state_change = time.time()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                self.last_state_change = time.time()
            elif self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_state_change = time.time()

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.last_state_change = time.time()
                    return True
                return False
            if self.state == CircuitState.HALF_OPEN:
                return self.success_count < self.config.half_open_max_calls
            return False

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "last_state_change": self.last_state_change,
            }


class SlidingWindowRateLimiter:
    def __init__(self, redis_client=None, fallback_memory: bool = True):
        self._redis = redis_client
        self._memory_windows: Dict[str, Deque[SlidingWindowEntry]] = defaultdict(deque)
        self._memory_lock = threading.RLock()
        self._fallback_memory = fallback_memory
        self._use_redis = redis_client is not None and redis is not None

    def _get_redis_key(self, identifier: str, window_seconds: float) -> str:
        return f"ratelimit:{identifier}:{int(window_seconds)}"

    def _cleanup_window(self, window: Deque[SlidingWindowEntry], window_start: float) -> None:
        while window and window[0].timestamp < window_start:
            window.popleft()

    def _get_current_count_redis(self, key: str, window_start: float, window_seconds: float) -> int:
        if not self._use_redis or self._redis is None:
            return 0
        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = pipe.execute()
            return results[1]
        except Exception:
            self._use_redis = False
            return 0

    def _increment_redis(self, key: str, window_seconds: float, count: int = 1) -> int:
        if not self._use_redis or self._redis is None:
            return 0
        try:
            now = time.time()
            pipe = self._redis.pipeline()
            pipe.zadd(key, {f"{now}:{uuid.uuid4().hex[:8]}": now})
            pipe.expire(key, int(window_seconds) + 1)
            pipe.zcard(key)
            results = pipe.execute()
            return results[2]
        except Exception:
            self._use_redis = False
            return 0

    def check_and_increment(
        self, identifier: str, config: RateLimitConfig, count: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        now = time.time()
        window_start = now - config.window_seconds
        key = self._get_redis_key(identifier, config.window_seconds)

        if self._use_redis:
            current = self._get_current_count_redis(key, window_start, config.window_seconds)
            if current + count <= config.requests_per_window + config.burst_allowance:
                new_count = self._increment_redis(key, config.window_seconds, count)
                remaining = max(0, config.requests_per_window + config.burst_allowance - new_count)
                return True, {
                    "allowed": True,
                    "current": new_count,
                    "limit": config.requests_per_window + config.burst_allowance,
                    "remaining": remaining,
                    "reset_at": now + config.window_seconds,
                    "backend": "redis",
                }

        with self._memory_lock:
            window = self._memory_windows[key]
            self._cleanup_window(window, window_start)
            current = sum(entry.count for entry in window)
            if current + count <= config.requests_per_window + config.burst_allowance:
                window.append(SlidingWindowEntry(timestamp=now, count=count))
                new_count = current + count
                remaining = max(0, config.requests_per_window + config.burst_allowance - new_count)
                return True, {
                    "allowed": True,
                    "current": new_count,
                    "limit": config.requests_per_window + config.burst_allowance,
                    "remaining": remaining,
                    "reset_at": now + config.window_seconds,
                    "backend": "memory",
                }

        retry_after = config.window_seconds
        if self._use_redis:
            try:
                oldest = self._redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = max(0.1, oldest[0][1] + config.window_seconds - now)
            except Exception:
                pass
        else:
            with self._memory_lock:
                window = self._memory_windows[key]
                if window:
                    retry_after = max(0.1, window[0].timestamp + config.window_seconds - now)

        return False, {
            "allowed": False,
            "current": current,
            "limit": config.requests_per_window + config.burst_allowance,
            "remaining": 0,
            "retry_after": retry_after,
            "backend": "redis" if self._use_redis else "memory",
        }

    def get_current_usage(self, identifier: str, window_seconds: float) -> int:
        now = time.time()
        window_start = now - window_seconds
        key = self._get_redis_key(identifier, window_seconds)

        if self._use_redis:
            return self._get_current_count_redis(key, window_start, window_seconds)

        with self._memory_lock:
            window = self._memory_windows[key]
            self._cleanup_window(window, window_start)
            return sum(entry.count for entry in window)

    def reset(self, identifier: str, window_seconds: float) -> None:
        key = self._get_redis_key(identifier, window_seconds)
        if self._use_redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        with self._memory_lock:
            self._memory_windows.pop(key, None)

    def is_using_redis(self) -> bool:
        return self._use_redis


@dataclass
class QueuedRequest:
    request_id: str
    tenant_id: str
    priority: int
    enqueue_time: float
    timeout: float
    execute_fn: Callable[[], Any]
    future: Any = None


class ScaleManager:
    _instance: Optional["ScaleManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._rate_limiter = SlidingWindowRateLimiter()
        self._tenant_configs: Dict[str, TenantQuotaConfig] = {}
        self._tenant_configs_lock = threading.RLock()

        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._circuit_breakers_lock = threading.RLock()

        self._concurrency_semaphores: Dict[str, threading.Semaphore] = {}
        self._concurrency_lock = threading.RLock()

        self._request_queues: Dict[str, Deque[QueuedRequest]] = defaultdict(deque)
        self._queue_lock = threading.RLock()
        self._queue_processor_threads: Dict[str, threading.Thread] = {}
        self._queue_stop_events: Dict[str, threading.Event] = {}

        self._load_shedding_strategy = LoadSheddingStrategy.REJECT_NEW
        self._global_max_concurrent = 1000
        self._global_active_count = 0
        self._global_active_lock = threading.Lock()

        self._health_checks: Dict[str, Callable[[], bool]] = {}
        self._health_check_lock = threading.RLock()

        self._degradation_mode = False
        self._degradation_lock = threading.Lock()

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ScaleManager":
        return cls()

    def configure_redis(self, host: str = "localhost", port: int = 6379, db: int = 0, **kwargs) -> bool:
        if redis is None:
            return False
        try:
            client = redis.Redis(host=host, port=port, db=db, decode_responses=True, **kwargs)
            client.ping()
            self._rate_limiter = SlidingWindowRateLimiter(redis_client=client)
            return True
        except Exception:
            self._rate_limiter = SlidingWindowRateLimiter(redis_client=None)
            return False

    def configure_tenant(self, tenant_id: str, config: TenantQuotaConfig) -> None:
        with self._tenant_configs_lock:
            self._tenant_configs[tenant_id] = config
            self._ensure_semaphore(tenant_id, config.max_concurrent)

    def get_tenant_config(self, tenant_id: str) -> Optional[TenantQuotaConfig]:
        with self._tenant_configs_lock:
            return self._tenant_configs.get(tenant_id)

    def remove_tenant(self, tenant_id: str) -> None:
        with self._tenant_configs_lock:
            self._tenant_configs.pop(tenant_id, None)
        with self._concurrency_lock:
            self._concurrency_semaphores.pop(tenant_id, None)
        self._rate_limiter.reset(f"tenant:{tenant_id}", 60.0)
        self._stop_queue_processor(tenant_id)

    def _ensure_semaphore(self, tenant_id: str, max_concurrent: int) -> None:
        with self._concurrency_lock:
            if tenant_id not in self._concurrency_semaphores:
                self._concurrency_semaphores[tenant_id] = threading.Semaphore(max_concurrent)
            else:
                current = self._concurrency_semaphores[tenant_id]
                if hasattr(current, '_value'):
                    while current._value < max_concurrent:
                        current.release()
                    while current._value > max_concurrent:
                        try:
                            current.acquire(blocking=False)
                        except ValueError:
                            break

    def configure_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> None:
        with self._circuit_breakers_lock:
            self._circuit_breakers[name] = CircuitBreaker(name, config)

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        with self._circuit_breakers_lock:
            return self._circuit_breakers.get(name)

    def execute_with_circuit_breaker(
        self, name: str, fn: Callable[[], Any], fallback: Optional[Callable[[], Any]] = None
    ) -> Any:
        breaker = self.get_circuit_breaker(name)
        if breaker is None:
            return fn()

        if not breaker.can_execute():
            if fallback:
                return fallback()
            raise RuntimeError(f"Circuit breaker '{name}' is OPEN")

        try:
            result = fn()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            if fallback:
                return fallback()
            raise

    def acquire_concurrency(self, tenant_id: str, blocking: bool = True, timeout: float = 30.0) -> bool:
        with self._concurrency_lock:
            sem = self._concurrency_semaphores.get(tenant_id)

        with self._global_active_lock:
            if self._global_active_count >= self._global_max_concurrent:
                return False
            self._global_active_count += 1

        if sem is None:
            return True

        try:
            if blocking:
                acquired = sem.acquire(blocking=True, timeout=timeout)
            else:
                acquired = sem.acquire(blocking=False)
            if not acquired:
                with self._global_active_lock:
                    self._global_active_count -= 1
            return acquired
        except Exception:
            with self._global_active_lock:
                self._global_active_count -= 1
            return False

    def release_concurrency(self, tenant_id: str) -> None:
        with self._concurrency_lock:
            sem = self._concurrency_semaphores.get(tenant_id)
            if sem:
                try:
                    sem.release()
                except ValueError:
                    pass
        with self._global_active_lock:
            self._global_active_count = max(0, self._global_active_count - 1)

    def check_rate_limit(self, tenant_id: str, count: int = 1) -> Tuple[bool, Dict[str, Any]]:
        config = self.get_tenant_config(tenant_id)
        if not config:
            return True, {"allowed": True, "reason": "no_config"}

        identifier = f"tenant:{tenant_id}"
        return self._rate_limiter.check_and_increment(identifier, config.rate_limit, count)

    def enqueue_request(
        self,
        tenant_id: str,
        execute_fn: Callable[[], Any],
        priority: int = 1,
        timeout: float = 30.0,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        config = self.get_tenant_config(tenant_id)
        if not config:
            return False, None, {"error": "tenant_not_configured"}

        with self._queue_lock:
            queue = self._request_queues[tenant_id]
            if len(queue) >= config.max_queue_size:
                return self._apply_load_shedding(tenant_id, queue, execute_fn, priority, timeout)

            request_id = uuid.uuid4().hex[:12]
            request = QueuedRequest(
                request_id=request_id,
                tenant_id=tenant_id,
                priority=priority,
                enqueue_time=time.time(),
                timeout=timeout,
                execute_fn=execute_fn,
            )
            queue.append(request)
            self._sort_queue_by_priority(tenant_id)

        self._ensure_queue_processor(tenant_id)
        return True, request_id, {"queued": True, "position": len(queue)}

    def _sort_queue_by_priority(self, tenant_id: str) -> None:
        queue = self._request_queues[tenant_id]
        sorted_items = sorted(queue, key=lambda r: (-r.priority, r.enqueue_time))
        queue.clear()
        queue.extend(sorted_items)

    def _apply_load_shedding(
        self,
        tenant_id: str,
        queue: Deque[QueuedRequest],
        execute_fn: Callable[[], Any],
        priority: int,
        timeout: float,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        if self._load_shedding_strategy == LoadSheddingStrategy.REJECT_NEW:
            return False, None, {"error": "queue_full", "strategy": "reject_new"}

        if self._load_shedding_strategy == LoadSheddingStrategy.DROP_OLDEST:
            dropped = queue.popleft()
            request_id = uuid.uuid4().hex[:12]
            request = QueuedRequest(
                request_id=request_id,
                tenant_id=tenant_id,
                priority=priority,
                enqueue_time=time.time(),
                timeout=timeout,
                execute_fn=execute_fn,
            )
            queue.append(request)
            self._sort_queue_by_priority(tenant_id)
            return True, request_id, {"queued": True, "dropped": dropped.request_id}

        if self._load_shedding_strategy == LoadSheddingStrategy.DROP_LOWEST_PRIORITY:
            if queue and queue[-1].priority < priority:
                dropped = queue.pop()
                request_id = uuid.uuid4().hex[:12]
                request = QueuedRequest(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    priority=priority,
                    enqueue_time=time.time(),
                    timeout=timeout,
                    execute_fn=execute_fn,
                )
                queue.append(request)
                self._sort_queue_by_priority(tenant_id)
                return True, request_id, {"queued": True, "dropped": dropped.request_id}
            return False, None, {"error": "queue_full", "strategy": "drop_lowest_priority"}

        if self._load_shedding_strategy == LoadSheddingStrategy.QUEUE_WITH_TIMEOUT:
            request_id = uuid.uuid4().hex[:12]
            request = QueuedRequest(
                request_id=request_id,
                tenant_id=tenant_id,
                priority=priority,
                enqueue_time=time.time(),
                timeout=timeout,
                execute_fn=execute_fn,
            )
            queue.append(request)
            self._sort_queue_by_priority(tenant_id)
            return True, request_id, {"queued": True, "warning": "queue_over_capacity"}

        return False, None, {"error": "queue_full"}

    def _ensure_queue_processor(self, tenant_id: str) -> None:
        if tenant_id in self._queue_processor_threads:
            return

        stop_event = threading.Event()
        self._queue_stop_events[tenant_id] = stop_event

        def processor():
            while not stop_event.is_set():
                request = None
                with self._queue_lock:
                    queue = self._request_queues.get(tenant_id, deque())
                    if queue:
                        now = time.time()
                        while queue and (now - queue[0].enqueue_time) > queue[0].timeout:
                            expired = queue.popleft()
                        if queue:
                            request = queue.popleft()

                if request:
                    try:
                        if self.acquire_concurrency(request.tenant_id, blocking=False, timeout=1.0):
                            try:
                                request.execute_fn()
                            finally:
                                self.release_concurrency(request.tenant_id)
                        else:
                            with self._queue_lock:
                                self._request_queues[request.tenant_id].appendleft(request)
                            time.sleep(0.1)
                    except Exception:
                        pass
                else:
                    time.sleep(0.05)

        thread = threading.Thread(target=processor, daemon=True, name=f"queue-processor-{tenant_id}")
        self._queue_processor_threads[tenant_id] = thread
        thread.start()

    def _stop_queue_processor(self, tenant_id: str) -> None:
        stop_event = self._queue_stop_events.pop(tenant_id, None)
        if stop_event:
            stop_event.set()
        thread = self._queue_processor_threads.pop(tenant_id, None)
        if thread:
            thread.join(timeout=1.0)

    def set_load_shedding_strategy(self, strategy: LoadSheddingStrategy) -> None:
        self._load_shedding_strategy = strategy

    def set_global_max_concurrent(self, max_concurrent: int) -> None:
        with self._global_active_lock:
            self._global_max_concurrent = max_concurrent

    def register_health_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        with self._health_check_lock:
            self._health_checks[name] = check_fn

    def unregister_health_check(self, name: str) -> bool:
        with self._health_check_lock:
            if name in self._health_checks:
                del self._health_checks[name]
                return True
            return False

    def run_health_checks(self) -> Dict[str, Any]:
        results = {}
        overall = HealthStatus.HEALTHY

        with self._health_check_lock:
            for name, check_fn in self._health_checks.items():
                try:
                    healthy = check_fn()
                    results[name] = {"healthy": healthy, "error": None}
                    if not healthy:
                        overall = HealthStatus.DEGRADED if overall == HealthStatus.HEALTHY else HealthStatus.UNHEALTHY
                except Exception as e:
                    results[name] = {"healthy": False, "error": str(e)}
                    overall = HealthStatus.UNHEALTHY

        with self._degradation_lock:
            if self._degradation_mode:
                overall = HealthStatus.DEGRADED

        return {
            "status": overall.value,
            "checks": results,
            "degradation_mode": self._degradation_mode,
            "timestamp": time.time(),
        }

    def set_degradation_mode(self, enabled: bool) -> None:
        with self._degradation_lock:
            self._degradation_mode = enabled

    def is_degraded(self) -> bool:
        with self._degradation_lock:
            return self._degradation_mode

    def get_health_summary(self) -> Dict[str, Any]:
        health = self.run_health_checks()

        with self._tenant_configs_lock:
            tenant_count = len(self._tenant_configs)

        with self._concurrency_lock:
            concurrency = {
                tid: getattr(sem, '_value', 'unknown')
                for tid, sem in self._concurrency_semaphores.items()
            }

        with self._circuit_breakers_lock:
            circuits = {name: cb.get_status() for name, cb in self._circuit_breakers.items()}

        with self._queue_lock:
            queue_stats = {
                tid: len(queue) for tid, queue in self._request_queues.items()
            }

        rate_limiter_backend = "redis" if self._rate_limiter.is_using_redis() else "memory"

        return {
            **health,
            "tenants_configured": tenant_count,
            "concurrency_slots": concurrency,
            "circuit_breakers": circuits,
            "queue_depths": queue_stats,
            "global_active": self._global_active_count,
            "global_max_concurrent": self._global_max_concurrent,
            "load_shedding_strategy": self._load_shedding_strategy.value,
            "rate_limiter_backend": rate_limiter_backend,
        }

    def get_tenant_status(self, tenant_id: str) -> Dict[str, Any]:
        config = self.get_tenant_config(tenant_id)
        if not config:
            return {"error": "tenant_not_configured"}

        with self._concurrency_lock:
            sem = self._concurrency_semaphores.get(tenant_id)
            available = getattr(sem, '_value', 0) if sem else 0

        with self._queue_lock:
            queue_depth = len(self._request_queues.get(tenant_id, deque()))

        rate_limit_info = self._rate_limiter.check_and_increment(
            f"tenant:{tenant_id}", config.rate_limit, 0
        )

        return {
            "tenant_id": tenant_id,
            "rate_limit": {
                "requests_per_window": config.rate_limit.requests_per_window,
                "window_seconds": config.rate_limit.window_seconds,
                "current_usage": self._rate_limiter.get_current_usage(f"tenant:{tenant_id}", config.rate_limit.window_seconds),
            },
            "concurrency": {
                "max": config.max_concurrent,
                "available": available,
                "used": config.max_concurrent - available,
            },
            "queue": {
                "max_size": config.max_queue_size,
                "current_depth": queue_depth,
            },
            "priority": config.priority,
        }

    def reset_tenant_limits(self, tenant_id: str) -> None:
        self._rate_limiter.reset(f"tenant:{tenant_id}", 60.0)
        with self._concurrency_lock:
            sem = self._concurrency_semaphores.get(tenant_id)
            if sem:
                while getattr(sem, '_value', 0) < self._tenant_configs.get(tenant_id, TenantQuotaConfig(RateLimitConfig(100, 60))).max_concurrent:
                    try:
                        sem.release()
                    except ValueError:
                        break

    def shutdown(self) -> None:
        for tenant_id in list(self._queue_stop_events.keys()):
            self._stop_queue_processor(tenant_id)


def get_scale_manager() -> ScaleManager:
    return ScaleManager.get_instance()


__all__ = [
    "ScaleManager",
    "get_scale_manager",
    "RateLimitConfig",
    "TenantQuotaConfig",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitState",
    "LoadSheddingStrategy",
    "HealthStatus",
    "SlidingWindowRateLimiter",
    "QueuedRequest",
]