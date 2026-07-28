import threading
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from config import settings

logger = logging.getLogger("crossmind.queue_manager")

class IngestionTask:
    def __init__(
        self,
        task_id: str,
        documents: List[Dict[str, Any]],
        priority: int = 0,
        source: str = "api",
        callback: Optional[Callable] = None,
    ):
        self.task_id = task_id
        self.documents = documents
        self.priority = priority
        self.source = source
        self.callback = callback
        self.status = "queued"
        self.created_at = time.time()
        self.completed_at = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

class QueueManager:
    def __init__(self):
        self._queue: List[IngestionTask] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self.max_batch_size = settings.INGESTION_BATCH_SIZE
        self.max_retries = settings.INGESTION_MAX_RETRIES
        self._total_processed = 0
        self._total_failed = 0

    def enqueue(self, documents: List[Dict[str, Any]], source: str = "api", priority: int = 0, callback: Optional[Callable] = None) -> str:
        import uuid
        task_id = str(uuid.uuid4())
        task = IngestionTask(task_id=task_id, documents=documents, priority=priority, source=source, callback=callback)
        with self._lock:
            self._queue.append(task)
            self._queue.sort(key=lambda t: t.priority, reverse=True)
            self._condition.notify()
        logger.info(f"Enqueued ingestion task {task_id} with {len(documents)} documents (source={source})")
        return task_id

    def start(self, process_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]]):
        if self._running:
            logger.warning("Queue manager already running.")
            return
        self._running = True
        self._process_fn = process_fn
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Queue manager worker started.")

    def _worker_loop(self):
        while self._running:
            task = None
            with self._lock:
                while not self._queue and self._running:
                    self._condition.wait(timeout=1.0)
                if not self._running:
                    break
                if self._queue:
                    task = self._queue.pop(0)
            if task:
                self._process_task(task)

    def _process_task(self, task: IngestionTask):
        task.status = "processing"
        logger.info(f"Processing task {task.task_id} ({len(task.documents)} docs)")
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._process_fn(task.documents)
                task.result = result
                task.status = "completed"
                task.completed_at = time.time()
                self._total_processed += 1
                logger.info(f"Task {task.task_id} completed on attempt {attempt}: {len(result.get('inserted_ids', []))} docs")
                if task.callback:
                    try:
                        task.callback(task.task_id, result)
                    except Exception:
                        pass
                return
            except Exception as exc:
                logger.error(f"Task {task.task_id} failed on attempt {attempt}: {exc}")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
        task.status = "failed"
        task.error = str(exc)
        task.completed_at = time.time()
        self._total_failed += 1

    def stop(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info("Queue manager worker stopped.")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queued": len(self._queue),
                "total_processed": self._total_processed,
                "total_failed": self._total_failed,
                "running": self._running,
            }

_queue_manager_instance = None

def get_queue_manager() -> QueueManager:
    global _queue_manager_instance
    if _queue_manager_instance is None:
        _queue_manager_instance = QueueManager()
    return _queue_manager_instance