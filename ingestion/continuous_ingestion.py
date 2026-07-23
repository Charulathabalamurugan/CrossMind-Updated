import threading
import time
import logging
from typing import List, Dict, Any, Callable, Optional
from config import settings

logger = logging.getLogger("crossmind.continuous_ingestion")

class ContinuousIngestionWorker:
    def __init__(self, pipeline, callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None):
        self.pipeline = pipeline
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.interval = settings.CONTINUOUS_INGESTION_INTERVAL
        self.batch_size = settings.INGESTION_BATCH_SIZE
        self.max_retries = settings.INGESTION_MAX_RETRIES

    def _process_batch(self, items: List[Dict[str, Any]]):
        if not items:
            return
        for attempt in range(1, self.max_retries + 1):
            try:
                inserted = self.pipeline.ingest_documents(items[: self.batch_size])
                logger.info(f"Continuous ingestion batch processed: {len(inserted)} docs (attempt {attempt})")
                if self.callback:
                    self.callback(inserted)
                return
            except Exception as exc:
                logger.error(f"Ingestion batch failed on attempt {attempt}: {exc}")
                time.sleep(2 ** attempt)

    def start(self):
        self._running = True

        def _loop():
            while self._running:
                try:
                    connector_items = []
                    try:
                        from ingestion.dynamic_connectors import get_dynamic_connectors
                        manager = get_dynamic_connectors()
                        connector_items = manager.queue.drain()
                    except Exception as exc:
                        logger.debug(f"No connector items available: {exc}")

                    self._process_batch(connector_items)
                except Exception as exc:
                    logger.error(f"Continuous ingestion loop error: {exc}")
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.info("Started continuous ingestion worker.")

    def stop(self):
        self._running = False
