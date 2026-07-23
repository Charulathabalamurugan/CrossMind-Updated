import os
import logging
import threading
import time
import hashlib
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger("crossmind.dynamic_connectors")

class BaseConnector:
    """Base class for all ingestion connectors."""
    connector_type: str = "base"

    def __init__(self, name: str):
        self.name = name
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def fetch(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def start(self, callback):
        self._running = True
        def _loop():
            while self._running:
                try:
                    items = self.fetch()
                    if items:
                        callback(items)
                except Exception as exc:
                    logger.warning(f"Connector {self.name} fetch error: {exc}")
                time.sleep(settings.CONTINUOUS_INGESTION_INTERVAL)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started dynamic connector: {self.name}")

    def stop(self):
        self._running = False

class FileConnector(BaseConnector):
    connector_type = "file"

    def __init__(self, name: str, watch_paths: List[str]):
        super().__init__(name)
        self.watch_paths = watch_paths
        self._seen_hashes = set()

    def fetch(self) -> List[Dict[str, Any]]:
        documents = []
        for path in self.watch_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
            content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            if content_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(content_hash)
            documents.append({
                "title": os.path.basename(path),
                "content": raw,
                "domain": "general",
                "source": self.name,
                "content_hash": content_hash,
            })
        return documents

class ApiConnector(BaseConnector):
    connector_type = "api"

    def __init__(self, name: str, endpoint: str, headers: Optional[Dict[str, str]] = None):
        super().__init__(name)
        self.endpoint = endpoint
        self.headers = headers or {}
        self._last_state = ""

    def fetch(self) -> List[Dict[str, Any]]:
        try:
            import requests
            resp = requests.get(self.endpoint, headers=self.headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            state = str(hash(str(data)))
            if state == self._last_state:
                return []
            self._last_state = state
            if isinstance(data, list):
                return [{**item, "source": self.name} for item in data]
            if isinstance(data, dict) and "items" in data:
                return [{**item, "source": self.name} for item in data["items"]]
            return [{**data, "source": self.name}]
        except Exception as exc:
            logger.warning(f"ApiConnector {self.name} fetch error: {exc}")
            return []

class WebhookConnector(BaseConnector):
    connector_type = "webhook"

    def __init__(self, name: str, queue: "DynamicConnectorQueue"):
        super().__init__(name)
        self.queue = queue

    def enqueue(self, payload: Dict[str, Any]):
        self.queue.put({"source": self.name, "payload": payload})

    def fetch(self) -> List[Dict[str, Any]]:
        return []

class DynamicConnectorQueue:
    def __init__(self):
        import queue
        self._queue = queue.Queue()

    def put(self, item: Dict[str, Any]):
        self._queue.put(item)

    def drain(self) -> List[Dict[str, Any]]:
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except Exception:
                break
        return items

class DynamicConnectorManager:
    def __init__(self):
        self.queue = DynamicConnectorQueue()
        self.connectors: Dict[str, BaseConnector] = {}
        self._lock = threading.Lock()

    def register(self, connector: BaseConnector):
        with self._lock:
            self.connectors[connector.name] = connector

    def start_all(self, callback):
        for connector in self.connectors.values():
            connector.start(callback)

    def stop_all(self):
        for connector in self.connectors.values():
            connector.stop()

    def load_from_env(self, callback):
        if not settings.DYNAMIC_CONNECTORS_ENABLED:
            return
        sources = [s.strip() for s in settings.DYNAMIC_CONNECTOR_SOURCES.split(",") if s.strip()]
        for source in sources:
            if source == "file":
                watch_paths = os.getenv("DYNAMIC_FILE_PATHS", "").split(",")
                watch_paths = [p.strip() for p in watch_paths if p.strip()]
                if watch_paths:
                    self.register(FileConnector("file_connector", watch_paths))
            elif source == "api":
                endpoint = os.getenv("DYNAMIC_API_ENDPOINT", "")
                if endpoint:
                    self.register(ApiConnector("api_connector", endpoint))
            elif source == "webhook":
                self.register(WebhookConnector("webhook_connector", self.queue))
                logger.info("Registered webhook connector. Use `DynamicConnectorManager.queue.put()` to inject documents.")

_dynamic_connector_manager: Optional[DynamicConnectorManager] = None

def get_dynamic_connectors() -> DynamicConnectorManager:
    global _dynamic_connector_manager
    if _dynamic_connector_manager is None:
        _dynamic_connector_manager = DynamicConnectorManager()
    return _dynamic_connector_manager
