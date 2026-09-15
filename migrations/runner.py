"""Migration runner with Redis-backed version tracking."""

import json
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

from config import settings

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("crossmind.migrations")


class Migration:
    def __init__(self, version: int, description: str, up: Callable[[], None], down: Optional[Callable[[], None]] = None) -> None:
        self.version = version
        self.description = description
        self.up = up
        self.down = down


class MigrationRunner:
    def __init__(self) -> None:
        self._redis_client = None
        if REDIS_AVAILABLE:
            try:
                self._redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD.get_secret_value() or None,
                    decode_responses=True,
                )
                self._redis_client.ping()
            except Exception:
                self._redis_client = None

    def _get_version_key(self) -> str:
        return "crossmind:migrations:version"

    def get_current_version(self) -> int:
        if not self._redis_client:
            return 0
        value = self._redis_client.get(self._get_version_key())
        return int(value) if value else 0

    def set_version(self, version: int) -> None:
        if self._redis_client:
            self._redis_client.set(self._get_version_key(), str(version))

    def run(self, migrations: List[Migration]) -> List[int]:
        current = self.get_current_version()
        applied = []
        for migration in sorted(migrations, key=lambda m: m.version):
            if migration.version > current:
                logger.info("Running migration %s: %s", migration.version, migration.description)
                try:
                    migration.up()
                    self.set_version(migration.version)
                    applied.append(migration.version)
                except Exception as exc:
                    logger.error("Migration %s failed: %s", migration.version, exc)
                    raise
        return applied

    def rollback(self, migrations: List[Migration], target_version: int) -> List[int]:
        current = self.get_current_version()
        rolled_back = []
        for migration in sorted(migrations, key=lambda m: m.version, reverse=True):
            if migration.down and migration.version > target_version and migration.version <= current:
                logger.info("Rolling back migration %s", migration.version)
                try:
                    migration.down()
                    rolled_back.append(migration.version)
                except Exception as exc:
                    logger.error("Rollback %s failed: %s", migration.version, exc)
                    raise
        if rolled_back:
            self.set_version(target_version)
        return rolled_back

    def status(self, migrations: List[Migration]) -> Dict[str, Any]:
        current = self.get_current_version()
        pending = [m for m in sorted(migrations, key=lambda m: m.version) if m.version > current]
        return {
            "current_version": current,
            "pending_count": len(pending),
            "pending": [{"version": m.version, "description": m.description} for m in pending],
            "timestamp": datetime.utcnow().isoformat(),
        }


__all__ = ["Migration", "MigrationRunner"]
