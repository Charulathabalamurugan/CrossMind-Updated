"""Ensure Redis keyspace prefixes and TTL defaults are set."""

from migrations.runner import Migration


def up() -> None:
    try:
        import redis
        from config import settings

        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD.get_secret_value() or None,
            decode_responses=True,
        )
        client.config_set("notify-keyspace-events", "KEA")
        client.set("crossmind:init:completed", "1", ex=3600)
    except Exception as exc:
        raise RuntimeError(f"Redis keyspace migration failed: {exc}")


def down() -> None:
    pass


migration = Migration(
    version=3,
    description="Redis keyspace configuration",
    up=up,
    down=down,
)
