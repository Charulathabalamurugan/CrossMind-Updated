"""CrossMind FastAPI application entrypoint."""
import logging
from fastapi import FastAPI

from app.observability import configure_logging

configure_logging()
logger = logging.getLogger("crossmind.api")