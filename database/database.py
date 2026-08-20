import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import app.logger  # noqa: F401
from app.config import DATABASE_URL
from database.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
        )
        Base.metadata.create_all(_engine)
        logger.info("Database initialised: %s", DATABASE_URL)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal()
