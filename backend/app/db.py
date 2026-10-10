"""Database engine, session factory and declarative base (SQLAlchemy 2.x)."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def make_engine(url: str, **kwargs: object) -> Engine:
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        # FastAPI may use the session from a different thread than the one that created it.
        connect_args["check_same_thread"] = False
    engine = create_engine(url, connect_args=connect_args, **kwargs)
    if url.startswith("sqlite"):
        # SQLite does not enforce foreign keys unless asked to.
        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine = make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session (overridden in tests)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
