from __future__ import annotations

import tempfile
from pathlib import Path

from alembic.config import Config

from alembic import command


def test_alembic_upgrade_head() -> None:
    """Run migrations on a temporary SQLite file to ensure they execute successfully."""
    # Alembic runs from the backend directory
    backend_dir = Path(__file__).resolve().parents[1]

    # TemporaryDirectory (not NamedTemporaryFile): on Windows an open temp file
    # cannot be reopened by sqlite.
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create Alembic Config
        alembic_cfg = Config(str(backend_dir / "alembic.ini"))
        # Set the root directory for migrations
        alembic_cfg.set_main_option("script_location", str(backend_dir / "alembic"))
        # Override the database URL to point to our temp file
        db_url = f"sqlite:///{Path(tmp_dir, 'test.sqlite3').as_posix()}"
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # Upgrade to head
        command.upgrade(alembic_cfg, "head")

        # Downgrade to base
        command.downgrade(alembic_cfg, "base")
