"""
Database connection, session management, and schema verification for PostgreSQL + pgvector.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Any
import psycopg
from pgvector.psycopg import register_vector
from ragcli.config import settings

logger = logging.getLogger(__name__)


def get_connection(dbname: str | None = None) -> psycopg.Connection:
    """Create and return a raw psycopg connection."""
    conn_str = settings.database_url
    if dbname:
        auth = settings.db_user
        if settings.db_password:
            auth += f":{settings.db_password}"
        conn_str = f"postgresql://{auth}@{settings.db_host}:{settings.db_port}/{dbname}"

    conn = psycopg.connect(conn_str, autocommit=False)
    register_vector(conn)
    return conn


@contextmanager
def get_db_cursor(dbname: str | None = None) -> Generator[psycopg.Cursor, None, None]:
    """Provide a transactional cursor context."""
    conn = get_connection(dbname)
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        conn.close()


def check_db_health() -> dict[str, Any]:
    """Check connectivity, PostgreSQL version, and pgvector extension status."""
    result: dict[str, Any] = {
        "connected": False,
        "database": settings.db_name,
        "pg_version": None,
        "pgvector_installed": False,
        "pgvector_version": None,
        "tables": [],
        "errors": [],
    }

    try:
        with get_db_cursor() as cur:
            result["connected"] = True
            
            # Check version
            cur.execute("SELECT version();")
            row = cur.fetchone()
            if row:
                result["pg_version"] = row[0]

            # Check pgvector
            cur.execute(
                "SELECT installed_version FROM pg_available_extensions WHERE name = 'vector';"
            )
            vrow = cur.fetchone()
            if vrow and vrow[0]:
                result["pgvector_installed"] = True
                result["pgvector_version"] = vrow[0]

            # List tables
            cur.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
                """
            )
            result["tables"] = [r[0] for r in cur.fetchall()]

    except Exception as e:
        result["errors"].append(str(e))

    return result


def init_database() -> bool:
    """Execute init_db.sql to initialize database schemas and extensions."""
    import os
    sql_path = os.path.join(os.path.dirname(__file__), "../../scripts/init_db.sql")
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"Database init script not found at {sql_path}")

    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    return True
