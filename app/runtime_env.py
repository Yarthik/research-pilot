"""Start-up helpers for hosts such as Streamlit Community Cloud (pure functions, unit-tested).

researchpilot reads its settings from environment variables. Community Cloud keeps them in the app's
secrets (TOML) and, at server start, copies root-level *string, int and float* values into
os.environ. It leaves out booleans: `ALLOW_INGEST = false` would silently not arrive, and the public
demo would keep its upload and delete buttons. export_secrets_to_env closes that gap.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

MIN_SQLITE = (3, 35, 0)  # ChromaDB refuses to start on anything older


def export_secrets_to_env(secrets: Mapping, environ: dict | None = None) -> list[str]:
    """Copy root-level scalar secrets into `environ` (default os.environ); returns the names it set.

    Values that are already set win, so a host's own environment variables and Streamlit's own copy
    are never overwritten. Booleans become "true" / "false". Tables (`[section]`) are ignored.
    """
    environ = os.environ if environ is None else environ
    exported = []
    for name, value in secrets.items():
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            text = str(value)
        else:
            continue
        if name not in environ:
            environ[name] = text
            exported.append(name)
    return exported


def load_streamlit_secrets(st_secrets) -> list[str]:
    """export_secrets_to_env for `st.secrets`, which raises when the app has no secrets at all."""
    try:
        return export_secrets_to_env(dict(st_secrets))
    except Exception:  # no secrets.toml (local run) or an unreadable one: settings come from .env / env
        return []


def ensure_modern_sqlite(minimum: tuple = MIN_SQLITE) -> None:
    """Make sure ChromaDB will find SQLite >= 3.35, swapping in `pysqlite3` when the system one is older.

    Some hosts ship an old system SQLite. Call this before anything imports chromadb. Raises a
    RuntimeError that says how to fix it if neither is usable.
    """
    import sqlite3

    if sqlite3.sqlite_version_info >= minimum:
        return
    try:
        import pysqlite3
    except ImportError as exc:
        found = ".".join(map(str, sqlite3.sqlite_version_info))
        wanted = ".".join(map(str, minimum))
        raise RuntimeError(
            f"This host's SQLite is {found} but ChromaDB needs {wanted} or newer. "
            "Add `pysqlite3-binary` to requirements.txt and redeploy."
        ) from exc
    sys.modules["sqlite3"] = pysqlite3
