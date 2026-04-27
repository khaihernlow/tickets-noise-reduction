import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

DB_PATH = Path(__file__).parent.parent.parent / "data" / "nrc.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_number  TEXT PRIMARY KEY,
    title          TEXT,
    description    TEXT,
    account        TEXT,
    resources      TEXT,
    status         TEXT,
    created        TEXT,
    total_hours    REAL,
    billed_hours   REAL,
    sub_issue_type TEXT,
    issue_type     TEXT,
    imported_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS recommendation_cache (
    cache_key      TEXT PRIMARY KEY,
    pattern_type   TEXT NOT NULL,
    account        TEXT NOT NULL,
    issue_type     TEXT NOT NULL,
    ticket_numbers TEXT NOT NULL,   -- JSON array
    result_json    TEXT NOT NULL,   -- full recommendation dict from LLM
    model          TEXT NOT NULL,
    created_at     TEXT DEFAULT (datetime('now'))
);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ── ticket store ───────────────────────────────────────────────────────────────

def import_tickets(df: pd.DataFrame, conn: sqlite3.Connection) -> tuple[int, int]:
    """Insert new tickets from a DataFrame, silently skip duplicates.

    Returns (new_count, skipped_count).
    """
    new_count = 0
    skipped_count = 0

    for _, row in df.iterrows():
        try:
            conn.execute(
                """
                INSERT INTO tickets
                    (ticket_number, title, description, account, resources,
                     status, created, total_hours, billed_hours,
                     sub_issue_type, issue_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["ticket_number"],
                    row["title"],
                    row["description"],
                    row["account"],
                    row["resources"],
                    row["status"],
                    str(row["created"]),
                    float(row["total_hours"]),
                    float(row["billed_hours"]),
                    row["sub_issue_type"],
                    row["issue_type"],
                ),
            )
            new_count += 1
        except sqlite3.IntegrityError:
            skipped_count += 1

    conn.commit()
    return new_count, skipped_count


def load_all_tickets(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load every ticket from the store as a DataFrame."""
    rows = conn.execute("SELECT * FROM tickets ORDER BY created").fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["created"] = pd.to_datetime(df["created"], errors="coerce")
    df["total_hours"] = pd.to_numeric(df["total_hours"], errors="coerce").fillna(0.0)
    df["billed_hours"] = pd.to_numeric(df["billed_hours"], errors="coerce").fillna(0.0)
    return df


def ticket_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


# ── recommendation cache ───────────────────────────────────────────────────────

def _cache_key(pattern_type: str, account: str, issue_type: str,
               ticket_numbers: list[str], model: str) -> str:
    """Stable SHA-256 fingerprint for a pattern cluster.

    The key changes whenever the set of tickets changes or the model changes,
    which automatically invalidates stale cache entries.
    """
    canonical = f"{pattern_type}|{account}|{issue_type}|{','.join(sorted(ticket_numbers))}|{model}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def cache_get(pattern_type: str, account: str, issue_type: str,
              ticket_numbers: list[str], model: str,
              conn: sqlite3.Connection) -> Optional[dict]:
    """Return cached recommendation dict, or None if not cached."""
    key = _cache_key(pattern_type, account, issue_type, ticket_numbers, model)
    row = conn.execute(
        "SELECT result_json FROM recommendation_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    return json.loads(row["result_json"]) if row else None


def cache_set(pattern_type: str, account: str, issue_type: str,
              ticket_numbers: list[str], model: str,
              result: dict, conn: sqlite3.Connection) -> None:
    """Store a recommendation result in the cache."""
    key = _cache_key(pattern_type, account, issue_type, ticket_numbers, model)
    conn.execute(
        """
        INSERT OR REPLACE INTO recommendation_cache
            (cache_key, pattern_type, account, issue_type,
             ticket_numbers, result_json, model)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key,
            pattern_type,
            account,
            issue_type,
            json.dumps(sorted(ticket_numbers)),
            json.dumps(result),
            model,
        ),
    )
    conn.commit()


def cache_clear(conn: sqlite3.Connection) -> int:
    """Delete all cached recommendations. Returns number of rows deleted."""
    cursor = conn.execute("DELETE FROM recommendation_cache")
    conn.commit()
    return cursor.rowcount


def cache_stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM recommendation_cache").fetchone()[0]
    return {"cached_recommendations": total}
