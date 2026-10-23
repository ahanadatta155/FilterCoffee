"""Small SQLite cache; each operation uses its own WAL-enabled connection."""

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Cache:
    def __init__(
        self, path: str | Path = "data/cache.db", *, clock: Callable[[], float] = time.time
    ):
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS pages (
                    url_hash TEXT PRIMARY KEY, url TEXT NOT NULL, html TEXT NOT NULL,
                    fetched_at REAL NOT NULL, tier TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS results (
                    key TEXT PRIMARY KEY, result_json TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm (
                    prompt_hash TEXT PRIMARY KEY, response TEXT NOT NULL, model TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _get(
        self, table: str, key_column: str, key: str, value_columns: str, time_column: str, ttl: int
    ):
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {value_columns}, {time_column} FROM {table} WHERE {key_column}=?", (key,)
            ).fetchone()
            if row is None:
                return None
            if self.clock() - row[-1] >= ttl:
                connection.execute(f"DELETE FROM {table} WHERE {key_column}=?", (key,))
                return None
            return row[:-1]

    def get_page(self, url: str):
        return self._get("pages", "url_hash", digest(url), "html, tier", "fetched_at", 6 * 3600)

    def set_page(self, url: str, html: str, tier: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?)",
                (digest(url), url, html, self.clock(), tier),
            )

    def get_result(self, key: str) -> dict[str, Any] | None:
        row = self._get("results", "key", key, "result_json", "created_at", 24 * 3600)
        return json.loads(row[0]) if row else None

    def set_result(self, key: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO results VALUES (?, ?, ?)",
                (key, json.dumps(result, ensure_ascii=False), self.clock()),
            )

    def get_llm(self, prompt_hash: str):
        return self._get(
            "llm", "prompt_hash", prompt_hash, "response, model", "created_at", 7 * 24 * 3600
        )

    def set_llm(self, prompt_hash: str, response: str, model: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO llm VALUES (?, ?, ?, ?)",
                (prompt_hash, response, model, self.clock()),
            )

    def clear(self) -> None:
        with self._connect() as connection:
            for table in ("pages", "results", "llm"):
                connection.execute(f"DELETE FROM {table}")
