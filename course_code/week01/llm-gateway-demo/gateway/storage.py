from __future__ import annotations

import asyncio
from contextlib import contextmanager
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from gateway.schemas import UsageRecord


def default_data_path() -> str:
    """Return an external app-data path, never a path inside this repository."""

    root = Path(os.getenv("LOCALAPPDATA", tempfile.gettempdir())) / "llm-gateway"
    return str(root / "gateway.sqlite3")


class GatewayStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or default_data_path()
        if self.db_path != ":memory:":
            resolved = Path(self.db_path).expanduser().resolve()
            workspace = Path(__file__).resolve().parents[1]
            if resolved == workspace or resolved.is_relative_to(workspace):
                raise ValueError("Gateway storage must be outside the workspace.")
            self.db_path = str(resolved)

    def _connect(self) -> sqlite3.Connection:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS templates (
                    name TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    stream INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    error_code TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    ttft_ms REAL,
                    retries INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_usage_model_created
                    ON usage_records(model, created_at DESC);
                """
            )

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize)

    def _upsert_template(self, name: str, content: str, description: str) -> dict[str, Any]:
        now = time.time()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT created_at FROM templates WHERE name = ?", (name,)
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            connection.execute(
                """
                INSERT INTO templates(name, content, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    content = excluded.content,
                    description = excluded.description,
                    updated_at = excluded.updated_at
                """,
                (name, content, description, created_at, now),
            )
        return {
            "name": name,
            "content": content,
            "description": description,
            "created_at": created_at,
            "updated_at": now,
        }

    async def upsert_template(self, name: str, content: str, description: str = "") -> dict[str, Any]:
        return await asyncio.to_thread(self._upsert_template, name, content, description)

    def _get_template(self, name: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM templates WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    async def get_template(self, name: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_template, name)

    def _list_templates(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM templates ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    async def list_templates(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_templates)

    def _delete_template(self, name: str) -> bool:
        with self._connection() as connection:
            result = connection.execute("DELETE FROM templates WHERE name = ?", (name,))
        return result.rowcount > 0

    async def delete_template(self, name: str) -> bool:
        return await asyncio.to_thread(self._delete_template, name)

    def _record_usage(self, record: UsageRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO usage_records(
                    request_id, model, provider, endpoint, stream, status, status_code,
                    error_code, prompt_tokens, completion_tokens, reasoning_tokens,
                    cached_prompt_tokens, cache_creation_tokens, total_tokens,
                    latency_ms, ttft_ms, retries, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.model,
                    record.provider,
                    record.endpoint,
                    int(record.stream),
                    record.status,
                    record.status_code,
                    record.error_code,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.reasoning_tokens,
                    record.cached_prompt_tokens,
                    record.cache_creation_tokens,
                    record.total_tokens,
                    record.latency_ms,
                    record.ttft_ms,
                    record.retries,
                    record.created_at,
                ),
            )

    async def record_usage(self, record: UsageRecord) -> None:
        await asyncio.to_thread(self._record_usage, record)

    def _usage(self, model: str | None, limit: int) -> dict[str, Any]:
        with self._connection() as connection:
            if model:
                rows = connection.execute(
                    "SELECT * FROM usage_records WHERE model = ? ORDER BY created_at DESC LIMIT ?",
                    (model, limit),
                ).fetchall()
                summary_rows = connection.execute(
                    """
                    SELECT model, provider, COUNT(*) AS calls,
                           SUM(prompt_tokens) AS prompt_tokens,
                           SUM(completion_tokens) AS completion_tokens,
                           SUM(reasoning_tokens) AS reasoning_tokens,
                           SUM(cached_prompt_tokens) AS cached_prompt_tokens,
                           SUM(total_tokens) AS total_tokens,
                           AVG(latency_ms) AS avg_latency_ms,
                           AVG(ttft_ms) AS avg_ttft_ms
                    FROM usage_records WHERE model = ? GROUP BY model, provider
                    """,
                    (model,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM usage_records ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
                summary_rows = connection.execute(
                    """
                    SELECT model, provider, COUNT(*) AS calls,
                           SUM(prompt_tokens) AS prompt_tokens,
                           SUM(completion_tokens) AS completion_tokens,
                           SUM(reasoning_tokens) AS reasoning_tokens,
                           SUM(cached_prompt_tokens) AS cached_prompt_tokens,
                           SUM(total_tokens) AS total_tokens,
                           AVG(latency_ms) AS avg_latency_ms,
                           AVG(ttft_ms) AS avg_ttft_ms
                    FROM usage_records GROUP BY model, provider
                    """
                ).fetchall()
        return {
            "records": [dict(row) for row in rows],
            "summary": [dict(row) for row in summary_rows],
        }

    async def usage(self, model: str | None = None, limit: int = 100) -> dict[str, Any]:
        return await asyncio.to_thread(self._usage, model, min(max(limit, 1), 500))
