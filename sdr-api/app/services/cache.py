"""In-memory cache for run and comparison results."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CacheEntry:
    data: Any
    created_at: float = field(default_factory=time.time)
    status: str = "complete"


class RunCache:
    def __init__(self, ttl_seconds: Optional[int] = None):
        self.ttl = ttl_seconds or int(os.environ.get("RUN_CACHE_TTL_SECONDS", "900"))
        self._runs: dict[str, CacheEntry] = {}
        self._comparisons: dict[str, CacheEntry] = {}

    def new_id(self) -> str:
        return str(uuid.uuid4())

    def set_run(self, run_id: str, data: dict, status: str = "complete") -> None:
        self._runs[run_id] = CacheEntry(data=data, status=status)

    def get_run(self, run_id: str) -> Optional[CacheEntry]:
        entry = self._runs.get(run_id)
        if entry is None:
            return None
        if time.time() - entry.created_at > self.ttl:
            del self._runs[run_id]
            return None
        return entry

    def update_run_status(self, run_id: str, status: str, data: Optional[dict] = None) -> None:
        entry = self._runs.get(run_id)
        if entry is None:
            self._runs[run_id] = CacheEntry(data=data or {}, status=status)
        else:
            entry.status = status
            if data is not None:
                entry.data = data

    def set_comparison(self, comparison_id: str, data: dict) -> None:
        self._comparisons[comparison_id] = CacheEntry(data=data)

    def get_comparison(self, comparison_id: str) -> Optional[CacheEntry]:
        entry = self._comparisons.get(comparison_id)
        if entry is None:
            return None
        if time.time() - entry.created_at > self.ttl:
            del self._comparisons[comparison_id]
            return None
        return entry


cache = RunCache()
