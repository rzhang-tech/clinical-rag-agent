import re
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import config
from db.postgres_manager import PostgresManager

logger = logging.getLogger(__name__)


class ParentStoreManager:
    """
    Stores parent chunks in PostgreSQL when available.
    Falls back to JSON files on disk when PostgreSQL is unreachable,
    so local development without Docker still works.
    """

    def __init__(self, pg: Optional[PostgresManager] = None, fallback_path: str = config.PARENT_STORE_PATH):
        self._pg = pg or PostgresManager()
        self._fallback_path = Path(fallback_path)
        self._use_pg = False
        self._connect()

    def _connect(self) -> None:
        try:
            self._pg.connect()
            self._use_pg = True
            logger.info("ParentStoreManager using PostgreSQL")
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s) — falling back to JSON file store", exc)
            self._use_pg = False
            self._fallback_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        if self._use_pg:
            self._pg.save_parent(parent_id, content, metadata)
        else:
            file_path = self._fallback_path / f"{parent_id}.json"
            file_path.write_text(
                json.dumps({"page_content": content, "metadata": metadata}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def save_many(self, parents: List) -> None:
        """parents: list of (parent_id, LangChain Document)"""
        if self._use_pg:
            records = [(pid, doc.page_content, doc.metadata) for pid, doc in parents]
            self._pg.save_many_parents(records)
        else:
            for pid, doc in parents:
                self.save(pid, doc.page_content, doc.metadata)

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    def load_content(self, parent_id: str) -> Optional[Dict]:
        if self._use_pg:
            return self._pg.load_parent(parent_id)
        # JSON fallback
        file_path = self._fallback_path / (
            parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        )
        if not file_path.exists():
            return None
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return {"content": data["page_content"], "parent_id": parent_id, "metadata": data["metadata"]}

    @staticmethod
    def _sort_key(pid: str) -> int:
        match = re.search(r'_parent_(\d+)$', pid)
        return int(match.group(1)) if match else 0

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        unique = list(dict.fromkeys(parent_ids))
        if self._use_pg:
            results = self._pg.load_many_parents(unique)
        else:
            results = [r for pid in unique if (r := self.load_content(pid)) is not None]
        results.sort(key=lambda r: self._sort_key(r["parent_id"]))
        return results

    # ------------------------------------------------------------------ #
    # Clear                                                                #
    # ------------------------------------------------------------------ #

    def clear_store(self) -> None:
        if self._use_pg:
            self._pg.clear_parents()
        else:
            if self._fallback_path.exists():
                shutil.rmtree(self._fallback_path)
            self._fallback_path.mkdir(parents=True, exist_ok=True)
