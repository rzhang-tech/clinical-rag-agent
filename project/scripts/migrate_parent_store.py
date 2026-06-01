"""
One-time migration: load all JSON parent store files into PostgreSQL.
Run once after switching from the file-based store to PostgreSQL.

Usage:
    python scripts/migrate_parent_store.py
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import config
from db.postgres_manager import PostgresManager


def migrate():
    store_path = Path(config.PARENT_STORE_PATH)
    json_files = sorted(store_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {store_path}")
        return

    print(f"Found {len(json_files)} parent chunks to migrate...")

    pg = PostgresManager()
    pg.connect()

    batch = []
    for i, f in enumerate(json_files):
        data = json.loads(f.read_text(encoding="utf-8"))
        parent_id = f.stem
        content = data.get("page_content", "")
        metadata = data.get("metadata", {})
        batch.append((parent_id, content, metadata))

        if len(batch) == 200:
            pg.save_many_parents(batch)
            print(f"  Inserted {i + 1}/{len(json_files)}...")
            batch = []

    if batch:
        pg.save_many_parents(batch)

    pg.close()
    print(f"Done — {len(json_files)} parent chunks migrated to PostgreSQL.")


if __name__ == "__main__":
    migrate()
