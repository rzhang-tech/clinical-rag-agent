import json
import logging
from typing import Dict, List, Optional
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

import config

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id   TEXT PRIMARY KEY,
    content     TEXT        NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id              TEXT PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    num_in_kb           INT,
    num_out_kb          INT,
    in_kb_accuracy      FLOAT,
    out_kb_refusal_rate FLOAT,
    overall_accuracy    FLOAT,
    avg_time_seconds    FLOAT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    question_index  INT,
    category        TEXT,
    question        TEXT,
    correct_answer  TEXT,
    agent_choice    TEXT,
    is_correct      BOOLEAN,
    is_refusal      BOOLEAN,
    has_sources     BOOLEAN,
    elapsed_seconds FLOAT,
    response        TEXT
);
"""


class PostgresManager:

    def __init__(self, url: str = config.POSTGRES_URL):
        self._url = url
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        try:
            self._conn = psycopg2.connect(self._url)
            self._conn.autocommit = False
            with self._conn.cursor() as cur:
                cur.execute(_DDL)
            self._conn.commit()
            logger.info("PostgreSQL connected and schema initialised")
        except Exception as exc:
            logger.error("PostgreSQL connection failed: %s", exc)
            self._conn = None
            raise

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def _cursor(self):
        if self._conn is None or self._conn.closed:
            self.connect()
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------ #
    # Parent chunk CRUD                                                    #
    # ------------------------------------------------------------------ #

    def save_parent(self, parent_id: str, content: str, metadata: Dict) -> None:
        sql = """
            INSERT INTO parent_chunks (parent_id, content, metadata)
            VALUES (%s, %s, %s)
            ON CONFLICT (parent_id) DO UPDATE
                SET content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
        """
        with self._cursor() as cur:
            cur.execute(sql, (parent_id, content, json.dumps(metadata)))

    def save_many_parents(self, parents: List[tuple]) -> None:
        """parents: list of (parent_id, content, metadata_dict)"""
        sql = """
            INSERT INTO parent_chunks (parent_id, content, metadata)
            VALUES %s
            ON CONFLICT (parent_id) DO UPDATE
                SET content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
        """
        records = [(pid, content, json.dumps(meta)) for pid, content, meta in parents]
        with self._cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, records)

    def load_parent(self, parent_id: str) -> Optional[Dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT content, metadata FROM parent_chunks WHERE parent_id = %s",
                (parent_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"content": row["content"], "parent_id": parent_id, "metadata": row["metadata"]}

    def load_many_parents(self, parent_ids: List[str]) -> List[Dict]:
        if not parent_ids:
            return []
        with self._cursor() as cur:
            cur.execute(
                "SELECT parent_id, content, metadata FROM parent_chunks WHERE parent_id = ANY(%s)",
                (list(parent_ids),),
            )
            rows = cur.fetchall()
        id_order = {pid: i for i, pid in enumerate(parent_ids)}
        results = [
            {"content": r["content"], "parent_id": r["parent_id"], "metadata": r["metadata"]}
            for r in rows
        ]
        results.sort(key=lambda r: id_order.get(r["parent_id"], 9999))
        return results

    def clear_parents(self) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM parent_chunks")

    # ------------------------------------------------------------------ #
    # Evaluation metrics                                                   #
    # ------------------------------------------------------------------ #

    def save_eval_run(self, run_id: str, timestamp: str, metrics: Dict) -> None:
        sql = """
            INSERT INTO eval_runs (
                run_id, timestamp, num_in_kb, num_out_kb,
                in_kb_accuracy, out_kb_refusal_rate, overall_accuracy, avg_time_seconds
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO NOTHING
        """
        with self._cursor() as cur:
            cur.execute(sql, (
                run_id, timestamp,
                metrics.get("num_in_kb"),
                metrics.get("num_out_kb"),
                metrics.get("in_kb_accuracy"),
                metrics.get("out_kb_refusal_rate"),
                metrics.get("overall_accuracy"),
                metrics.get("avg_time_seconds"),
            ))

    def save_eval_results(self, run_id: str, results: List[Dict]) -> None:
        sql = """
            INSERT INTO eval_results (
                run_id, question_index, category, question, correct_answer,
                agent_choice, is_correct, is_refusal, has_sources, elapsed_seconds, response
            ) VALUES %s
        """
        records = [
            (
                run_id,
                r.get("index"),
                r.get("category"),
                r.get("question"),
                r.get("correct_answer"),
                r.get("agent_choice"),
                r.get("is_correct"),
                r.get("is_refusal"),
                r.get("has_sources"),
                r.get("elapsed_seconds"),
                r.get("response"),
            )
            for r in results
        ]
        with self._cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, records)
