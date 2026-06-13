"""
Unattended orchestrator: wait for the 18-book Qdrant import to finish, verify the
index is complete (all 18 textbooks present), then run the full RAG eval and write
the no-RAG vs RAG ablation report.

Designed to run DETACHED (survives the terminal / Claude session closing). Logs
every step to eval_results/orchestrator.log so you can see what happened on return.

Launch (detached):
    Start-Process -FilePath <env python> -ArgumentList scripts\run_after_import.py -WindowStyle Hidden
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from urllib.request import urlopen

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
QDRANT_INFO = "http://localhost:6333/collections/document_child_chunks"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "document_child_chunks"
LOG = os.path.join(PROJ, "eval_results", "orchestrator.log")

# Tuning
STABLE_CHECKS = 5          # points unchanged for this many 60s polls => import idle
MIN_POINTS_FLOOR = 70000   # must exceed the old partial (67,647) before we trust "done"
EXPECTED_BOOKS = 18
MAX_WAIT_HOURS = 9
RAG_CONCURRENCY = "4"      # docker Qdrant server handles concurrency; modest to stay safe


def log(msg):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def points_count():
    try:
        d = json.load(urlopen(QDRANT_INFO, timeout=15))
        return d["result"]["points_count"]
    except Exception as e:
        log(f"points_count() error: {e}")
        return -1


def distinct_sources():
    try:
        from qdrant_client import QdrantClient
        c = QdrantClient(url=QDRANT_URL)
        srcs = set()
        offset = None
        while True:
            pts, offset = c.scroll(COLLECTION, limit=5000, offset=offset,
                                   with_payload=True, with_vectors=False)
            for p in pts:
                pl = p.payload or {}
                meta = pl.get("metadata") or pl
                s = meta.get("source") or pl.get("source")
                if s:
                    srcs.add(s)
            if offset is None:
                break
        return srcs
    except Exception as e:
        log(f"distinct_sources() error: {e}")
        return set()


def wait_for_import():
    prev = -1
    stable = 0
    t0 = time.time()
    while True:
        p = points_count()
        if p == prev and p > 0:
            stable += 1
        else:
            stable = 0
            prev = p
        log(f"waiting for import: points={p} stable={stable}/{STABLE_CHECKS}")

        if stable >= STABLE_CHECKS and p >= MIN_POINTS_FLOOR:
            srcs = distinct_sources()
            log(f"index appears idle. distinct books = {len(srcs)}: {sorted(srcs)}")
            if len(srcs) >= EXPECTED_BOOKS:
                log(f"VERIFIED: all {len(srcs)} books indexed, {p} points. Proceeding.")
                return True
            log(f"only {len(srcs)}/{EXPECTED_BOOKS} books — import may still be running or crashed. Waiting more.")
            stable = 0

        if time.time() - t0 > MAX_WAIT_HOURS * 3600:
            log(f"TIMEOUT after {MAX_WAIT_HOURS}h waiting for import. Aborting.")
            return False
        time.sleep(60)


def main():
    log("=" * 60)
    log("ORCHESTRATOR STARTED — will run RAG eval after import completes.")
    log("=" * 60)

    if not wait_for_import():
        log("Import not verified complete. NOT running RAG eval. Exiting.")
        sys.exit(1)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    log(f"Launching full RAG eval (concurrency={RAG_CONCURRENCY}) ...")
    rc = subprocess.run(
        [PY, os.path.join(PROJ, "scripts", "evaluate.py"),
         "--mode", "rag", "--full", "--concurrency", RAG_CONCURRENCY],
        cwd=PROJ, env=env,
    ).returncode
    log(f"RAG eval finished, rc={rc}")

    log("Building no-RAG vs RAG ablation report ...")
    subprocess.run([PY, os.path.join(PROJ, "scripts", "combine_ablation.py")], cwd=PROJ, env=env)

    log("ORCHESTRATOR DONE. See ablation_summary_*.txt in eval_results/.")


if __name__ == "__main__":
    main()
