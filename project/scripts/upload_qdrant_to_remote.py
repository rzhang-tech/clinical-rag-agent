"""
One-time migration: upload local Qdrant data (file-based) to a remote Qdrant server.
Run this from your local machine AFTER the remote Qdrant server is up.

Usage:
    python scripts/upload_qdrant_to_remote.py --remote http://<EC2-IP>:6333
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import config
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

COLLECTION = config.CHILD_COLLECTION
BATCH_SIZE = 100


def migrate(remote_url: str):
    print(f"Source : local path  → {config.QDRANT_DB_PATH}")
    print(f"Target : remote server → {remote_url}")
    print()

    local = QdrantClient(path=config.QDRANT_DB_PATH)
    remote = QdrantClient(url=remote_url, timeout=60)

    # -- Fetch collection config from local --
    info = local.get_collection(COLLECTION)
    vec_size = info.config.params.vectors.size
    distance = info.config.params.vectors.distance

    # -- Recreate collection on remote --
    if remote.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' already exists on remote — deleting and recreating...")
        remote.delete_collection(COLLECTION)

    remote.create_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(size=vec_size, distance=distance),
        sparse_vectors_config={
            config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()
        },
    )
    print(f"Collection created on remote (dim={vec_size})\n")

    # -- Scroll and upload in batches --
    total = local.count(COLLECTION).count
    print(f"Uploading {total} points in batches of {BATCH_SIZE}...")

    offset = None
    uploaded = 0

    while True:
        records, offset = local.scroll(
            collection_name=COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        if not records:
            break

        points = []
        for r in records:
            vectors = {}
            if isinstance(r.vector, dict):
                # named vectors (dense + sparse)
                if "" in r.vector:
                    vectors[""] = r.vector[""]
                if config.SPARSE_VECTOR_NAME in r.vector:
                    vectors[config.SPARSE_VECTOR_NAME] = r.vector[config.SPARSE_VECTOR_NAME]
            else:
                vectors = r.vector

            points.append(qmodels.PointStruct(
                id=r.id,
                vector=vectors,
                payload=r.payload,
            ))

        remote.upsert(collection_name=COLLECTION, points=points)
        uploaded += len(records)
        print(f"  {uploaded}/{total} ({uploaded / total * 100:.1f}%)", end="\r")

        if offset is None:
            break

    print(f"\nDone — {uploaded} points uploaded to {remote_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True, help="Remote Qdrant URL, e.g. http://<EC2-IP>:6333")
    args = parser.parse_args()
    migrate(args.remote)
