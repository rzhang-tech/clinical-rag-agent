"""
Import MedRAG textbooks dataset into the Clinical RAG Agent.

Loads pre-chunked medical textbook snippets from HuggingFace (MedRAG/textbooks),
groups them into parent chunks (2000-4000 chars), splits into child chunks (500 chars),
and stores them in Qdrant + parent store.

Usage:
    # Dry run — see stats without writing anything
    python project/scripts/import_medrag.py --dry-run

    # Import a single textbook
    python project/scripts/import_medrag.py --titles Anatomy_Gray

    # Import multiple textbooks
    python project/scripts/import_medrag.py --titles Anatomy_Gray InternalMed_Harrison

    # Import all 18 textbooks
    python project/scripts/import_medrag.py

    # List available textbook titles
    python project/scripts/import_medrag.py --list-titles

NOTE: Do NOT run this while the Gradio app (app.py) is running — they share the
same Qdrant database and will conflict on file locks.
"""

import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

# Allow imports from the project directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import MedRAG textbooks into Clinical RAG Agent"
    )
    parser.add_argument(
        "--titles", nargs="+", default=None,
        help="Textbook titles to import (e.g. Anatomy_Gray). Omit to import all."
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Number of child chunks per Qdrant insertion batch (default: 500)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics without writing to the database"
    )
    parser.add_argument(
        "--list-titles", action="store_true",
        help="List available textbook titles and exit"
    )
    return parser.parse_args()


def load_medrag_dataset(titles=None):
    """Load MedRAG textbooks from HuggingFace and group by title."""
    from datasets import load_dataset

    print("Loading MedRAG/textbooks from HuggingFace...")
    ds = load_dataset("MedRAG/textbooks", split="train")
    print(f"  Loaded {len(ds)} chunks total")

    if titles:
        title_set = set(titles)
        ds = ds.filter(lambda row: row["title"] in title_set)
        found = set(row["title"] for row in ds)
        missing = title_set - found
        if missing:
            print(f"  WARNING: titles not found in dataset: {missing}")
        print(f"  Filtered to {len(ds)} chunks from {len(found)} textbook(s)")

    # Group by textbook and sort by chunk index
    books = defaultdict(list)
    for row in ds:
        books[row["title"]].append(row)

    for title in books:
        books[title].sort(key=lambda r: int(r["id"].rsplit("_", 1)[-1]))

    return books


def group_into_parents(chunks, title):
    """Group consecutive MedRAG chunks into parent-sized blocks (2000-4000 chars)."""
    parents = []
    current_text = ""
    current_ids = []

    def finalize_parent():
        parent_id = f"medrag_{title}_parent_{len(parents)}"
        parents.append((parent_id, Document(
            page_content=current_text,
            metadata={
                "source": f"MedRAG/{title}",
                "parent_id": parent_id,
                "dataset": "MedRAG/textbooks",
            }
        )))

    for chunk in chunks:
        candidate = (current_text + "\n\n" + chunk["content"]).strip() if current_text else chunk["content"]

        # If adding this chunk would exceed max and we already have content, finalize first
        if len(candidate) > config.MAX_PARENT_SIZE and current_text:
            finalize_parent()
            current_text = chunk["content"]
            current_ids = [chunk["id"]]
        else:
            current_text = candidate
            current_ids.append(chunk["id"])

        # If we've reached min size, finalize
        if len(current_text) >= config.MIN_PARENT_SIZE:
            finalize_parent()
            current_text = ""
            current_ids = []

    # Handle remainder
    if current_text:
        if parents and len(current_text) < config.MIN_PARENT_SIZE:
            # Merge into last parent
            last_pid, last_doc = parents[-1]
            last_doc.page_content += "\n\n" + current_text
        else:
            finalize_parent()

    return parents


def split_into_children(parent_tuples):
    """Split parent chunks into child chunks using the same config as the main system."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
    )
    all_children = []
    for _parent_id, parent_doc in parent_tuples:
        children = splitter.split_documents([parent_doc])
        all_children.extend(children)
    return all_children


def main():
    args = parse_args()

    # Load dataset
    books = load_medrag_dataset(args.titles)

    if args.list_titles:
        print(f"\nAvailable textbooks ({len(books)}):")
        for title in sorted(books.keys()):
            print(f"  - {title} ({len(books[title])} chunks)")
        return

    if not books:
        print("No textbooks to import.")
        return

    print(f"\nWill process {len(books)} textbook(s)")
    if not args.dry_run:
        print("WARNING: This will write to Qdrant and parent store.")
        print("         Do NOT run while the Gradio app is running.\n")

    # Initialize storage (unless dry run)
    collection = None
    parent_store = None
    if not args.dry_run:
        from db.vector_db_manager import VectorDbManager
        from db.parent_store_manager import ParentStoreManager

        print("Initializing vector database and parent store...")
        vector_db = VectorDbManager()
        vector_db.create_collection(config.CHILD_COLLECTION)
        collection = vector_db.get_collection(config.CHILD_COLLECTION)
        parent_store = ParentStoreManager()
        print()

    # Process each textbook
    from tqdm import tqdm

    total_stats = {"chunks": 0, "parents": 0, "children": 0}
    start_time = time.time()

    for title in tqdm(sorted(books.keys()), desc="Textbooks", unit="book"):
        chunks = books[title]
        total_stats["chunks"] += len(chunks)

        # Group into parents
        parent_tuples = group_into_parents(chunks, title)
        total_stats["parents"] += len(parent_tuples)

        # Split into children
        children = split_into_children(parent_tuples)
        total_stats["children"] += len(children)

        if args.dry_run:
            continue

        # Save parents
        parent_store.save_many(parent_tuples)

        # Insert children in batches
        for i in range(0, len(children), args.batch_size):
            batch = children[i : i + args.batch_size]
            collection.add_documents(batch)

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Import complete in {elapsed:.1f}s")
    print(f"  Textbooks processed: {len(books)}")
    print(f"  MedRAG chunks consumed: {total_stats['chunks']}")
    print(f"  Parent chunks created: {total_stats['parents']}")
    print(f"  Child chunks created: {total_stats['children']}")
    if not args.dry_run:
        print(f"\n  Parents stored in: {config.PARENT_STORE_PATH}/")
        print(f"  Children indexed in Qdrant collection: {config.CHILD_COLLECTION}")


if __name__ == "__main__":
    main()
