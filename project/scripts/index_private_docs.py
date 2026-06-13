"""
Index the synthetic documents used by the agent-behavior regression suite into
their own Qdrant collections (kept separate from the 287k MedQA index).

All content here is FICTIONAL (invented drug names, thresholds, codes) so that a
closed-book model provably cannot know it — the no-RAG baseline on these will fail,
which is exactly what makes the document-grounded cases meaningful.

Collections:
  private_v3        = RB-204 v3.2 + Bractinib monograph   (cases D1, D2, C3)
  private_conflict  = RB-204 v2.1 + v3.2                  (case D3: version conflict)

Run: python scripts/index_private_docs.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db.vector_db_manager import VectorDbManager
from db.cache_manager import CacheManager
from db.parent_store_manager import ParentStoreManager

DOCS = {
    "RB-204_v3.2": (
        "Riverbend General Hospital - Internal Protocol RB-204. "
        "Management of acute hyperkalemia in the cardiac step-down unit "
        "(version 3.2, effective 2026-03; CURRENT VERSION). "
        "Activation threshold: Protocol RB-204 is initiated when serum potassium "
        "exceeds 6.1 mmol/L on a venous sample confirmed within 20 minutes. "
        "The internal alert code for activation is 'Code Maple'. "
        "First-line agent: administer Kalexor 40 mg IV over 10 minutes; "
        "do not substitute oral resins under this protocol. "
        "Monitoring: recheck serum potassium at 90 minutes post-administration; "
        "if still above 5.8 mmol/L, escalate to the on-call nephrology fellow (pager 7731). "
        "Contraindication: RB-204 must not be used in patients flagged Halvorsen-marker "
        "positive; for these patients use Protocol RB-209 instead."
    ),
    "RB-204_v2.1": (
        "Riverbend General Hospital - Internal Protocol RB-204. "
        "Management of acute hyperkalemia (version 2.1, effective 2024-01; SUPERSEDED). "
        "Activation threshold: Protocol RB-204 is initiated when serum potassium "
        "exceeds 6.5 mmol/L. First-line agent: Kalexor 40 mg IV. "
        "Internal alert code: 'Code Maple'. "
        "Note: this version has been superseded by a newer revision."
    ),
    "Meridian_Bractinib": (
        "Meridian Health System Formulary - Monograph: Bractinib. "
        "Bractinib is a kinase inhibitor used per the Meridian formulary. "
        "Standard dose: 120 mg PO daily. "
        "Renal dose adjustment: if eGFR 30-59 mL/min, reduce to 80 mg daily; "
        "if eGFR below 30 mL/min, reduce to 60 mg every other day. "
        "Drug interactions: avoid co-administration with Velartine due to QT prolongation. "
        "Therapeutic drug monitoring: target trough concentration 45-60 ng/mL."
        # Deliberately NO pregnancy-safety section (case C3 = partial-evidence abstention).
    ),
}

COLLECTIONS = {
    "private_v3": ["RB-204_v3.2", "Meridian_Bractinib"],
    "private_conflict": ["RB-204_v2.1", "RB-204_v3.2"],
}


def main():
    cache = CacheManager()
    try:
        cache.connect()
    except Exception as exc:
        print("Redis unavailable (ok):", exc)

    vdb = VectorDbManager(cache=cache)
    pstore = ParentStoreManager()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE, chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )

    for coll, docnames in COLLECTIONS.items():
        vdb.create_collection(coll)
        collection = vdb.get_collection(coll)
        parents, children = [], []
        for dn in docnames:
            pid = f"private_{dn}_parent_0"
            pdoc = Document(page_content=DOCS[dn],
                            metadata={"source": f"{dn}.md", "parent_id": pid})
            parents.append((pid, pdoc))
            children.extend(splitter.split_documents([pdoc]))
        collection.add_documents(children)
        pstore.save_many(parents)
        print(f"[{coll}] indexed {len(children)} child chunks from {len(parents)} docs: {docnames}")

    print("Done. Verify via http://localhost:6333/collections")


if __name__ == "__main__":
    main()
