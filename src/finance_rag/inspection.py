"""Inspect persisted records without executing the notebook or opening an index."""
import sqlite3
from collections import Counter
from pathlib import Path

from .settings import ROOT, read_json


def inspect_snapshot():
    from pypdf import PdfReader

    path = ROOT / "data/chroma/snapshot/chroma.sqlite3"
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        collection = connection.execute("SELECT id, name, dimension FROM collections").fetchall()
        if len(collection) != 1:
            raise ValueError("Expected one original collection")
        segment_id = connection.execute(
            "SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA'", (collection[0][0],)
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT e.id, m.key, m.string_value, m.int_value FROM embeddings e "
            "JOIN embedding_metadata m ON e.id = m.id WHERE e.segment_id = ?", (segment_id,)
        ).fetchall()
        records = {}
        for record_id, key, text, number in rows:
            records.setdefault(record_id, {})[key] = text if text is not None else number
        publishers = Counter(record.get("publisher") for record in records.values())
        issues = []
        pdfs = []
        page_counts = {}
        labels = {}
        for doc in read_json("documents.json"):
            reader = PdfReader(ROOT / doc["path"])
            count = len(reader.pages)
            page_counts[Path(doc["path"]).name] = count
            pdfs.append({"id": doc["id"], "publisher": doc["publisher"], "pages": count,
                         "forecast_year": doc["forecast_year"], "publication_date": doc["publication_date"]})
            if count != doc["expected_pages"]:
                issues.append(f"Unexpected page count for {doc['id']}")
        for record_id, record in records.items():
            source = record.get("source")
            page = record.get("page")
            if not record.get("chroma:document", "").strip():
                issues.append(f"Empty text in record {record_id}")
            if record.get("publisher") not in {d["publisher"] for d in read_json("documents.json")}:
                issues.append(f"Unknown publisher in record {record_id}")
            if source not in page_counts or not isinstance(page, int) or not 0 <= page < page_counts[source]:
                issues.append(f"Invalid source/page in record {record_id}")
            if record.get("page_label") is not None:
                labels[source] = True
        unique_pages = {source: len({r["page"] for r in records.values() if r.get("source") == source})
                        for source in page_counts}
        return {"collection": collection[0][1], "dimensions": collection[0][2],
                "chunks": len(records), "chunks_by_publisher": dict(publishers),
                "pdfs": pdfs, "indexed_pages_by_source": unique_pages,
                "page_labels_available": labels, "issues": issues,
                "limitation": "Structural check only; it does not establish answer correctness or embedding provenance."}
    finally:
        connection.close()
