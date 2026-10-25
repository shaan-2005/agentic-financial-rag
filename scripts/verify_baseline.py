"""Real retrieval smoke checks, not a scored RAG evaluation dataset."""
import hashlib
import json
import time

import numpy as np

from finance_rag.inspection import inspect_snapshot
from finance_rag.retrieval import Retriever
from finance_rag.settings import ROOT


def main():
    started = time.perf_counter()
    report = inspect_snapshot()
    retriever = Retriever()
    report["runtime_count"] = retriever.collection.count()
    checks = []
    for publisher in ["Goldman Sachs", "JP Morgan", "IMF"]:
        stored = retriever.collection.get(where={"publisher": publisher}, limit=1,
                                           include=["documents", "metadatas", "embeddings"])
        vector = np.array(stored["embeddings"][0])
        reproduced = retriever.model.encode(stored["documents"], normalize_embeddings=False)[0]
        cosine = float(np.dot(vector, reproduced) / (np.linalg.norm(vector) * np.linalg.norm(reproduced)))
        hits = retriever.search(stored["documents"][0], publisher=publisher, k=1)
        checks.append({"publisher": publisher, "stored_vector_cosine_with_current_model": cosine,
                       "self_retrieval_matches_text": bool(hits and hits[0].text == stored["documents"][0]),
                       "embedding_matches": cosine >= 0.9999})
    report["embedding_checks"] = checks
    questions = [
        "What does Goldman Sachs project for US economic growth in 2025?",
        "What risks to global growth does the IMF identify?",
        "What does JP Morgan say about inflation and portfolio resilience?",
    ]
    retrieval_checks = []
    for question, publisher in zip(questions, ["Goldman Sachs", "IMF", "JP Morgan"]):
        hits = retriever.search(question, publisher=publisher)
        retrieval_checks.append({"question": question, "publisher": publisher,
                                 "evidence": [hit.as_dict() for hit in hits],
                                 "filter_respected": all(hit.publisher == publisher for hit in hits),
                                 "six_results": len(hits) == 6})
    report["retrieval_checks"] = retrieval_checks
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    report["passed"] = (not report["issues"] and report["runtime_count"] == 1623
                        and all(c["embedding_matches"] and c["self_retrieval_matches_text"] for c in checks)
                        and all(c["filter_respected"] and c["six_results"] for c in retrieval_checks))
    report["limitation"] = "Smoke checks establish index usability, not retrieval accuracy, citation support, or answer quality. No model API or AWS calls were made."
    report["snapshot_sha256"] = hashlib.sha256((ROOT / "data/chroma/snapshot/chroma.sqlite3").read_bytes()).hexdigest()
    target = ROOT / "reports/baseline-verification.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "runtime_count": report["runtime_count"],
                      "embedding_checks": checks, "report": str(target)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
