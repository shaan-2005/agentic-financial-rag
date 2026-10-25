"""Local passage-level retrieval evaluation. Never calls a generator or AWS."""
import hashlib
import json
import re
import statistics
import time
import unicodedata
from dataclasses import replace
from pathlib import Path

from .settings import ROOT, read_json, settings


def normalize(text):
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    # PDF line breaks and typography should not invalidate otherwise identical evidence.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_benchmark():
    path = ROOT / "evaluation/benchmark.json"
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    questions = benchmark["questions"]
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Benchmark question IDs must be unique")
    for question in questions:
        if question["answerable"] != bool(question["required_evidence"]):
            raise ValueError("Answerable cases must have evidence; abstention cases must not")
    return benchmark, hashlib.sha256(path.read_bytes()).hexdigest()


def matches_evidence(hit, evidence):
    return (hit.source == evidence["source"]
            and normalize(evidence["excerpt"]) in normalize(hit.text))


def score_hits(hits, required):
    if not required:
        return None
    ranks = [next((rank for rank, hit in enumerate(hits, 1) if matches_evidence(hit, evidence)), None)
             for evidence in required]
    found = sum(rank is not None for rank in ranks)
    earliest = min((rank for rank in ranks if rank is not None), default=None)
    return {"evidence_recall": found / len(required), "all_evidence_found": found == len(required),
            "reciprocal_rank": 1 / earliest if earliest else 0, "required_evidence_ranks": ranks}


def audit_benchmark(collection):
    """Check annotations against original PDF pages and current indexed passages."""
    from pypdf import PdfReader
    benchmark, checksum = load_benchmark()
    documents = read_json("documents.json")
    readers = {Path(doc["path"]).name: PdfReader(ROOT / doc["path"]) for doc in documents}
    for doc in documents:
        actual = hashlib.sha256((ROOT / doc["path"]).read_bytes()).hexdigest()
        if benchmark["document_sha256"][Path(doc["path"]).name] != actual:
            raise ValueError("Corpus has changed; recheck benchmark evidence before scoring")
    data = collection.get(include=["documents", "metadatas"])
    evidence_checks = []
    for question in benchmark["questions"]:
        for evidence in question["required_evidence"]:
            source = evidence["source"]
            index = evidence["pdf_page"] - 1
            reader = readers[source]
            in_page = normalize(evidence["excerpt"]) in normalize(reader.pages[index].extract_text() or "")
            chunks = [chunk_id for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
                      if meta["source"] == source and normalize(evidence["excerpt"]) in normalize(text)]
            if not in_page or not chunks:
                raise ValueError(f"Invalid or unavailable gold evidence for {question['id']}: {source}")
            evidence_checks.append({"question_id": question["id"], "source": source,
                                    "pdf_page": evidence["pdf_page"], "page_label": reader.page_labels[index],
                                    "matching_chunk_ids": chunks})
    return {"benchmark_sha256": checksum, "question_count": len(benchmark["questions"]),
            "evidence_annotations": len(evidence_checks), "passed": True,
            "checks": evidence_checks,
            "limitation": "Source consistency check only. Independent human review of questions and labels is pending."}


def aggregate(results, score_key="score"):
    scored = [r for r in results if r[score_key] is not None]
    if not scored:
        return {"scored_questions": 0}
    return {"scored_questions": len(scored),
            "mean_evidence_recall_at_k": statistics.mean(r[score_key]["evidence_recall"] for r in scored),
            "all_evidence_success_rate_at_k": statistics.mean(r[score_key]["all_evidence_found"] for r in scored),
            "mrr_at_k": statistics.mean(r[score_key]["reciprocal_rank"] for r in scored)}


def evaluate(index="rebuilt", split="dev", filter_mode="none", k=6, strategy="dense",
             query_mode="original", context_mode="none"):
    from .retrieval import Retriever
    if split not in {"dev", "test"} or filter_mode not in {"none", "publisher"}:
        raise ValueError("Invalid evaluation split or filter mode")
    if not 1 <= k <= 100:
        raise ValueError("Evaluation k must be between 1 and 100")
    benchmark, checksum = load_benchmark()
    retriever = Retriever(index, strategy=strategy, query_mode=query_mode, context_mode=context_mode)
    audit_benchmark(retriever.collection)
    # Build/download outside query timing. Reported latency includes ranking, not startup.
    if strategy != "dense":
        _ = retriever.keywords
    if strategy == "hybrid-rerank":
        _ = retriever.reranker
    results = []
    for question in benchmark["questions"]:
        if question["split"] != split:
            continue
        publishers = {e["publisher"] for e in question["required_evidence"]}
        publisher = next(iter(publishers)) if filter_mode == "publisher" and len(publishers) == 1 else None
        started = time.perf_counter()
        hits = retriever.search(question["question"], publisher=publisher, k=k)
        latency = time.perf_counter() - started
        score = score_hits(hits, question["required_evidence"])
        expanded = [replace(hit, text=hit.context_text or hit.text) for hit in hits]
        context_score = score_hits(expanded, question["required_evidence"])
        results.append({"id": question["id"], "category": question["category"], "question": question["question"],
                        "publisher_filter": publisher, "answerable": question["answerable"], "score": score,
                        "context_score": context_score,
                        "returned_text_characters": sum(len(h.text) + len(h.context_text or "") for h in hits),
                        "latency_seconds": round(latency, 4),
                        "retrieved": [{"chunk_id": h.chunk_id, "source": h.source, "page_label": h.page_label,
                                       "distance": h.distance, "retrieval_score": h.retrieval_score,
                                       "rerank_score": h.rerank_score, "text": h.text,
                                       "context_text": h.context_text, "context_status": h.context_status} for h in hits],
                        "expected_answer": question["expected_answer"],
                        "required_evidence": question["required_evidence"]})
    return {"benchmark_sha256": checksum, "index": index, "collection": retriever.collection.name,
            "collection_count": retriever.collection.count(), "split": split, "filter_mode": filter_mode,
            "k": k, "strategy": strategy, "query_mode": query_mode, "context_mode": context_mode,
            "configuration": settings(), "metrics": aggregate(results),
            "context_metrics": aggregate(results, "context_score"),
            "by_category": {category: aggregate([r for r in results if r["category"] == category])
                            for category in sorted({r["category"] for r in results})},
            "context_by_category": {category: aggregate([r for r in results if r["category"] == category], "context_score")
                                    for category in sorted({r["category"] for r in results})},
            "median_query_seconds": statistics.median(r["latency_seconds"] for r in results),
            "results": results, "aws_calls": 0,
            "limitations": ["Assistant-authored draft benchmark; independent human review pending.",
                            "Context metrics include extra same-page text; compare separately from original chunk-only scores.",
                            "Passage matching uses source filename and normalized excerpt, allowing equivalent text on other pages.",
                            "Gold evidence is incomplete; valid paraphrases/alternative passages may be marked missing.",
                            "Source pages and facts are concentrated; this is not a representative corpus-wide score.",
                            "Recall is averaged per question; multi-source cases require every evidence item for full success.",
                            "MRR measures the first required evidence item, not completion of multi-source answers.",
                            "Abstention cases are excluded from retrieval metrics; generation abstention is not tested.",
                            "Latency is warm-query timing, excluding model startup and generation.",
                            "This directly tests question retrieval, not the agent's rewritten tool queries or generated answers."]}
