"""Local answer diagnostics, not a semantic truth or entailment judge."""
import re
from decimal import Decimal

from .evaluation import normalize


REFERENCE = re.compile(r"\[chunk:([^\]\s]+)\]")
CITATION = re.compile(r"\(([^()]+?),\s*(\d{4}),\s*page\s+([^()]+?)\)\s*\[chunk:([^\]\s]+)\]", re.I)


def is_abstention(answer):
    # Exact response only: appending a fabricated claim must not bypass checks.
    return normalize(answer).rstrip(".") == "not found in retrieved documents"


def percentages(text):
    text = text.replace("−", "-")
    return {str(Decimal(value).normalize()) for value in
            re.findall(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:%|percent\b)", text, re.I)}


def check_answer(answer, evidence):
    """Validate citation identity; flag numeric/format problems for human review.

    Matching a number anywhere in cited text does NOT establish the right subject,
    year, basis or unit. Do not report these diagnostics as answer accuracy.
    """
    records = {}
    errors, warnings = [], []
    for item in evidence:
        key = item["chunk_id"]
        if key in records and records[key] != item:
            errors.append({"code": "conflicting_evidence_id", "chunk_id": key})
        records[key] = item
    if not answer.strip():
        errors.append({"code": "empty_answer"})
    abstained = is_abstention(answer)
    matches = list(CITATION.finditer(answer))
    paired_spans = {match.span(4) for match in matches}
    citations = []
    for ref in REFERENCE.finditer(answer):
        if ref.span(1) not in paired_spans:
            errors.append({"code": "unpaired_chunk_reference", "chunk_id": ref.group(1)})
    for match in matches:
        publisher, year, page, key = match.groups()
        item = records.get(key)
        problems = []
        if item is None:
            problems.append("unknown_chunk_id")
        else:
            if normalize(publisher) != normalize(item["publisher"]):
                problems.append("publisher_mismatch")
            if int(year) != item["year"]:
                problems.append("year_mismatch")
            if normalize(page) != normalize(str(item["page_label"])):
                problems.append("page_mismatch")
        citations.append({"chunk_id": key, "publisher": publisher, "year": int(year),
                          "page_label": page, "metadata_valid": not problems})
        errors.extend({"code": problem, "chunk_id": key} for problem in problems)
    if not abstained and not matches:
        errors.append({"code": "no_paired_citations"})
    cited_text = "\n".join((records[c["chunk_id"]].get("context_text") or records[c["chunk_id"]]["text"])
                           for c in citations if c["metadata_valid"])
    missing = sorted(percentages(answer) - percentages(cited_text))
    if missing:
        warnings.append({"code": "percentage_not_in_cited_text", "values": missing})
    if "<thinking>" in answer or "</thinking>" in answer:
        warnings.append({"code": "unwanted_output_wrapper"})
    return {"abstained": abstained, "citation_count": len(citations), "citations": citations,
            "citation_metadata_passed": not errors, "errors": errors, "warnings": warnings,
            "automated_checks_passed": not errors and not warnings,
            "semantic_review_required": True,
            "limitation": "Metadata integrity and percentage presence only; claim support, completeness, subject/year/basis and uncited claims still require review."}


def fact_pattern_checks(answer, facts):
    """Draft, case-specific text checks. These are not general factual accuracy."""
    return [{"label": fact["label"],
             "matched": any(re.search(pattern, normalize(answer), re.I) for pattern in fact["patterns"])}
            for fact in facts]


def verify_saved_evidence(evidence, collection):
    """Do not trust a saved report's source text without checking the stored index."""
    if not evidence:
        return
    from .retrieval import Evidence, PageContext
    context = PageContext()
    data = collection.get(ids=list({item["chunk_id"] for item in evidence}), include=["documents", "metadatas"])
    records = {key: (text, meta) for key, text, meta in zip(data["ids"], data["documents"], data["metadatas"])}
    for item in evidence:
        if item["chunk_id"] not in records:
            raise ValueError("Saved evidence ID is missing from the selected index")
        text, meta = records[item["chunk_id"]]
        expected = {"text": text, "publisher": meta["publisher"], "year": meta["year"],
                    "source": meta["source"], "page_index": meta["page"], "page_label": str(meta["page_label"])}
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved evidence text or provenance differs from the selected index")
        if item.get("context_text") is not None:
            hit = Evidence(item["chunk_id"], **expected, distance=item.get("distance"))
            if context.expand(hit).context_text != item["context_text"]:
                raise ValueError("Saved context differs from the verified PDF page window")


def audit_saved_answers():
    """Replay two explicitly selected historical smoke answers; no inference calls."""
    import hashlib
    import json
    from .settings import ROOT
    from .retrieval import open_collection
    path = ROOT / "evaluation/answer-cases.json"
    specification = json.loads(path.read_text(encoding="utf-8"))
    collection = open_collection("rebuilt")
    results = []
    for case in specification["cases"]:
        report_path = (ROOT / case["report"]).resolve()
        if not report_path.is_relative_to(ROOT):
            raise ValueError("Saved answer report must be inside the project")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["question"] != case["question"]:
            raise ValueError("Saved question does not match the declared answer case")
        verify_saved_evidence(report["evidence"], collection)
        checks = check_answer(report["answer"], report["evidence"])
        facts = fact_pattern_checks(report["answer"], case["facts"])
        results.append({"id": case["id"], "question": report["question"], "answer": report["answer"],
                        "report": case["report"], "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        "evidence_verified_against_index": True, "checks": checks, "fact_patterns": facts})
    return {"mode": "saved_answer_replay", "cases": len(results), "results": results,
            "case_spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "citation_metadata_passed": sum(r["checks"]["citation_metadata_passed"] for r in results),
            "automated_checks_passed": sum(r["checks"]["automated_checks_passed"] for r in results),
            "aws_calls": 0, "new_inference_cost_usd": 0,
            "reserved_test_queries_run": 0,
            "limitations": ["Two historical smoke answers, not a new run of the improved retrieval system.",
                            "Assistant-authored draft fact patterns; independent human review pending.",
                            "Regex concept/number presence is not factual accuracy or semantic citation support.",
                            "Abstention was tested with mocked model responses, not live Bedrock."]}
