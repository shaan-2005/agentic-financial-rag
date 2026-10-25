"""Reuse MiniLM + Chroma; retain dense baseline and add local hybrid retrieval."""
from dataclasses import asdict, dataclass, replace
import hashlib
import os
import math
import re
from collections import Counter
from functools import cached_property
from pathlib import Path
from io import BytesIO

from .settings import ROOT, collection_name, db_path, read_json, settings


@dataclass
class Evidence:
    chunk_id: str
    text: str
    publisher: str
    year: int
    source: str
    page_index: int
    page_label: str
    distance: float | None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    context_text: str | None = None
    context_status: str | None = None

    def as_dict(self):
        return asdict(self)


def open_collection(index="baseline"):
    import chromadb
    from chromadb.config import Settings

    path = db_path(index)
    if not (path / "chroma.sqlite3").exists():
        raise FileNotFoundError(f"Missing index at {path}. Run ingest for a rebuilt index.")
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    return client.get_collection(collection_name(index), embedding_function=None)


def embedding_model():
    # Keep model caches inside this project, independent of host-wide HF settings.
    os.environ["HF_HOME"] = str(ROOT / "data/models/hf-home")
    os.environ["HF_XET_CACHE"] = str(ROOT / "data/models/xet")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer
    config = settings()
    if not config.get("embedding_revision"):
        raise ValueError("Pin embedding_revision in config.json before embedding documents or queries.")
    torch.set_num_threads(4)
    snapshot = ROOT / "data/models" / ("models--" + config["embedding_model"].replace("/", "--")) / "snapshots" / config["embedding_revision"]
    return SentenceTransformer(config["embedding_model"], revision=config["embedding_revision"],
                               device="cpu", trust_remote_code=False, token=False,
                               local_files_only=(snapshot / "model.safetensors").exists(),
                               cache_folder=str(ROOT / "data/models"))


PUBLISHER_PATTERNS = {
    "Goldman Sachs": r"\bgoldman(?:\s+sachs)?\b",
    "JP Morgan": r"\bj\.?\s*p\.?\s*morgan\b|\bjpmorgan\b",
    "IMF": r"\bimf\b|\binternational monetary fund\b",
}


def named_publishers(question):
    """Route only from explicit names in the user's query, never benchmark labels."""
    return [name for name, pattern in PUBLISHER_PATTERNS.items() if re.search(pattern, question, re.I)]


def content_query(question):
    for pattern in PUBLISHER_PATTERNS.values():
        question = re.sub(pattern, " ", question, flags=re.I)
    return question.strip() or "outlook"


def focused_query(question):
    """Remove comparison boilerplate, retaining topic, years, numbers and basis.

    Used only for explicitly multi-publisher queries. No model, benchmark labels
    or guessed answer values are involved. The original question goes to the LLM.
    """
    noise = {"compare", "compared", "comparison", "differ", "difference", "between", "reports", "report",
             "outlook", "outlooks", "account", "publication", "timing", "january", "february", "march",
             "april", "may", "june", "july", "august", "september", "october", "november", "december"}
    terms = [term for term in keyword_tokens(content_query(question)) if term not in noise]
    return " ".join(terms) or content_query(question)


def page_window(page_text, anchor, before=400, after=800, max_chars=2200):
    """Expand around an exact anchor, within one page and a fixed character cap."""
    position = page_text.find(anchor)
    if position < 0 or len(anchor) > max_chars:
        return None
    start = max(0, position - before)
    end = min(len(page_text), position + len(anchor) + after)
    if end - start > max_chars:
        start = max(start, position + len(anchor) - max_chars)
        end = min(end, start + max_chars)
    return page_text[start:end]


class PageContext:
    """Read surrounding text only from the verified local PDF, with page provenance."""
    def __init__(self, root=ROOT, documents=None):
        self.root = root
        self.documents = {Path(doc["path"]).name: doc for doc in
                          (read_json("documents.json") if documents is None else documents)}
        self.readers = {}
        self.pages = {}

    def expand(self, hit):
        doc = self.documents.get(hit.source)
        if doc is None:
            return replace(hit, context_status="source_not_in_manifest")
        if hit.source not in self.readers:
            path = self.root / doc["path"]
            if not path.exists():
                return replace(hit, context_status="local_pdf_missing")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != doc["expected_sha256"]:
                raise ValueError(f"PDF checksum mismatch for context: {hit.source}")
            from pypdf import PdfReader
            self.readers[hit.source] = PdfReader(BytesIO(payload))
        reader = self.readers[hit.source]
        if not 0 <= hit.page_index < len(reader.pages):
            return replace(hit, context_status="invalid_page")
        if str(reader.page_labels[hit.page_index]) != hit.page_label:
            return replace(hit, context_status="page_label_mismatch")
        key = (hit.source, hit.page_index)
        if key not in self.pages:
            self.pages[key] = reader.pages[hit.page_index].extract_text() or ""
        context = page_window(self.pages[key], hit.text)
        return replace(hit, context_text=context,
                       context_status="same_page_verified" if context else "anchor_not_found")


def keyword_tokens(text):
    text = text.casefold().replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    for pattern, replacement in [(r"\bunited states\b|\bu\.s\.\b", "us"),
                                 (r"\beuro area\b", "eurozone"),
                                 (r"\bjapanese\b", "japan"),
                                 (r"\bemerging[- ]markets?\b", "emerging market")]:
        text = re.sub(pattern, replacement, text)
    stop = {"the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "is", "are", "does",
            "do", "what", "how", "its", "these", "with", "from", "by", "as", "it", "that", "this"}
    return [word for word in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if word not in stop]


class KeywordIndex:
    """Small in-memory BM25 index over the same Chroma chunks (k1=1.5, b=.75)."""
    def __init__(self, records):
        self.records = records
        self.counts = [Counter(keyword_tokens(record.text)) for record in records]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average_length = sum(self.lengths) / max(len(records), 1) or 1
        self.df = Counter(term for count in self.counts for term in count)

    def search(self, question, publisher, k):
        scores = []
        for record, terms, length in zip(self.records, self.counts, self.lengths):
            if publisher and record.publisher != publisher:
                continue
            score = 0.0
            for term in set(keyword_tokens(question)):
                freq = terms[term]
                if freq:
                    idf = math.log(1 + (len(self.records) - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / self.average_length))
            if score > 0:
                scores.append((score, record))
        return [record for _, record in sorted(scores, key=lambda pair: (-pair[0], pair[1].chunk_id))[:k]]


def fuse_rankings(rankings, constant=60):
    scores, records = {}, {}
    for ranking in rankings:
        for rank, record in enumerate(ranking, 1):
            records[record.chunk_id] = record
            scores[record.chunk_id] = scores.get(record.chunk_id, 0) + 1 / (constant + rank)
    from dataclasses import replace
    return [replace(records[key], retrieval_score=scores[key])
            for key in sorted(scores, key=lambda key: (-scores[key], key))]


class Retriever:
    def __init__(self, index="baseline", strategy=None, query_mode=None, context_mode=None):
        self.collection = open_collection(index)
        self.model = embedding_model()
        self.config = settings()
        self.strategy = strategy or ("dense" if index == "baseline" else self.config.get("retrieval_strategy", "dense"))
        if self.strategy not in {"dense", "hybrid", "hybrid-rerank"}:
            raise ValueError("Unknown retrieval strategy")
        self.query_mode = query_mode or ("original" if index == "baseline" else self.config.get("query_mode", "original"))
        self.context_mode = context_mode or ("none" if index == "baseline" else self.config.get("context_mode", "none"))
        if self.query_mode not in {"original", "focused", "multi-view"} or self.context_mode not in {"none", "page-window"}:
            raise ValueError("Unknown query or context mode")

    @cached_property
    def page_context(self):
        return PageContext()

    def with_context(self, hits):
        if self.context_mode == "none":
            return hits
        return [self.page_context.expand(hit) for hit in hits]

    @cached_property
    def keywords(self):
        data = self.collection.get(include=["documents", "metadatas"])
        records = [Evidence(key, text, meta.get("publisher", "Unknown"), meta.get("year", 0),
                            meta.get("source", "Unknown"), meta.get("page", -1),
                            str(meta.get("page_label", "Unknown")), None)
                   for key, text, meta in zip(data["ids"], data["documents"], data["metadatas"])]
        # Keyword-only hits have no dense distance. None is serialized as JSON null.
        return KeywordIndex(records)

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder
        name, revision = self.config["reranker_model"], self.config["reranker_revision"]
        snapshot = ROOT / "data/models" / ("models--" + name.replace("/", "--")) / "snapshots" / revision
        return CrossEncoder(name, revision=revision, device="cpu", trust_remote_code=False, token=False,
                            local_files_only=(snapshot / "model.safetensors").exists(),
                            cache_folder=str(ROOT / "data/models"), max_length=512)

    def search(self, question, publisher=None, k=None):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        k = self.config["top_k"] if k is None else k
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        if self.strategy == "dense":
            return self.with_context(self.dense_search(question, publisher, k))
        sources = [publisher] if publisher else (named_publishers(question) or [None])
        if k < len(sources):
            raise ValueError("k must allow at least one passage per named publisher")
        query = focused_query(question) if self.query_mode == "focused" and len(sources) > 1 else content_query(question)
        views = [query]
        if self.query_mode == "multi-view" and len(sources) > 1:
            focus = focused_query(question)
            if focus != query:
                views.append(focus)
        selected = []
        for position, source in enumerate(sources):
            rankings = []
            for view in views:
                rankings.append(self.dense_search(view, source, self.config["candidate_k"]))
                rankings.append(self.keywords.search(view, source, self.config["candidate_k"]))
            candidates = fuse_rankings(rankings, self.config["rrf_constant"])[:self.config["candidate_k"]]
            if self.strategy == "hybrid-rerank" and candidates:
                scores = self.reranker.predict([(query, h.text) for h in candidates],
                                               batch_size=16, show_progress_bar=False)
                for hit, score in zip(candidates, scores):
                    hit.rerank_score = float(score)
                candidates.sort(key=lambda h: (-h.rerank_score, h.chunk_id))
            quota = k // len(sources) + (position < k % len(sources))
            selected.append(candidates[:quota])
        # Interleave sources so both sides of a comparison appear early.
        return self.with_context([ranking[i] for i in range(k) for ranking in selected if i < len(ranking)][:k])

    def dense_search(self, question, publisher=None, k=None):
        if not question.strip():
            raise ValueError("Question cannot be empty")
        k = k or self.config["top_k"]
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        vector = self.model.encode([question], normalize_embeddings=self.config["normalize_embeddings"])[0]
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(query_embeddings=[vector.tolist()], n_results=min(k, count),
                                       where={"publisher": publisher} if publisher else None,
                                       include=["documents", "metadatas", "distances"])
        return [Evidence(chunk_id=chunk_id, text=text,
                         publisher=meta.get("publisher", "Unknown"), year=meta.get("year", 0),
                         source=meta.get("source", "Unknown"), page_index=meta.get("page", -1),
                         page_label=str(meta.get("page_label", "Unknown")), distance=float(distance))
                for chunk_id, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                                          result["metadatas"][0], result["distances"][0])]
