"""Fill the notebook's missing indexing step, without replacing its RAG pipeline."""
import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .retrieval import embedding_model
from .settings import ROOT, collection_name, db_path, read_json, settings


def digest(value):
    return hashlib.sha256(value).hexdigest()


def make_chunk_id(doc_id, file_hash, pipeline_hash, page, position, text):
    value = json.dumps([doc_id, file_hash, pipeline_hash, page, position, text], ensure_ascii=False)
    return digest(value.encode("utf-8"))


def pipeline_configuration():
    config = settings()
    pipeline = {key: config[key] for key in ["embedding_model", "embedding_revision", "normalize_embeddings",
                                            "chunk_size", "chunk_overlap"]}
    import importlib.metadata
    pipeline["pypdf_version"] = importlib.metadata.version("pypdf")
    pipeline["fonttools_version"] = importlib.metadata.version("fonttools")
    pipeline["splitter_version"] = importlib.metadata.version("langchain-text-splitters")
    pipeline_hash = digest(json.dumps(pipeline, sort_keys=True).encode())
    return pipeline, pipeline_hash


def prepare_chunks(source="local"):
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    config = settings()
    pipeline, pipeline_hash = pipeline_configuration()
    splitter = RecursiveCharacterTextSplitter(chunk_size=config["chunk_size"],
                                             chunk_overlap=config["chunk_overlap"])
    documents = read_json("documents.json")
    if len({doc["id"] for doc in documents}) != len(documents):
        raise ValueError("Document IDs must be unique")
    chunks = []
    if source not in {"local", "s3"}:
        raise ValueError("Document source must be local or s3")
    for doc in documents:
        if source == "s3":
            from .storage import cached_document_path
            path = cached_document_path(doc)
        else:
            path = ROOT / doc["path"]
        file_hash = digest(path.read_bytes())
        if file_hash != doc.get("expected_sha256"):
            raise ValueError(f"PDF differs from the approved manifest: {path.name}")
        reader = PdfReader(path)
        labels = reader.page_labels
        for page_index, page in enumerate(reader.pages):
            for position, text in enumerate(splitter.split_text(page.extract_text() or "")):
                metadata = {"doc_id": doc["id"], "source": path.name, "publisher": doc["publisher"],
                            "document_type": doc["document_type"], "year": doc["forecast_year"],
                            "page": page_index, "page_label": str(labels[page_index]),
                            "pdf_page": page_index + 1, "file_sha256": file_hash, "pipeline_sha256": pipeline_hash}
                if doc["publication_date"]:
                    metadata["publication_date"] = doc["publication_date"]
                source_signature = file_hash + ":" + digest(json.dumps(metadata, sort_keys=True).encode())
                chunks.append({"id": make_chunk_id(doc["id"], source_signature, pipeline_hash, page_index, position, text),
                               "text": text, "metadata": metadata})
    counts = Counter(chunk["metadata"]["doc_id"] for chunk in chunks)
    if any(counts[doc["id"]] == 0 for doc in documents):
        raise ValueError("A manifest PDF produced no chunks; the existing index was retained")
    return chunks, pipeline, pipeline_hash


@contextmanager
def exclusive_index():
    from filelock import FileLock, Timeout
    folder = db_path("rebuilt").parent
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(folder / "ingestion.lock"), timeout=0):
            yield
    except Timeout:
        raise RuntimeError("Another process is indexing this project; try after it finishes") from None


def ingest(source="local"):
    with exclusive_index():
        return _ingest(source)


def _ingest(source="local"):
    """Internal ingestion entry point; caller must hold exclusive_index()."""
    import chromadb
    from chromadb.config import Settings

    chunks, pipeline, pipeline_hash = prepare_chunks(source=source)
    path = db_path("rebuilt")
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(collection_name("rebuilt"), embedding_function=None,
                                                  metadata={"pipeline_sha256": pipeline_hash,
                                                            "pipeline": json.dumps(pipeline, sort_keys=True)})
    if collection.metadata.get("pipeline_sha256") != pipeline_hash:
        if collection.count() == 0:
            collection.modify(metadata={"pipeline_sha256": pipeline_hash,
                                        "pipeline": json.dumps(pipeline, sort_keys=True)})
        else:
            raise ValueError("Pipeline configuration changed. Choose a new rebuilt_collection in config.json.")
    existing = set(collection.get(include=[])["ids"])
    incoming = {c["id"] for c in chunks}
    pending = [c for c in chunks if c["id"] not in existing]
    if pending:
        model = embedding_model()
        # Encode before changing stored records; original baseline is never touched.
        vectors = model.encode([c["text"] for c in pending], batch_size=32, show_progress_bar=True,
                               normalize_embeddings=pipeline["normalize_embeddings"])
        for start in range(0, len(pending), 100):
            batch = pending[start:start + 100]
            collection.upsert(ids=[c["id"] for c in batch], documents=[c["text"] for c in batch],
                              metadatas=[c["metadata"] for c in batch], embeddings=vectors[start:start + 100].tolist())
    stale = sorted(existing - incoming)
    for start in range(0, len(stale), 100):
        collection.delete(ids=stale[start:start + 100])
    actual = collection.get(include=["documents", "metadatas"])
    expected = {chunk["id"]: chunk for chunk in chunks}
    if set(actual["ids"]) != incoming or any(
            text != expected[key]["text"] or metadata != expected[key]["metadata"]
            for key, text, metadata in zip(actual["ids"], actual["documents"], actual["metadatas"])):
        raise RuntimeError("Stored Chroma chunks do not match the prepared documents; indexing was not verified")
    return {"collection": collection.name, "source": source, "chunks": collection.count(),
            "new_chunks": len(incoming - existing), "removed_chunks": len(stale),
            "pipeline_sha256": pipeline_hash, "pipeline": pipeline,
            "documents": [{"doc_id": doc["id"], "source": Path(doc["path"]).name,
                           "sha256": doc["expected_sha256"],
                           "chunks": sum(chunk["metadata"]["doc_id"] == doc["id"] for chunk in chunks)}
                          for doc in read_json("documents.json")]}
