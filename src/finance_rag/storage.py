"""Download the manifest PDFs only, using conditional reads and a checked cache."""
import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath

from .settings import ROOT, read_json


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_object(doc, config):
    filename = Path(doc["path"]).name
    if not filename.lower().endswith(".pdf") or filename in {"", ".", ".."}:
        raise ValueError("Manifest documents must have PDF filenames")
    prefix = config["prefix"]
    if not prefix.endswith("/") or prefix.startswith("/") or ".." in PurePosixPath(prefix).parts:
        raise ValueError("Invalid S3 document prefix")
    key = prefix + filename
    local = (ROOT / "data/s3-cache" / filename).resolve()
    if not local.is_relative_to((ROOT / "data/s3-cache").resolve()):
        raise ValueError("Document cache path escapes the project")
    return key, local


def check_documents(client, owner):
    config = read_json("aws.json")
    documents = read_json("documents.json")
    from botocore.exceptions import ClientError, BotoCoreError

    results = []
    for doc in documents:
        key, _ = document_object(doc, config)
        try:
            head = client.head_object(Bucket=config["bucket"], Key=key, ExpectedBucketOwner=owner)
        except (ClientError, BotoCoreError) as error:
            raise RuntimeError(f"Cannot read documents/{Path(doc['path']).name}. Check the upload and S3 permissions.") from error
        if not 0 < head["ContentLength"] <= config["max_document_bytes"]:
            raise ValueError(f"Document {key} is empty or exceeds the 50 MiB safety limit")
        results.append({"key": key, "bytes": head["ContentLength"],
                        "etag": head["ETag"], "encryption": head.get("ServerSideEncryption", "unknown")})
    return {"region": config["region"], "bucket": config["bucket"], "documents": results}


def sync_documents(client, owner, refresh=False):
    config = read_json("aws.json")
    documents = read_json("documents.json")
    from botocore.exceptions import ClientError, BotoCoreError

    if len({Path(doc["path"]).name for doc in documents}) != len(documents):
        raise ValueError("Each document needs a unique filename for this cache")
    results = []
    for doc in documents:
        key, local = document_object(doc, config)
        expected_hash = doc.get("expected_sha256")
        if not expected_hash:
            raise ValueError("Set expected_sha256 in the manifest before downloading")
        if not refresh and local.exists() and sha256_file(local) == expected_hash:
            results.append({"key": key, "status": "cached", "sha256": expected_hash})
            continue
        temporary = None
        try:
            head = client.head_object(Bucket=config["bucket"], Key=key, ExpectedBucketOwner=owner)
            size = head["ContentLength"]
            if not 0 < size <= config["max_document_bytes"]:
                raise ValueError(f"Document {key} is empty or too large")
            response = client.get_object(Bucket=config["bucket"], Key=key,
                                         ExpectedBucketOwner=owner, IfMatch=head["ETag"])
            local.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            total = 0
            with response["Body"] as body:
                with tempfile.NamedTemporaryFile(dir=local.parent, suffix=".part", delete=False) as target:
                    temporary = Path(target.name)
                    for block in iter(lambda: body.read(1024 * 1024), b""):
                        total += len(block)
                        if total > config["max_document_bytes"]:
                            raise ValueError("Download exceeded the configured size limit")
                        target.write(block)
                        digest.update(block)
            if total != size or digest.hexdigest() != expected_hash:
                raise ValueError(f"Downloaded {key} differs from the verified local PDF. Existing cache was retained.")
            os.replace(temporary, local)
            temporary = None
            results.append({"key": key, "status": "downloaded", "bytes": total,
                            "sha256": expected_hash, "encryption": response.get("ServerSideEncryption", "unknown")})
        except (ClientError, BotoCoreError) as error:
            raise RuntimeError(f"S3 download failed for {key}. Check login, permissions, and whether the object changed.") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"bucket": config["bucket"], "source": "s3", "documents": results}


def cached_document_path(doc):
    _, path = document_object(doc, read_json("aws.json"))
    if not path.exists() or sha256_file(path) != doc.get("expected_sha256"):
        raise ValueError("Verified S3 cache missing. Run sync-s3 first; no automatic local fallback is used.")
    return path
