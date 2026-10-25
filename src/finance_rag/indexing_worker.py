"""Manual local worker: approved S3 documents -> local Chroma -> guarded status."""
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .settings import ROOT, read_json, collection_name
from .registration import TABLE, aws_session, require_free_plan, deployment_policy

LEASE_SECONDS = 900


class IndexingRunError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(report["error"])


def prepare_index_permissions():
    from .aws_access import verify_login_profile
    account = verify_login_profile(read_json("aws.json")).split(":")[4]
    path = ROOT / "reports/registration-deploy-policy.json"
    path.write_text(json.dumps(deployment_policy(account), indent=2) + "\n", encoding="utf-8")
    return {"policy_file": "reports/registration-deploy-policy.json",
            "customer_managed_policy": "FinanceRagRegistrationDeploy",
            "change": "Add dynamodb:UpdateItem on finance-rag-documents for the local worker; the Lambda role is unchanged.",
            "aws_calls": 0}


def local_instance_id():
    """Called under the local ingestion lock; identifies this local index, not a server."""
    path = ROOT / "data/chroma/index-instance.json"
    if path.exists():
        return str(uuid.UUID(json.loads(path.read_text(encoding="utf-8"))["instance_id"]))
    identifier = str(uuid.uuid4())
    path.write_text(json.dumps({"instance_id": identifier}) + "\n", encoding="utf-8")
    return identifier


def read_snapshot(doc, table, s3, owner):
    config = read_json("aws.json")
    key = config["prefix"] + Path(doc["path"]).name
    identifier = hashlib.sha256((config["bucket"] + "\0" + key).encode()).hexdigest()
    item = table.get_item(Key={"document_id": identifier}, ConsistentRead=True).get("Item")
    if not item:
        raise RuntimeError(f"Upload registration is missing for {key}; wait for Lambda before indexing")
    head = s3.head_object(Bucket=config["bucket"], Key=key, ExpectedBucketOwner=owner)
    if (item.get("bucket_name") != config["bucket"] or item.get("object_key") != key
            or item.get("etag") != head["ETag"].strip('"')
            or int(item.get("size_bytes", -1)) != head["ContentLength"]
            or not item.get("sequencer")):
        raise RuntimeError(f"Registry and S3 differ for {key}; wait for the current upload event")
    if not 0 < head["ContentLength"] <= config["max_document_bytes"]:
        raise ValueError("Registered PDF is empty or exceeds the size limit")
    return {"doc": doc, "document_id": identifier, "key": key, "sequencer": item["sequencer"],
            "etag": item["etag"], "last_modified": head["LastModified"], "item": item}


def assert_remote_unchanged(snapshot, s3, owner):
    head = s3.head_object(Bucket=read_json("aws.json")["bucket"], Key=snapshot["key"], ExpectedBucketOwner=owner)
    if head["ETag"].strip('"') != snapshot["etag"] or head["LastModified"] != snapshot["last_modified"]:
        raise RuntimeError("S3 object changed during indexing; the run cannot mark it INDEXED")


def claim_indexing(table, snapshot, run_id, now=None):
    epoch = int(time.time()) if now is None else now
    table.update_item(Key={"document_id": snapshot["document_id"]},
        UpdateExpression="SET #status=:indexing, index_run_id=:run, index_lease_until=:until, "
                         "index_started_at=:started, content_verified=:verified REMOVE last_index_error",
        ConditionExpression="#seq=:seq AND etag=:etag AND "
                            "(#status IN (:uploaded,:failed,:indexed) OR "
                            "(#status=:indexing AND (attribute_not_exists(index_lease_until) OR index_lease_until < :now)))",
        ExpressionAttributeNames={"#status": "status", "#seq": "sequencer"},
        ExpressionAttributeValues={":indexing": "INDEXING", ":uploaded": "UPLOADED", ":failed": "FAILED",
            ":indexed": "INDEXED", ":seq": snapshot["sequencer"], ":etag": snapshot["etag"],
            ":run": run_id, ":until": epoch + LEASE_SECONDS, ":now": epoch, ":verified": False,
            ":started": datetime.fromtimestamp(epoch, timezone.utc).isoformat()})


def finish_indexing(table, snapshot, run_id, instance, pipeline_hash, chunk_count, now=None):
    if chunk_count < 1:
        raise ValueError("Cannot mark an empty document INDEXED")
    epoch = int(time.time()) if now is None else now
    table.update_item(Key={"document_id": snapshot["document_id"]},
        UpdateExpression="SET #status=:indexed, content_verified=:verified, indexed_sha256=:hash, "
                         "indexed_pipeline_sha256=:pipeline, indexed_chunk_count=:count, indexed_at=:at, "
                         "indexed_sequencer=:seq, index_instance_id=:instance, index_collection=:collection "
                         "REMOVE index_run_id, index_lease_until, last_index_error",
        ConditionExpression="#seq=:seq AND etag=:etag AND index_run_id=:run "
                            "AND #status=:indexing AND index_lease_until >= :now",
        ExpressionAttributeNames={"#status": "status", "#seq": "sequencer"},
        ExpressionAttributeValues={":indexed": "INDEXED", ":indexing": "INDEXING", ":verified": True,
            ":seq": snapshot["sequencer"], ":etag": snapshot["etag"], ":run": run_id, ":now": epoch,
            ":hash": snapshot["doc"]["expected_sha256"], ":pipeline": pipeline_hash, ":count": chunk_count,
            ":at": datetime.fromtimestamp(epoch, timezone.utc).isoformat(), ":instance": instance,
            ":collection": collection_name("rebuilt")})


def fail_if_owned(table, snapshot, run_id, code="LOCAL_INDEXING_FAILED"):
    """Only update this run's version; never overwrite a new upload or another worker."""
    from botocore.exceptions import ClientError, BotoCoreError
    try:
        table.update_item(Key={"document_id": snapshot["document_id"]},
            UpdateExpression="SET #status=:failed, last_index_error=:error, content_verified=:verified "
                             "REMOVE index_run_id, index_lease_until",
            ConditionExpression="#seq=:seq AND etag=:etag AND index_run_id=:run AND #status=:indexing",
            ExpressionAttributeNames={"#status": "status", "#seq": "sequencer"},
            ExpressionAttributeValues={":failed": "FAILED", ":error": code, ":verified": False,
                ":seq": snapshot["sequencer"], ":etag": snapshot["etag"], ":run": run_id, ":indexing": "INDEXING"})
        return "marked_failed"
    except ClientError as error:
        return "newer_version_or_owner_preserved" if error.response["Error"]["Code"] == "ConditionalCheckFailedException" else "failure_status_write_failed"
    except BotoCoreError:
        return "failure_status_write_failed"


def verified_local_count(collection, doc, pipeline_hash):
    if collection is None or collection.metadata.get("pipeline_sha256") != pipeline_hash:
        return 0
    data = collection.get(where={"doc_id": doc["id"]}, include=["metadatas"])
    if not data["ids"] or any(meta.get("file_sha256") != doc["expected_sha256"]
            or meta.get("pipeline_sha256") != pipeline_hash for meta in data["metadatas"]):
        return 0
    return len(data["ids"])


def already_current(snapshot, collection, instance, pipeline_hash):
    from .storage import cached_document_path
    item, doc = snapshot["item"], snapshot["doc"]
    count = verified_local_count(collection, doc, pipeline_hash)
    try:
        cached_document_path(doc)
    except ValueError:
        return False
    return (item.get("status") == "INDEXED" and item.get("content_verified") is True
            and item.get("indexed_sha256") == doc["expected_sha256"]
            and item.get("indexed_pipeline_sha256") == pipeline_hash
            and item.get("indexed_sequencer") == snapshot["sequencer"]
            and item.get("index_instance_id") == instance
            and item.get("index_collection") == collection_name("rebuilt")
            and int(item.get("indexed_chunk_count", -1)) == count and count > 0)


def run_registered_indexing(dry_run=False):
    from .ingestion import exclusive_index, pipeline_configuration, _ingest
    from .retrieval import open_collection
    from .storage import sync_documents
    from botocore.exceptions import ClientError, BotoCoreError
    with exclusive_index():
        session, owner, sdk = aws_session()
        plan = require_free_plan(session.client("freetier", region_name="us-east-1", config=sdk).get_account_plan_state())
        table = session.resource("dynamodb", config=sdk).Table(TABLE)
        s3 = session.client("s3", config=sdk)
        documents = read_json("documents.json")
        snapshots = [read_snapshot(doc, table, s3, owner) for doc in documents]
        if dry_run:
            return {"dry_run": True, "documents": [{"key": s["key"], "status": s["item"]["status"]} for s in snapshots],
                    "free_plan": plan, "aws_read_calls": True, "status_writes": 0, "bedrock_calls": 0}
        instance = local_instance_id()
        _, pipeline_hash = pipeline_configuration()
        try:
            collection = open_collection("rebuilt")
        except FileNotFoundError:
            collection = None
        pending = [snapshot for snapshot in snapshots if not already_current(snapshot, collection, instance, pipeline_hash)]
        if not pending:
            return {"status": "already_indexed", "documents": len(documents), "index_instance_id": instance,
                    "status_writes": 0, "downloads": 0, "new_chunks": 0, "bedrock_calls": 0, "free_plan": plan}
        run_id = str(uuid.uuid4())
        claimed, completed = [], []
        try:
            for snapshot in pending:
                claim_indexing(table, snapshot, run_id)
                claimed.append(snapshot)
            # First indexing of a registered version re-downloads and verifies bytes.
            # S3-cache-only ingestion then parses locally; no Bedrock or hosted embeddings.
            download = sync_documents(s3, owner, refresh=True)
            report = _ingest(source="s3")
            collection = open_collection("rebuilt")
            for snapshot in pending:
                assert_remote_unchanged(snapshot, s3, owner)
                count = verified_local_count(collection, snapshot["doc"], report["pipeline_sha256"])
                if count == 0:
                    raise RuntimeError("Local Chroma document verification failed")
                finish_indexing(table, snapshot, run_id, instance, report["pipeline_sha256"], count)
                completed.append({"key": snapshot["key"], "status": "INDEXED", "chunks": count})
        except Exception as error:
            finished = {item["key"] for item in completed}
            cleanup = [{"key": s["key"], "result": fail_if_owned(table, s, run_id)} for s in claimed if s["key"] not in finished]
            code = error.response["Error"]["Code"] if isinstance(error, ClientError) else type(error).__name__
            raise IndexingRunError({"status": "failed", "run_id": run_id, "completed": completed,
                "cleanup": cleanup, "error": f"Registered indexing stopped ({code}); inspect source versions, local index or IAM UpdateItem permission.",
                "bedrock_calls": 0}) from error
        return {"status": "completed", "run_id": run_id, "index_instance_id": instance, "indexed": completed,
                "previously_current": len(snapshots) - len(pending), "ingestion": report,
                "downloads": len(download["documents"]), "free_plan": plan, "bedrock_calls": 0,
                "meaning": "INDEXED refers to verified chunks in this local Chroma instance; the vector index is not hosted in AWS."}
