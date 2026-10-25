"""Standalone Lambda: register upload metadata, never download/parse/embed PDFs."""
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import unquote_plus

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def upload_metadata(record, bucket, prefix, region, max_bytes):
    """Read a direct S3 ObjectCreated event; reject unexpected sources and sizes."""
    if record.get("eventSource") != "aws:s3" or record.get("awsRegion") != region:
        raise ValueError("wrong_event_source_or_region")
    version = str(record.get("eventVersion", ""))
    if not re.fullmatch(r"2\.\d+", version) or int(version.split(".")[1]) < 1:
        raise ValueError("unsupported_event_version")
    if not str(record.get("eventName", "")).startswith("ObjectCreated:"):
        raise ValueError("not_an_upload_event")
    s3 = record["s3"]
    if s3["bucket"]["name"] != bucket:
        raise ValueError("wrong_bucket")
    obj = s3["object"]
    key = unquote_plus(obj["key"])
    if (not key.startswith(prefix) or not key.endswith(".pdf") or "\x00" in key
            or ".." in PurePosixPath(key).parts or len(key.encode("utf-8")) > 1024):
        raise ValueError("unexpected_object_key")
    size = obj["size"]
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= max_bytes:
        raise ValueError("invalid_document_size")
    sequence = obj["sequencer"]
    if not isinstance(sequence, str) or not re.fullmatch(r"[0-9a-fA-F]{1,1024}", sequence):
        raise ValueError("invalid_sequencer")
    # Numeric ordering without a fixed padding width: compare length, then hex text.
    sequence = sequence.upper().lstrip("0") or "0"
    event_time = record["eventTime"]
    parsed = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("event_time_missing_timezone")
    return {"document_id": hashlib.sha256((bucket + "\0" + key).encode()).hexdigest(),
            "bucket": bucket, "object_key": key, "size_bytes": size,
            "etag": str(obj.get("eTag", ""))[:256], "sequencer": sequence,
            "sequence_length": len(sequence), "event_time": event_time}


def register_records(event, table, bucket, prefix, region, max_bytes, now=None):
    """Atomic conditional writes prevent duplicate/older events resetting status."""
    from botocore.exceptions import ClientError
    if event.get("Event") == "s3:TestEvent":
        return {"registered": 0, "duplicate_or_older": 0, "ignored": 0, "test_event": True}
    records = event.get("Records")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty S3 Records array")
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    totals = {"registered": 0, "duplicate_or_older": 0, "ignored": 0}
    for record in records:
        try:
            item = upload_metadata(record, bucket, prefix, region, max_bytes)
        except (KeyError, TypeError, ValueError, AttributeError):
            totals["ignored"] += 1
            LOGGER.warning(json.dumps({"action": "ignored_invalid_upload"}))
            continue
        values = {":bucket": item["bucket"], ":key": item["object_key"], ":size": item["size_bytes"],
                  ":etag": item["etag"], ":seq": item["sequencer"], ":length": item["sequence_length"],
                  ":event_time": item["event_time"], ":registered": timestamp,
                  ":status": "UPLOADED", ":verified": False}
        try:
            table.update_item(
                Key={"document_id": item["document_id"]},
                UpdateExpression="SET bucket_name=:bucket, object_key=:key, size_bytes=:size, etag=:etag, "
                                 "#seq=:seq, #length=:length, event_time=:event_time, registered_at=:registered, "
                                 "#status=:status, content_verified=:verified",
                ConditionExpression="attribute_not_exists(document_id) OR #length < :length "
                                    "OR (#length = :length AND #seq < :seq)",
                ExpressionAttributeNames={"#seq": "sequencer", "#length": "sequence_length", "#status": "status"},
                ExpressionAttributeValues=values)
        except ClientError as error:
            if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
                totals["duplicate_or_older"] += 1
                continue
            LOGGER.error(json.dumps({"action": "registration_failed", "document_id": item["document_id"]}))
            raise  # Infrastructure failures must reach Lambda's retry mechanism.
        totals["registered"] += 1
        LOGGER.info(json.dumps({"action": "registered", "document_id": item["document_id"], "status": "UPLOADED"}))
    LOGGER.info(json.dumps({"action": "batch_complete", **totals}))
    return totals


def lambda_handler(event, context):
    """AWS invokes this entry point with the upload notification JSON."""
    import boto3
    from botocore.config import Config
    table = boto3.resource("dynamodb", config=Config(connect_timeout=2, read_timeout=3,
                      retries={"mode": "standard", "total_max_attempts": 2})).Table(os.environ["TABLE_NAME"])
    return register_records(event, table, os.environ["BUCKET_NAME"], os.environ["DOCUMENT_PREFIX"],
                            os.environ["AWS_REGION"], int(os.environ["MAX_DOCUMENT_BYTES"]))
