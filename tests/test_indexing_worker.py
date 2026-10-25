import copy
import unittest
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from finance_rag import indexing_worker as worker


def snapshot():
    doc = {"id": "doc", "path": "data/documents/report.pdf", "expected_sha256": "verified-hash"}
    return {"doc": doc, "document_id": "id", "key": "documents/report.pdf", "sequencer": "A", "etag": "etag",
            "last_modified": datetime(2026, 10, 13, tzinfo=timezone.utc),
            "item": {"status": "UPLOADED", "sequencer": "A", "etag": "etag"}}


class StateTable:
    """Simulate version/lease conditions, including adversarial orderings."""
    def __init__(self):
        self.item = {"status": "UPLOADED", "sequencer": "A", "etag": "etag"}
        self.calls = []

    def update_item(self, **request):
        values = request["ExpressionAttributeValues"]
        item = self.item
        self.calls.append(request)
        same_version = item["sequencer"] == values[":seq"] and item["etag"] == values[":etag"]
        claiming = ":uploaded" in values
        owned = item.get("index_run_id") == values[":run"] and item["status"] == "INDEXING"
        if claiming:
            allowed = same_version and (item["status"] in {"UPLOADED", "FAILED", "INDEXED"}
                      or (item["status"] == "INDEXING" and item.get("index_lease_until", 0) < values[":now"]))
        else:
            allowed = same_version and owned
            if ":indexed" in values:
                allowed = allowed and item.get("index_lease_until", 0) >= values[":now"]
        if not allowed:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        if claiming:
            item.update(status="INDEXING", index_run_id=values[":run"], index_lease_until=values[":until"])
        elif ":indexed" in values:
            item.update(status="INDEXED", content_verified=True, indexed_sha256=values[":hash"],
                        indexed_chunk_count=values[":count"], indexed_sequencer=values[":seq"])
            item.pop("index_run_id", None)
            item.pop("index_lease_until", None)
        else:
            item.update(status="FAILED", content_verified=False)
            item.pop("index_run_id", None)
            item.pop("index_lease_until", None)


class IndexingStateTests(unittest.TestCase):
    def test_only_one_worker_claims_a_current_lease(self):
        table = StateTable()
        worker.claim_indexing(table, snapshot(), "first", now=100)
        with self.assertRaises(ClientError):
            worker.claim_indexing(table, snapshot(), "second", now=101)
        self.assertEqual(table.item["index_run_id"], "first")

    def test_expired_lease_can_be_reclaimed_but_old_worker_cannot_finish(self):
        table = StateTable()
        worker.claim_indexing(table, snapshot(), "old", now=100)
        worker.claim_indexing(table, snapshot(), "new", now=1001)
        with self.assertRaises(ClientError):
            worker.finish_indexing(table, snapshot(), "old", "local", "pipeline", 5, now=1002)
        self.assertEqual(table.item["index_run_id"], "new")

    def test_new_upload_prevents_old_run_from_marking_indexed_or_failed(self):
        table = StateTable()
        worker.claim_indexing(table, snapshot(), "run", now=100)
        table.item.update(sequencer="B", status="UPLOADED")
        previous = copy.deepcopy(table.item)
        with self.assertRaises(ClientError):
            worker.finish_indexing(table, snapshot(), "run", "local", "pipeline", 5, now=101)
        self.assertEqual(worker.fail_if_owned(table, snapshot(), "run"), "newer_version_or_owner_preserved")
        self.assertEqual(table.item, previous)

    def test_completed_index_records_verified_hash_and_chunk_count(self):
        table = StateTable()
        worker.claim_indexing(table, snapshot(), "run", now=100)
        worker.finish_indexing(table, snapshot(), "run", "local", "pipeline", 5, now=101)
        self.assertEqual(table.item["status"], "INDEXED")
        self.assertTrue(table.item["content_verified"])
        self.assertEqual(table.item["indexed_chunk_count"], 5)
        self.assertEqual(table.item["indexed_sha256"], "verified-hash")
        self.assertNotIn("index_run_id", table.item)

    def test_expired_run_and_empty_document_cannot_be_marked_indexed(self):
        table = StateTable()
        worker.claim_indexing(table, snapshot(), "run", now=100)
        with self.assertRaises(ClientError):
            worker.finish_indexing(table, snapshot(), "run", "local", "pipeline", 5, now=1001)
        with self.assertRaises(ValueError):
            worker.finish_indexing(table, snapshot(), "run", "local", "pipeline", 0, now=101)

    def test_identical_content_reupload_is_detected_by_last_modified(self):
        s3 = Mock()
        s3.head_object.return_value = {"ETag": '"etag"', "LastModified": snapshot()["last_modified"] + timedelta(seconds=1)}
        with self.assertRaisesRegex(RuntimeError, "changed"):
            worker.assert_remote_unchanged(snapshot(), s3, "account")

    def test_missing_registration_cannot_start_a_download(self):
        table, s3 = Mock(), Mock()
        table.get_item.return_value = {}
        with self.assertRaisesRegex(RuntimeError, "registration is missing"):
            worker.read_snapshot(snapshot()["doc"], table, s3, "account")
        s3.head_object.assert_not_called()


class IndexingControllerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch("finance_rag.ingestion.exclusive_index", return_value=nullcontext()).start()
        patch("finance_rag.ingestion.pipeline_configuration", return_value=({}, "pipeline")).start()
        patch.object(worker, "local_instance_id", return_value="local").start()
        patch.object(worker, "read_json", return_value=[snapshot()["doc"]]).start()
        patch.object(worker, "read_snapshot", return_value=snapshot()).start()
        patch("finance_rag.retrieval.open_collection", return_value=Mock()).start()
        self.table = StateTable()
        self.session = Mock()
        self.session.resource.return_value.Table.return_value = self.table
        self.session.client.return_value.get_account_plan_state.return_value = {
            "accountPlanType": "FREE", "accountPlanStatus": "ACTIVE",
            "accountPlanExpirationDate": datetime.now(timezone.utc) + timedelta(days=1),
            "accountPlanRemainingCredits": {"amount": "100", "unit": "USD"}}
        patch.object(worker, "aws_session", return_value=(self.session, "account", None)).start()
        self.current = patch.object(worker, "already_current", return_value=False).start()
        self.download = patch("finance_rag.storage.sync_documents", return_value={"documents": [{}]}).start()
        self.ingest = patch("finance_rag.ingestion._ingest").start()

    def test_parser_failure_marks_owned_records_failed_without_success(self):
        self.ingest.side_effect = Exception("PDF parsing failed")
        with self.assertRaises(worker.IndexingRunError) as error:
            worker.run_registered_indexing()
        self.assertEqual(self.table.item["status"], "FAILED")
        self.assertEqual(error.exception.report["completed"], [])

    def test_repeat_current_run_skips_downloads_writes_and_ingestion(self):
        self.current.return_value = True
        report = worker.run_registered_indexing()
        self.assertEqual(report["status"], "already_indexed")
        self.assertEqual(self.table.calls, [])
        self.download.assert_not_called()
        self.ingest.assert_not_called()

    def test_dry_run_reads_state_without_claiming_or_ingesting(self):
        report = worker.run_registered_indexing(dry_run=True)
        self.assertTrue(report["aws_read_calls"])
        self.assertEqual(report["status_writes"], 0)
        self.assertEqual(self.table.calls, [])
        self.download.assert_not_called()
        self.ingest.assert_not_called()
