import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from finance_rag import aws_access, storage
from finance_rag.settings import ROOT


class S3DownloadTests(unittest.TestCase):
    def setUp(self):
        work = ROOT.parents[1] / "work"
        self.temporary = tempfile.TemporaryDirectory(dir=work, prefix="s3-test-")
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(work.resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(storage, "ROOT", self.root).start()
        self.contents = b"%PDF-1.7\nTest document bytes\n"
        self.expected = hashlib.sha256(self.contents).hexdigest()
        self.doc = {"path": "data/documents/report.pdf", "expected_sha256": self.expected}
        self.config = {"bucket": "test-project-bucket", "prefix": "documents/", "max_document_bytes": 1024}
        patch.object(storage, "read_json", side_effect=lambda name: self.config if name == "aws.json" else [self.doc]).start()
        self.client = Mock()
        self.client.head_object.return_value = {"ContentLength": len(self.contents), "ETag": '"version1"'}
        self.client.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(self.contents), "ServerSideEncryption": "AES256"}

    def test_first_download_is_verified_and_second_run_avoids_s3_reads(self):
        result = storage.sync_documents(self.client, "123456789012")
        self.assertEqual(result["documents"][0]["status"], "downloaded")
        self.client.get_object.assert_called_once_with(Bucket="test-project-bucket", Key="documents/report.pdf",
                                                       ExpectedBucketOwner="123456789012", IfMatch='"version1"')
        self.assertEqual(storage.cached_document_path(self.doc).read_bytes(), self.contents)
        self.client.reset_mock()
        result = storage.sync_documents(self.client, "123456789012")
        self.assertEqual(result["documents"][0]["status"], "cached")
        self.client.head_object.assert_not_called()
        self.client.get_object.assert_not_called()

    def test_bad_download_cannot_overwrite_good_cache(self):
        storage.sync_documents(self.client, "123456789012")
        self.client.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(b"corrupt")}
        with self.assertRaisesRegex(ValueError, "differs"):
            storage.sync_documents(self.client, "123456789012", refresh=True)
        self.assertEqual(storage.cached_document_path(self.doc).read_bytes(), self.contents)
        self.assertEqual(list((self.root / "data/s3-cache").glob("*.part")), [])

    def test_s3_source_never_silently_falls_back_to_original_pdf(self):
        original = self.root / self.doc["path"]
        original.parent.mkdir(parents=True)
        original.write_bytes(self.contents)
        with self.assertRaisesRegex(ValueError, "S3 cache missing"):
            storage.cached_document_path(self.doc)

    def test_oversized_object_is_rejected_before_download(self):
        self.client.head_object.return_value["ContentLength"] = 1025
        with self.assertRaisesRegex(ValueError, "too large"):
            storage.sync_documents(self.client, "123456789012")
        self.client.get_object.assert_not_called()

    def test_denied_read_reports_an_actionable_error(self):
        from botocore.exceptions import ClientError
        self.client.head_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
        with self.assertRaisesRegex(RuntimeError, "Check login, permissions"):
            storage.sync_documents(self.client, "123456789012")


class LoginIdentityTests(unittest.TestCase):
    def test_root_profile_is_rejected_before_any_service_call(self):
        work = ROOT.parents[1] / "work"
        with tempfile.TemporaryDirectory(dir=work, prefix="aws-login-test-") as name:
            folder = Path(name).resolve()
            self.assertTrue(folder.is_relative_to(work.resolve()))
            (folder / "config").write_text("[profile finance-rag-dev]\nlogin_session = arn:aws:iam::123456789012:root\n")
            with patch.object(aws_access, "configure_auth_paths", return_value=folder):
                with self.assertRaisesRegex(RuntimeError, "not root"):
                    aws_access.verify_login_profile({"profile": "finance-rag-dev", "expected_user": "finance-rag-dev"})


if __name__ == "__main__":
    unittest.main()
