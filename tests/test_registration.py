import copy
import importlib.util
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from botocore.exceptions import ClientError
from finance_rag import registration
from finance_rag.settings import ROOT, read_json

spec = importlib.util.spec_from_file_location("registration_handler", ROOT / "infra/registration_handler.py")
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)
BUCKET = read_json("aws.json")["bucket"]


def upload(seq="A", key="documents/annual+report.pdf", size=100):
    return {"eventVersion": "2.6", "eventSource": "aws:s3", "awsRegion": "us-east-1",
            "eventName": "ObjectCreated:Put", "eventTime": "2026-10-13T00:00:00Z",
            "s3": {"bucket": {"name": BUCKET}, "object": {"key": key, "size": size, "eTag": "etag", "sequencer": seq}}}


class ConditionalTable:
    def __init__(self):
        self.items = {}
        self.calls = []

    def update_item(self, **request):
        self.calls.append(request)
        key = request["Key"]["document_id"]
        values = request["ExpressionAttributeValues"]
        newer = (values[":length"], values[":seq"])
        if key in self.items and newer <= self.items[key]["sequence"]:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
        self.items[key] = {"sequence": newer, "status": values[":status"]}


class RegistrationHandlerTests(unittest.TestCase):
    def register(self, records, table):
        return handler.register_records({"Records": records}, table, BUCKET, "documents/", "us-east-1", 1024)

    def test_registers_metadata_and_decodes_key_without_download(self):
        table = ConditionalTable()
        result = self.register([upload()], table)
        self.assertEqual(result["registered"], 1)
        values = table.calls[0]["ExpressionAttributeValues"]
        self.assertEqual(values[":key"], "documents/annual report.pdf")
        self.assertEqual(values[":status"], "UPLOADED")
        self.assertFalse(values[":verified"])
        self.assertIn("attribute_not_exists", table.calls[0]["ConditionExpression"])

    def test_duplicates_and_old_events_do_not_reset_indexed_status(self):
        table = ConditionalTable()
        self.register([upload("0a")], table)
        key = next(iter(table.items))
        table.items[key]["status"] = "INDEXED"
        result = self.register([upload("000A"), upload("9")], table)
        self.assertEqual(result["duplicate_or_older"], 2)
        self.assertEqual(table.items[key]["status"], "INDEXED")

    def test_newer_sequence_with_different_length_resets_to_uploaded(self):
        table = ConditionalTable()
        self.register([upload("FF")], table)
        self.assertEqual(self.register([upload("100")], table)["registered"], 1)
        self.assertEqual(next(iter(table.items.values()))["sequence"], (3, "100"))

    def test_wrong_bucket_region_prefix_file_and_oversize_are_ignored(self):
        records = [upload(key="outside/report.pdf"), upload(key="documents/readme.txt"), upload(size=1025), upload(size=True), upload(seq="Z")]
        wrong_bucket = upload()
        wrong_bucket["s3"]["bucket"]["name"] = "other-bucket"
        wrong_region = upload()
        wrong_region["awsRegion"] = "eu-north-1"
        records.extend([wrong_bucket, wrong_region, upload(key="documents/../report.pdf")])
        table = ConditionalTable()
        self.assertEqual(self.register(records, table)["ignored"], 8)
        self.assertFalse(table.calls)

    def test_s3_test_notification_never_writes_a_record(self):
        table = ConditionalTable()
        result = handler.register_records({"Event": "s3:TestEvent"}, table, BUCKET, "documents/", "us-east-1", 1024)
        self.assertTrue(result["test_event"])
        self.assertFalse(table.calls)

    def test_database_failures_are_raised_for_retry(self):
        table = Mock()
        table.update_item.side_effect = ClientError({"Error": {"Code": "ProvisionedThroughputExceededException"}}, "UpdateItem")
        with self.assertRaises(ClientError):
            self.register([upload()], table)


class RegistrationInfrastructureTests(unittest.TestCase):
    def test_template_has_no_existing_bucket_or_iam_role_resource(self):
        template = registration.build_template()
        types = {item["Type"] for item in template["Resources"].values()}
        self.assertNotIn("AWS::S3::Bucket", types)
        self.assertNotIn("AWS::IAM::Role", types)
        props = template["Resources"]["DocumentTable"]["Properties"]
        self.assertEqual(props["ProvisionedThroughput"], {"ReadCapacityUnits": 1, "WriteCapacityUnits": 1})
        fn = template["Resources"]["RegistrationFunction"]["Properties"]
        self.assertEqual(fn["Code"]["ZipFile"], (ROOT / "infra/registration_handler.py").read_text(encoding="utf-8"))
        compile(fn["Code"]["ZipFile"], "index.py", "exec")
        self.assertEqual(template["Resources"]["AllowBucketInvoke"]["Properties"]["SourceAccount"], {"Ref": "AWS::AccountId"})

    def test_policies_cannot_modify_roles_or_upgrade_account(self):
        account = "123456789012"
        runtime = registration.runtime_policy(account)
        actions = {action for statement in runtime["Statement"] for action in statement["Action"]}
        self.assertEqual(actions, {"dynamodb:UpdateItem", "logs:CreateLogStream", "logs:PutLogEvents"})
        policy = registration.deployment_policy(account)
        actions = {action for statement in policy["Statement"] for action in statement["Action"]}
        self.assertNotIn("freetier:UpgradeAccountPlan", actions)
        self.assertNotIn("iam:PutRolePolicy", actions)
        self.assertNotIn("iam:CreateRole", actions)
        self.assertLess(len(json.dumps(policy, separators=(",", ":"))), 6144)

    def test_free_plan_guard_rejects_paid_expired_missing_and_zero_credit(self):
        now = datetime.now(timezone.utc)
        plan = {"accountPlanType": "FREE", "accountPlanStatus": "ACTIVE", "accountPlanExpirationDate": now + timedelta(days=1),
                "accountPlanRemainingCredits": {"amount": "100.00", "unit": "USD"}}
        self.assertEqual(registration.require_free_plan(plan, now)["plan"], "FREE")
        for changes in [{"accountPlanType": "PAID"}, {"accountPlanStatus": "EXPIRED"},
                        {"accountPlanExpirationDate": now - timedelta(days=1)},
                        {"accountPlanRemainingCredits": {"amount": "0", "unit": "USD"}},
                        {"accountPlanRemainingCredits": {"amount": "NaN", "unit": "USD"}},
                        {"accountPlanRemainingCredits": {"amount": "not-money", "unit": "USD"}}]:
            with self.assertRaises(RuntimeError):
                registration.require_free_plan(dict(plan, **changes), now)
        with self.assertRaises(RuntimeError):
            registration.require_free_plan({}, now)

    def test_notification_plan_preserves_other_destinations_and_is_repeatable(self):
        original = {"ResponseMetadata": {}, "EventBridgeConfiguration": {}, "TopicConfigurations": [
            {"Id": "images", "TopicArn": "other", "Events": ["s3:ObjectCreated:*"],
             "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": "images/"}]}}}]}
        saved = copy.deepcopy(original)
        desired, changed = registration.notification_plan(original, "function-arn")
        self.assertTrue(changed)
        self.assertEqual(original, saved)
        self.assertEqual(desired["TopicConfigurations"], original["TopicConfigurations"])
        self.assertIn("EventBridgeConfiguration", desired)
        self.assertNotIn("ResponseMetadata", desired)
        repeated, changed = registration.notification_plan(desired, "function-arn")
        self.assertFalse(changed)
        self.assertEqual(desired, repeated)

    def test_overlapping_notifications_are_not_replaced(self):
        original = {"QueueConfigurations": [{"Id": "all", "Events": ["s3:ObjectCreated:*"]}]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            registration.notification_plan(original, "function-arn")

    def test_dry_run_makes_no_aws_calls(self):
        from unittest.mock import patch
        with patch.object(registration, "aws_session") as session:
            self.assertFalse(registration.deploy()["deployed"])
            session.assert_not_called()
