"""Prepare and deploy a bounded AWS registration demo; dry runs are local."""
import json
from datetime import datetime, timezone
from decimal import Decimal

from .settings import ROOT, read_json

STACK = "finance-rag-registration"
FUNCTION = "finance-rag-register-document"
TABLE = "finance-rag-documents"
ROLE = "finance-rag-registration-role"
ROLE_POLICY = "FinanceRagRegistrationRuntime"
LOG_GROUP = "/aws/lambda/" + FUNCTION
NOTIFICATION_ID = "finance-rag-pdf-registration"


def arns(account):
    if not str(account).isdigit() or len(str(account)) != 12:
        raise ValueError("Expected a 12-digit AWS account ID, not credentials")
    region = read_json("aws.json")["region"]
    if region != "us-east-1":
        raise ValueError("This reviewed learning deployment is limited to us-east-1")
    return {"role": f"arn:aws:iam::{account}:role/{ROLE}",
            "function": f"arn:aws:lambda:{region}:{account}:function:{FUNCTION}",
            "table": f"arn:aws:dynamodb:{region}:{account}:table/{TABLE}",
            "logs": f"arn:aws:logs:{region}:{account}:log-group:{LOG_GROUP}",
            "stack": f"arn:aws:cloudformation:{region}:{account}:stack/{STACK}/*"}


def runtime_policy(account):
    names = arns(account)
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["dynamodb:UpdateItem"], "Resource": names["table"]},
        {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": names["logs"] + ":*"}]}


def deployment_policy(account):
    names = arns(account)
    bucket = read_json("aws.json")["bucket"]
    def allow(actions, resources, condition=None):
        statement = {"Effect": "Allow", "Action": actions, "Resource": resources}
        if condition:
            statement["Condition"] = condition
        return statement
    return {"Version": "2012-10-17", "Statement": [
        allow(["freetier:GetAccountPlanState", "cloudformation:ValidateTemplate", "logs:DescribeLogGroups"], "*"),
        allow(["cloudformation:CreateStack", "cloudformation:DescribeStacks", "cloudformation:DescribeStackEvents",
               "cloudformation:DescribeStackResources", "cloudformation:GetTemplate", "cloudformation:DeleteStack"], names["stack"]),
        allow(["iam:GetRole", "iam:GetRolePolicy", "iam:ListRolePolicies", "iam:ListAttachedRolePolicies"], names["role"]),
        allow(["iam:PassRole"], names["role"], {"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}}),
        allow(["lambda:CreateFunction", "lambda:GetFunction", "lambda:GetFunctionConfiguration", "lambda:GetPolicy",
               "lambda:ListTags", "lambda:TagResource", "lambda:UntagResource", "lambda:DeleteFunction",
               "lambda:GetRuntimeManagementConfig", "lambda:GetFunctionCodeSigningConfig",
               "lambda:PutFunctionEventInvokeConfig", "lambda:GetFunctionEventInvokeConfig",
               "lambda:DeleteFunctionEventInvokeConfig", "lambda:InvokeFunction"],
              [names["function"], names["function"] + ":$LATEST"]),
        allow(["lambda:AddPermission"], names["function"], {"StringEquals": {"lambda:Principal": "s3.amazonaws.com"}}),
        allow(["lambda:RemovePermission"], names["function"]),
        allow(["dynamodb:CreateTable", "dynamodb:DescribeTable", "dynamodb:UpdateTable", "dynamodb:DeleteTable",
               "dynamodb:DescribeContinuousBackups", "dynamodb:DescribeTimeToLive", "dynamodb:ListTagsOfResource", "dynamodb:TagResource",
               "dynamodb:UntagResource", "dynamodb:GetItem", "dynamodb:UpdateItem"], names["table"]),
        allow(["logs:CreateLogGroup", "logs:DeleteLogGroup", "logs:PutRetentionPolicy", "logs:DeleteRetentionPolicy",
               "logs:DescribeLogStreams", "logs:GetLogEvents", "logs:ListTagsForResource", "logs:TagResource",
               "logs:UntagResource"], [names["logs"], names["logs"] + ":*"]),
        allow(["s3:GetBucketNotification", "s3:PutBucketNotification"], "arn:aws:s3:::" + bucket)
    ]}


def build_template():
    config = read_json("aws.json")
    if config["region"] != "us-east-1":
        raise ValueError("Deploy the registration stack in us-east-1")
    code = (ROOT / "infra/registration_handler.py").read_text(encoding="utf-8")
    return {"AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Finance RAG upload metadata registration only; existing S3 bucket remains external. Create the scoped Lambda execution role first.",
        "Resources": {
            "DocumentTable": {"Type": "AWS::DynamoDB::Table", "Properties": {
                "TableName": TABLE, "BillingMode": "PROVISIONED", "TableClass": "STANDARD",
                "AttributeDefinitions": [{"AttributeName": "document_id", "AttributeType": "S"}],
                "KeySchema": [{"AttributeName": "document_id", "KeyType": "HASH"}],
                "ProvisionedThroughput": {"ReadCapacityUnits": 1, "WriteCapacityUnits": 1}}},
            "RegistrationLogs": {"Type": "AWS::Logs::LogGroup", "Properties": {
                "LogGroupName": LOG_GROUP, "RetentionInDays": 7}},
            "RegistrationFunction": {"Type": "AWS::Lambda::Function", "DependsOn": ["RegistrationLogs"], "Properties": {
                "FunctionName": FUNCTION, "Runtime": "python3.13", "Handler": "index.lambda_handler",
                "MemorySize": 128, "Timeout": 10, "Architectures": ["x86_64"],
                "Role": {"Fn::Sub": "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/" + ROLE},
                "Code": {"ZipFile": code}, "Environment": {"Variables": {
                    "TABLE_NAME": {"Ref": "DocumentTable"}, "BUCKET_NAME": config["bucket"],
                    "DOCUMENT_PREFIX": config["prefix"], "MAX_DOCUMENT_BYTES": str(config["max_document_bytes"])}}}},
            "RetryLimits": {"Type": "AWS::Lambda::EventInvokeConfig", "Properties": {
                "FunctionName": {"Ref": "RegistrationFunction"}, "Qualifier": "$LATEST",
                "MaximumRetryAttempts": 1, "MaximumEventAgeInSeconds": 3600}},
            "AllowBucketInvoke": {"Type": "AWS::Lambda::Permission", "Properties": {
                "FunctionName": {"Ref": "RegistrationFunction"}, "Action": "lambda:InvokeFunction",
                "Principal": "s3.amazonaws.com", "SourceArn": "arn:aws:s3:::" + config["bucket"],
                "SourceAccount": {"Ref": "AWS::AccountId"}}}},
        "Outputs": {"FunctionArn": {"Value": {"Fn::GetAtt": ["RegistrationFunction", "Arn"]}},
                    "TableName": {"Value": {"Ref": "DocumentTable"}}, "LogGroup": {"Value": LOG_GROUP}}}


def prepare_files():
    from .aws_access import verify_login_profile
    config = read_json("aws.json")
    account = verify_login_profile(config).split(":")[4]
    files = {"infra/registration-template.json": build_template(),
             "reports/registration-runtime-policy.json": runtime_policy(account),
             "reports/registration-deploy-policy.json": deployment_policy(account),
             "reports/registration-cost-plan.json": cost_plan()}
    for name, value in files.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return {"prepared": list(files), "execution_role": ROLE, "runtime_policy_name": ROLE_POLICY,
            "trust_policy": "infra/registration-role-trust.json", "aws_calls": 0, "deployed": False}


def cost_plan():
    return {"region": "us-east-1", "estimate_date": "2026-10-13",
            "assumptions": {"dynamodb_read_units": 1, "dynamodb_write_units": 1, "hours_per_month": 730,
                            "lambda_invocations": 100, "lambda_memory_mb": 128, "average_duration_seconds": 1,
                            "logs_ingested_mib": 1, "log_retention_days": 7},
            "before_allowances_usd": {"dynamodb_capacity_month": round(730 * (.00065 + .00013), 4),
                                      "lambda_compute_and_requests": round(100 * .125 * .0000166667 + 100 / 1e6 * .20, 6),
                                      "logs_ingestion": round(.50 / 1024, 6)},
            "planning_total": "Under $1/month for these assumptions, excluding existing S3/Bedrock usage, substantial retries or higher volume; not a bill or hard spending cap.",
            "free_allowances": "1 RCU/1 WCU is within DynamoDB Standard's 25 RCU/25 WCU allowance if available; Lambda and small log usage can also fall within available monthly allowances. Allowances are shared with other account usage.",
            "cash_constraint": "Deployment requires the API to confirm an active FREE plan with remaining USD credits and future expiry. Never upgrade automatically.",
            "sources": ["https://aws.amazon.com/dynamodb/pricing/", "https://aws.amazon.com/lambda/pricing/",
                        "https://aws.amazon.com/cloudwatch/pricing/",
                        "https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_freetier_GetAccountPlanState.html"]}


def require_free_plan(plan, now=None):
    """Never upgrade; missing/paid/expired state stops deployment."""
    now = now or datetime.now(timezone.utc)
    expiration = plan.get("accountPlanExpirationDate")
    credits = plan.get("accountPlanRemainingCredits", {})
    try:
        amount = Decimal(str(credits.get("amount", "0")))
        active_date = isinstance(expiration, datetime) and expiration.tzinfo is not None and expiration > now
    except (ArithmeticError, ValueError, TypeError):
        raise RuntimeError("AWS account-plan response is incomplete; stop before deployment") from None
    if (plan.get("accountPlanType") != "FREE" or plan.get("accountPlanStatus") != "ACTIVE"
            or not active_date or credits.get("unit") != "USD" or not amount.is_finite() or amount <= 0):
        raise RuntimeError("Cannot verify an active Free plan with remaining USD credits. Stop; do not upgrade.")
    return {"plan": "FREE", "status": "ACTIVE", "credits_remaining_usd": str(credits["amount"]),
            "expires_at": expiration.isoformat()}


def verify_runtime_role(iam, account):
    role = iam.get_role(RoleName=ROLE)["Role"]
    if role["Arn"] != arns(account)["role"]:
        raise RuntimeError("Unexpected Lambda execution role")
    expected_trust = read_json("infra/registration-role-trust.json")
    trust = role["AssumeRolePolicyDocument"]
    if trust != expected_trust:
        raise RuntimeError("Lambda execution role trust differs from the reviewed policy")
    attached = iam.list_attached_role_policies(RoleName=ROLE)["AttachedPolicies"]
    inline = iam.list_role_policies(RoleName=ROLE)["PolicyNames"]
    if attached or inline != [ROLE_POLICY]:
        raise RuntimeError("Execution role must have only FinanceRagRegistrationRuntime, with no managed policies")
    policy = iam.get_role_policy(RoleName=ROLE, PolicyName=ROLE_POLICY)["PolicyDocument"]
    if policy != runtime_policy(account):
        raise RuntimeError("Execution role permissions differ from the scoped runtime policy")


def aws_session():
    from .aws_access import verified_session
    from botocore.config import Config
    session, account = verified_session()
    return session, account, Config(connect_timeout=10, read_timeout=30,
                                   retries={"mode": "standard", "total_max_attempts": 2})


def deploy(apply=False):
    if not apply:
        return {"deployed": False, "stack": STACK, "region": "us-east-1", "aws_calls": 0,
                "resources": [FUNCTION, TABLE, LOG_GROUP], "requires": "Bootstrap the scoped role and deployment policy, then use --apply"}
    session, account, sdk = aws_session()
    plan = require_free_plan(session.client("freetier", region_name="us-east-1", config=sdk).get_account_plan_state())
    verify_runtime_role(session.client("iam", config=sdk), account)
    cloudformation = session.client("cloudformation", config=sdk)
    template = json.dumps(build_template())
    cloudformation.validate_template(TemplateBody=template)
    from botocore.exceptions import ClientError
    try:
        existing = cloudformation.describe_stacks(StackName=STACK)["Stacks"][0]
    except ClientError as error:
        if error.response["Error"]["Code"] != "ValidationError" or "does not exist" not in error.response["Error"].get("Message", ""):
            raise
    else:
        return {"created": False, "status": existing["StackStatus"], "stack": STACK,
                "message": "Existing stack was left unchanged; inspect it before any update.", "free_plan": plan}
    import uuid
    response = cloudformation.create_stack(StackName=STACK, TemplateBody=template, ClientRequestToken=str(uuid.uuid4()))
    return {"created": True, "stack": STACK, "stack_id": response["StackId"], "status": "CREATE_IN_PROGRESS", "free_plan": plan}


def stack_status(session=None, sdk=None):
    if session is None:
        session, _, sdk = aws_session()
    stack = session.client("cloudformation", config=sdk).describe_stacks(StackName=STACK)["Stacks"][0]
    return {"stack": STACK, "status": stack["StackStatus"],
            "outputs": {item["OutputKey"]: item["OutputValue"] for item in stack.get("Outputs", [])}}


def notification_plan(existing, function_arn):
    """Preserve other destinations; refuse overlapping upload filters before PUT."""
    import copy
    config = read_json("aws.json")
    target = {"Id": NOTIFICATION_ID, "LambdaFunctionArn": function_arn, "Events": ["s3:ObjectCreated:*"],
              "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": config["prefix"]},
                                                  {"Name": "suffix", "Value": ".pdf"}]}}}
    desired = {key: copy.deepcopy(value) for key, value in existing.items() if key in
               {"LambdaFunctionConfigurations", "TopicConfigurations", "QueueConfigurations", "EventBridgeConfiguration"}}
    for kind in ["LambdaFunctionConfigurations", "TopicConfigurations", "QueueConfigurations"]:
        for item in desired.get(kind, []):
            if item.get("Id") == NOTIFICATION_ID:
                if kind == "LambdaFunctionConfigurations" and item == target:
                    return desired, False
                raise ValueError("The registration notification ID already exists with different settings")
            if not any(event.startswith("s3:ObjectCreated:") for event in item.get("Events", [])):
                continue
            rules = {rule["Name"]: rule["Value"] for rule in item.get("Filter", {}).get("Key", {}).get("FilterRules", [])}
            prefix, suffix = rules.get("prefix", ""), rules.get("suffix", "")
            if ((prefix.startswith(config["prefix"]) or config["prefix"].startswith(prefix))
                    and (suffix.endswith(".pdf") or ".pdf".endswith(suffix))):
                raise ValueError("Existing upload notifications overlap documents/*.pdf; review without replacing them")
    desired.setdefault("LambdaFunctionConfigurations", []).append(target)
    return desired, True


def connect_notifications(apply=False):
    session, account, sdk = aws_session()
    state = stack_status(session, sdk)
    if state["status"] != "CREATE_COMPLETE":
        raise RuntimeError("Registration stack must finish CREATE_COMPLETE before connecting S3")
    function = state["outputs"].get("FunctionArn")
    if function != arns(account)["function"]:
        raise RuntimeError("Unexpected Lambda destination ARN")
    client = session.client("s3", config=sdk)
    bucket = read_json("aws.json")["bucket"]
    existing = client.get_bucket_notification_configuration(Bucket=bucket, ExpectedBucketOwner=account)
    desired, changed = notification_plan(existing, function)
    if apply and changed:
        require_free_plan(session.client("freetier", region_name="us-east-1", config=sdk).get_account_plan_state())
        # S3 PUT replaces the notification document: keep all other destinations.
        # There is no conditional/ETag update; do not edit notifications concurrently.
        client.put_bucket_notification_configuration(Bucket=bucket, ExpectedBucketOwner=account,
                                                    NotificationConfiguration=desired)
    return {"bucket": bucket, "changed": bool(apply and changed), "applied": apply,
            "notification_configuration": desired,
            "message": "Only future uploads trigger registration. Existing files need a deliberate re-upload."}


def check_registered_documents():
    import hashlib
    from pathlib import Path
    session, account, sdk = aws_session()
    table = session.resource("dynamodb", config=sdk).Table(TABLE)
    config = read_json("aws.json")
    results = []
    for doc in read_json("documents.json"):
        key = config["prefix"] + Path(doc["path"]).name
        identifier = hashlib.sha256((config["bucket"] + "\0" + key).encode()).hexdigest()
        item = table.get_item(Key={"document_id": identifier}, ConsistentRead=True).get("Item")
        results.append({"key": key, "registered": item is not None,
                        "status": item.get("status") if item else None,
                        "content_verified": item.get("content_verified") if item else None,
                        "event_time": item.get("event_time") if item else None,
                        "indexed_chunks": int(item["indexed_chunk_count"]) if item and "indexed_chunk_count" in item else None,
                        "index_instance_id": item.get("index_instance_id") if item else None,
                        "indexed_sequence_current": item.get("indexed_sequencer") == item.get("sequencer") if item and "indexed_sequencer" in item else False})
    return {"table": TABLE, "documents": results, "all_registered": all(item["registered"] for item in results),
            "meaning": "UPLOADED is registration only. INDEXED records a verified document in the named local index instance, not a hosted vector database."}
