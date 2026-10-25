"""Use project-scoped browser login; never collect or print secret credentials."""
import configparser
import os
import shutil
import subprocess

from .settings import ROOT, read_json


def configure_auth_paths():
    folder = ROOT / "data/aws"
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["AWS_CONFIG_FILE"] = str(folder / "config")
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(folder / "credentials")
    os.environ["AWS_LOGIN_CACHE_DIRECTORY"] = str(folder / "login-cache")
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    return folder


def login():
    config = read_json("aws.json")
    configure_auth_paths()
    executable = shutil.which("aws")
    if not executable:
        raise RuntimeError("AWS CLI v2.32+ is required for browser login.")
    result = subprocess.run([executable, "login", "--profile", config["profile"],
                             "--region", config["region"]], check=False)
    if result.returncode:
        raise RuntimeError("AWS login failed. Check SignInLocalDevelopmentAccess and try again.")
    verify_login_profile(config)
    return {"profile": config["profile"], "region": config["region"],
            "message": "Browser login configured. Run aws-check to verify your identity and document access."}


def verify_login_profile(config):
    folder = configure_auth_paths()
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(folder / "config", encoding="utf-8")
    section = "profile " + config["profile"]
    if not parser.has_section(section):
        raise RuntimeError("No project login found. Run .\\run.ps1 aws-login first.")
    arn = parser.get(section, "login_session", fallback="")
    if not arn.startswith("arn:aws:iam::") or not arn.endswith(":user/" + config["expected_user"]):
        raise RuntimeError("Login must use finance-rag-dev, not root or another user. Run aws-login again.")
    if any(parser.has_option(section, key) for key in ["aws_access_key_id", "aws_secret_access_key", "credential_process", "role_arn"]):
        raise RuntimeError("Use the dedicated browser-login profile, without static keys or external processes.")
    return arn


def verified_session():
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    config = read_json("aws.json")
    expected_arn = verify_login_profile(config)
    sdk_config = Config(connect_timeout=10, read_timeout=30,
                        retries={"mode": "standard", "total_max_attempts": 2})
    try:
        session = boto3.Session(profile_name=config["profile"], region_name=config["region"])
        identity = session.client("sts", config=sdk_config).get_caller_identity()
        if identity["Arn"] != expected_arn:
            raise RuntimeError("AWS identity does not match the project's IAM-user login.")
        return session, identity["Account"]
    except (BotoCoreError, ClientError) as error:
        raise RuntimeError("AWS credentials are missing, expired, or unavailable. Run aws-login again.") from error


def aws_clients():
    from botocore.config import Config
    session, owner = verified_session()
    sdk_config = Config(connect_timeout=10, read_timeout=30,
                        retries={"mode": "standard", "total_max_attempts": 2})
    return session.client("s3", config=sdk_config), owner
