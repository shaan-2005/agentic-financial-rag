import argparse
import json
import re
import sys

from .settings import ROOT


def emit(value, output=None, display=None):
    payload = json.dumps(value, indent=2, ensure_ascii=False)
    print(payload if display is None else display)
    if output:
        target = (ROOT / output).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError("Report path must be inside the project")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")


def display_answer(report):
    # Keep internal chunk IDs in saved reports, but show readable citations in the terminal.
    answer = re.sub(r"\s*\[chunk:[^\]\s]+\]", "", report["answer"]).strip()
    lines = [answer]
    checks = report["answer_checks"]
    if not checks["automated_checks_passed"]:
        issues = [item["code"].replace("_", " ") for item in checks["errors"] + checks["warnings"]]
        lines.append("\nCheck: review needed" + (" (" + ", ".join(issues) + ")" if issues else ""))
    if report.get("usage", {}).get("provider") == "bedrock":
        lines.append(f"\nEstimated Bedrock cost: ${report['usage']['estimated_usd']:.6f}")
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Financial RAG with S3 documents and metered Bedrock answers.")
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect", help="Inspect original snapshot and PDF metadata")
    inspect.add_argument("--output")
    ingest = subs.add_parser("ingest", help="Index PDFs into a SEPARATE reproducible collection")
    ingest.add_argument("--output")
    ingest.add_argument("--source", choices=["local", "s3"], default="local")
    subs.add_parser("aws-login", help="Sign in through your browser using temporary credentials")
    smoke = subs.add_parser("bedrock-check", help="One small, metered Bedrock inference call")
    smoke.add_argument("--output")
    answers = subs.add_parser("audit-answers", help="Check saved smoke answers locally; no new Bedrock calls")
    answers.add_argument("--output", default="reports/answer-audit.json")
    permission = subs.add_parser("prepare-index-permissions", help="Prepare the one-table status-write permission")
    permission.add_argument("--output")
    worker = subs.add_parser("index-registered", help="Index approved registered S3 PDFs locally and update DynamoDB")
    worker.add_argument("--dry-run", action="store_true", help="Read registry/S3 state without downloading, indexing or changing status")
    worker.add_argument("--output", default="reports/registered-indexing.json")
    for name in ["prepare-registration", "deploy-registration", "registration-status", "connect-registration", "registration-check"]:
        cmd = subs.add_parser(name, help="AWS upload registration: prepare, deploy, connect or inspect")
        cmd.add_argument("--output")
        if name in {"deploy-registration", "connect-registration"}:
            cmd.add_argument("--apply", action="store_true", help="Apply the reviewed AWS change")
    evaluation = subs.add_parser("evaluate", help="Local passage retrieval benchmark; no AWS or generator calls")
    evaluation.add_argument("--index", choices=["baseline", "rebuilt"], default="rebuilt")
    evaluation.add_argument("--split", choices=["dev", "test"], default="dev")
    evaluation.add_argument("--filter-mode", choices=["none", "publisher"], default="none")
    evaluation.add_argument("--k", type=int, default=6)
    evaluation.add_argument("--strategy", choices=["dense", "hybrid", "hybrid-rerank"], default="dense")
    evaluation.add_argument("--query-mode", choices=["original", "focused", "multi-view"], default="original")
    evaluation.add_argument("--context-mode", choices=["none", "page-window"], default="none")
    evaluation.add_argument("--output", default="reports/retrieval-evaluation.json")
    for command in ["aws-check", "sync-s3"]:
        cmd = subs.add_parser(command)
        cmd.add_argument("--output")
        if command == "sync-s3":
            cmd.add_argument("--refresh", action="store_true", help="Download even if the checked cache exists")
    for command in ["retrieve", "answer"]:
        cmd = subs.add_parser(command)
        cmd.add_argument("question")
        cmd.add_argument("--index", choices=["baseline", "rebuilt"],
                         default="rebuilt" if command == "answer" else "baseline")
        cmd.add_argument("--publisher", choices=["Goldman Sachs", "JP Morgan", "IMF"])
        cmd.add_argument("--strategy", choices=["dense", "hybrid", "hybrid-rerank"])
        cmd.add_argument("--query-mode", choices=["original", "focused", "multi-view"])
        cmd.add_argument("--context-mode", choices=["none", "page-window"])
        cmd.add_argument("--output")
        if command == "answer":
            cmd.add_argument("--provider", choices=["bedrock", "groq"], default="bedrock")
            cmd.add_argument("--json", action="store_true", help="Print the full answer, evidence and diagnostics as JSON")
    args = parser.parse_args()
    try:
        if args.command in {"prepare-index-permissions", "index-registered"}:
            from .indexing_worker import IndexingRunError, prepare_index_permissions, run_registered_indexing
            from botocore.exceptions import ClientError, BotoCoreError
            try:
                report = prepare_index_permissions() if args.command == "prepare-index-permissions" else run_registered_indexing(args.dry_run)
            except IndexingRunError as error:
                emit(error.report, args.output)
                return 1
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code", "Unknown")
                raise RuntimeError(f"Registered indexing returned {code}; check the approved document uploads and scoped DynamoDB UpdateItem permission") from None
            except BotoCoreError:
                raise RuntimeError("Registered indexing connection failed; check browser login and retry after resolving it") from None
            emit(report, args.output)
            return 0
        if args.command in {"prepare-registration", "deploy-registration", "registration-status", "connect-registration", "registration-check"}:
            from . import registration
            from botocore.exceptions import BotoCoreError, ClientError
            try:
                if args.command == "prepare-registration":
                    report = registration.prepare_files()
                elif args.command == "deploy-registration":
                    report = registration.deploy(apply=args.apply)
                elif args.command == "registration-status":
                    report = registration.stack_status()
                elif args.command == "connect-registration":
                    report = registration.connect_notifications(apply=args.apply)
                else:
                    report = registration.check_registered_documents()
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code", "Unknown")
                raise RuntimeError(f"AWS registration returned {code}. Check the scoped IAM setup and stack events; no automatic permission expansion or plan upgrade was made.") from None
            except BotoCoreError:
                raise RuntimeError("AWS registration connection failed. Check browser login; no automatic plan upgrade was made.") from None
            emit(report, args.output)
            return 0
        if args.command == "audit-answers":
            from .answer_checks import audit_saved_answers
            emit(audit_saved_answers(), args.output)
            return 0
        if args.command == "evaluate":
            from .evaluation import evaluate
            report = evaluate(index=args.index, split=args.split, filter_mode=args.filter_mode, k=args.k,
                              strategy=args.strategy, query_mode=args.query_mode, context_mode=args.context_mode)
            # Explicit evaluation modes preserve historical runs when defaults change.
            emit(report, args.output)
            return 0
        if args.command == "bedrock-check":
            from .bedrock import smoke_test
            emit(smoke_test(), args.output)
            return 0
        if args.command == "aws-login":
            from .aws_access import login
            emit(login())
            return 0
        if args.command in {"aws-check", "sync-s3"}:
            from .aws_access import aws_clients
            from .storage import check_documents, sync_documents
            client, owner = aws_clients()
            report = (check_documents(client, owner) if args.command == "aws-check"
                      else sync_documents(client, owner, refresh=args.refresh))
            emit(report, args.output)
            return 0
        if args.command == "inspect":
            from .inspection import inspect_snapshot
            report = inspect_snapshot()
            emit(report, args.output)
            return 1 if report["issues"] else 0
        if args.command == "ingest":
            from .ingestion import ingest
            emit(ingest(source=args.source), args.output)
            return 0
        from .retrieval import Retriever
        if len(args.question) > 2000:
            raise ValueError("Keep questions under 2,000 characters for the learning experiment.")
        retriever = Retriever(args.index, strategy=args.strategy,
                              query_mode=args.query_mode, context_mode=args.context_mode)
        if args.command == "retrieve":
            emit({"question": args.question, "index": args.index,
                  "evidence": [e.as_dict() for e in retriever.search(args.question, args.publisher)]}, args.output)
        else:
            from langchain_core.messages import HumanMessage
            from .agent import build_agent
            from .bedrock import BedrockChat
            evidence = {}
            llm = BedrockChat() if args.provider == "bedrock" else None
            try:
                result = build_agent(retriever, publisher=args.publisher, llm=llm, provider=args.provider,
                                     evidence_store=evidence).invoke(
                    {"messages": [HumanMessage(content=args.question)], "tool_rounds": 0},
                    config={"recursion_limit": 12})
            except RuntimeError as error:
                emit({"question": args.question, "status": "failed", "error": str(error),
                      "usage": llm.usage_report() if llm is not None else {"provider": "groq"}}, args.output)
                return 1
            from .answer_checks import check_answer
            answer = result["messages"][-1].content
            checks = check_answer(answer, list(evidence.values()))
            report = {"question": args.question, "answer": answer,
                  "status": "automated_checks_passed" if checks["automated_checks_passed"] else "needs_review",
                  "answer_checks": checks,
                  "tool_rounds": result.get("tool_rounds", 0),
                  "evidence": list(evidence.values()),
                  "usage": llm.usage_report() if llm is not None else {"provider": "groq"},
                  "limitation": "Local diagnostics are not a semantic accuracy guarantee; generated claims require review."}
            emit(report, args.output, display=None if args.json else display_answer(report))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
