# Financial Document RAG

This project answers questions about financial reports using source citations.
It preserves the original notebook's MiniLM embeddings, Chroma search, PDF parsing,
overlapping chunks, and LangGraph model/tool loop while adding an AWS document
registration flow, hybrid retrieval, and passage-level evaluation.

## Current stage

S3 holds the source PDFs. An S3 event invokes Lambda, which records upload metadata
and status in DynamoDB. A manually run local worker verifies approved PDFs, creates
MiniLM embeddings, and builds a local Chroma index. At query time, hybrid search
finds passages and LangGraph lets Nova Micro request retrieval through Bedrock.
CloudWatch stores Lambda logs. The searchable index and indexing worker remain local;
the project is a learning prototype, not a fully hosted production service.

The draft retrieval benchmark measures whether expected passages were found; it
does not measure general answer accuracy. Citation checks catch structural issues
but cannot prove that a generated claim is true. The copied notebook is sanitized
and its saved outputs have been cleared.

## Files

- `notebooks/original_sanitized.ipynb`: reference copy of your notebook, without its key.
- `data/documents/`: three copied PDFs (398 pages total).
- `data/chroma/snapshot/`: untouched copy of the archived Chroma database.
- `data/chroma/working/`: copy used for baseline retrieval; Chroma can migrate this copy.
- `data/chroma/rebuilt/`: separate index produced by reproducible ingestion.
- `data/s3-cache/`: verified copies of PDFs downloaded from S3.
- `data/aws/`: temporary browser-login cache and project profile (never commit or share).
- `documents.json`: document identities, forecast years, and known publication dates.
- `config.json`: model revision, chunking, retrieval, and bounded agent settings.
- `aws.example.json`: copy to ignored `aws.json` and set your own bucket and IAM user.
- `bedrock.json`: model, region, request/output limits, and reference cost rates.
- `infra/bedrock-invoke-policy.json`: model-specific IAM policy to attach to the user.
- `evaluation/benchmark.json`: 40 draft evaluation questions with expected answers and evidence.
- `src/finance_rag/`: reusable components extracted from the notebook workflow.
- `reports/`: verification evidence; not an answer-quality evaluation.

Documents, databases, model downloads, credentials, account-specific AWS settings,
generated deployment templates, and run reports are excluded from Git. Copy
`aws.example.json` to `aws.json` before using AWS commands, then fill in the bucket
and IAM user you created. Provide your own approved PDFs and update their hashes in
`documents.json`; the reference corpus is not included in this public repository.
The sanitized notebook documents the original Colab workflow; use the commands
below to run the local application.

## Run in this workspace

Open PowerShell in this project folder. `run.ps1` uses the isolated environment
already set up in the workspace. No environment activation is necessary.

```powershell
.\run.ps1 inspect
.\run.ps1 retrieve "What does Goldman Sachs project for US growth in 2025?" --publisher "Goldman Sachs"
.\run.ps1 ingest --output reports/ingestion.json
.\run.ps1 retrieve "What risks does the IMF identify?" --index rebuilt --publisher IMF
```

The first model use downloads public MiniLM model files if not already cached in
`data/models/`. Subsequent retrieval and ingestion can run locally from the cache.
It does not call a paid inference endpoint. Model files are pinned by revision.

## Setup after moving the project

Tested with Python 3.13 on Windows. `requirements.lock.txt` records the tested
environment. The project itself is installed separately, so the lock contains
no machine-specific editable path. CPU PyTorch is installed from its official index.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\run.ps1 inspect
```

Keep the local `data/` directory when moving the project; Git does not contain it.
Other Python versions/platforms need separately validated dependency versions.

## How reproducible ingestion works

Each PDF is parsed page by page with pypdf, the parser used under the notebook's
PyPDFLoader. The original 1,000-character chunk size and 200-character overlap are
preserved using RecursiveCharacterTextSplitter.

Chunk IDs depend on document identity, file content hash, page, chunk position,
text, and pipeline configuration. Repeating ingestion does not create duplicates
or re-embed unchanged chunks. The rebuilt index is a full mirror of documents.json;
after successful inserts, it removes chunks no longer in that manifest. It always
uses a separate collection and leaves the original baseline intact.

Publication dates and forecast years are separate. JP Morgan's exact publication
date remains unknown until verified. Internal page indices are zero-based;
`pdf_page` is one-based PDF position and `page_label` is the document page label.
The saved baseline retains its original metadata.

The tested rebuilt index has **1,627 chunks**, versus **1,623** in the original.
JP Morgan has four additional chunks with the current parser dependencies; text
or chunk boundaries differ on 102 pages across the corpus. This is recorded in
`reports/ingestion-comparison.json`. We retain both indexes so parsing changes
cannot be confused with retrieval improvements. Repeating ingestion produced
zero new chunks and zero deletions.

Changing embeddings or chunking requires a new `rebuilt_collection` name. We do
not mix incompatible pipelines. Interrupted ingestion can be rerun. It is not a
transactional or concurrent production ingestion service yet.

## Optional Groq answers

Revoke the old Groq key first. Supply a newly generated key through your terminal
environment as `GROQ_API_KEY`; do not place it in source or notebooks. `.env.example`
is documentation only and is not automatically loaded.

```powershell
.\run.ps1 answer "What does the IMF say about global growth?" --publisher IMF --provider groq
```

The agent keeps your model/tool loop, with three retrieval rounds maximum,
three tool calls per round maximum, a model timeout, limited retries, and an output
token limit. These limits reduce runaway usage; they do not guarantee a spending
cap or grounded answers. Generated citations and claim support still need review.

## Bedrock answer generation

Bedrock is AWS's managed model service. Nova Micro is the text model selected for
this learning stage. It supports tool use in `us-east-1`, so we retain the existing
LangGraph retrieval loop rather than replacing it with a managed RAG product.

The distinction between our components is:

1. Chroma and local MiniLM find passages.
2. LangGraph coordinates retrieval and generation.
3. The Bedrock adapter translates messages and tool requests into Converse API format.
4. AWS runs Nova Micro and returns generated text or a request to call our local tool.
5. The tool runs locally; Bedrock does not execute it or read Chroma directly.

Attach `infra/bedrock-invoke-policy.json` as the `FinanceRagBedrockAccess` inline
policy on finance-rag-dev through the root IAM console. It allows only
`bedrock:InvokeModel` on `amazon.nova-micro-v1:0` in `us-east-1`. There are no
Marketplace subscription permissions, cross-region inference profiles, provisioned
throughput, or hosted model endpoints in this stage. The same temporary login
profile used for S3 is reused; no Bedrock API key is needed.

```powershell
# One very short, metered model call to verify the connection.
.\run.ps1 bedrock-check --output reports/bedrock-connection.json

# Use the existing rebuilt Chroma collection, which was verified with the S3 copies.
.\run.ps1 answer "What does JP Morgan recommend for portfolio resilience?" --publisher "JP Morgan" --output reports/bedrock-jpm.json
```

`answer` prints a short answer with readable citations and the estimated Bedrock
cost. Use `--json` to print the complete evidence and diagnostics in the terminal,
or `--output reports/my-answer.json` to save that full JSON while keeping the
terminal output short. The citation checks are safeguards, not a factual-accuracy
guarantee.

`answer` defaults to the rebuilt index and Bedrock. `retrieve` still defaults to
the preserved baseline so its original retrieval behavior remains available.
Use explicit `--index baseline` or `--index rebuilt` for comparison experiments.

The initial configuration limits each answer to four model requests, 768 output
tokens per request, and 48,000 UTF-8 bytes in each serialized request. A byte limit
is a request-size safeguard, not an exact token count or an account spending cap.
Inference calls have no automatic retries to avoid accidentally repeating metered
requests after lost responses. Graph retrieval rounds and tool calls remain bounded.
The app rejects a final answer if no retrieval evidence was returned; this does not
prove that every generated claim is supported by that evidence.

Responses include retrieved evidence, token counts, latency, and estimated cost.
Cost is calculated from the model's reported token usage at the AWS-published
reference rates in bedrock.json: $0.035 per million input tokens and $0.14 per
million output tokens. For example, 10,000 total input and 1,000 total output tokens
across all requests would be about $0.00049 before other charges/taxes and credits.
Check current regional prices before extended use. This estimate is not a billing
record or a guarantee of free-plan eligibility. Repeated conversation context is
counted again on each request.

Only retrieved text excerpts and the question are sent to the model; the full PDF
and the vector database stay local. Retrieved content is treated as untrusted
evidence. Citations are requested through a prompt and evidence IDs are recorded,
but citation support and factual correctness have not been evaluated yet. Token
limit stops and permission errors are reported rather than accepted as complete answers.

If the account is not eligible for this model, stop and retain the local project.
Do not upgrade the Free plan automatically to enable access.

Live verification completed on October 13, 2026:

- One connection check succeeded.
- Two document questions each completed one retrieval round and two model calls.
- A JP Morgan repeat verified the revised prompt and cleaned output.
- All cited chunk IDs in the samples existed in the returned retrieval evidence.
- Seventeen local tests passed, including tool-message translation, limits, error
  handling, and removal of internal-looking thinking wrappers from displayed text.
- Seven inference calls reported 9,665 input and 1,462 output tokens in total;
  estimated model usage was approximately $0.000543 at the reference rates.

The first JP Morgan output added a detail broader than its cited passage, and
retrieval still includes introductory material. The revised JP Morgan sample
lists the supported approaches without that added claim. These are integration
observations, not a benchmark score or proof of general answer correctness.
See `reports/bedrock-verification.json` for recorded usage and test reports;
`reports/bedrock-jpm-clean.json` contains the current clean sample. The original
test outputs are retained separately for debugging. Failed requests can have
unreported usage, so estimates cover reported token usage only.

References: [Nova Micro](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-micro.html),
[Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html),
[AWS pricing reference](https://aws.amazon.com/blogs/machine-learning/cost-efficient-custom-text-to-sql-using-amazon-nova-micro-and-amazon-bedrock-on-demand-inference/),
[Current Bedrock pricing](https://aws.amazon.com/bedrock/pricing/).

## S3 document source

Set your bucket name in the ignored `aws.json` file. This project uses `us-east-1`
and PDF objects under `documents/`. S3 holds originals, while parsing and embeddings stay local.
Browser console sign-in alone does not authenticate Python.

The installed AWS CLI must be v2.32 or later. Attach the AWS-managed
`SignInLocalDevelopmentAccess` policy to the IAM user using the root account's
IAM console. This grants OAuth local-login operations, not additional S3 access.
Then use the **finance-rag-dev** identity in the login browser, never root.

```powershell
.\run.ps1 aws-login
.\run.ps1 aws-check --output reports/s3-access.json
.\run.ps1 sync-s3 --output reports/s3-download.json
.\run.ps1 ingest --source s3 --output reports/s3-ingestion.json
```

`aws-login` attempts to open the AWS browser authorization flow and prints a sign-in
link if the browser does not open automatically. Credentials expire and are
cached in the ignored `data/aws/` folder; do not send this folder or authorization
codes to anyone. The app uses a named login profile and checks its identity before
reading S3. It rejects a root login and does not embed static access keys.

`aws-check` verifies access to each of the three expected object keys. `sync-s3`
downloads the manifest files and checks each SHA-256 against your verified original
PDF. Files are written to temporary paths and moved into place only after checks
pass. A failed or changed download cannot replace a good cached PDF.

Repeating `sync-s3` uses a matching local cache without downloading again. Use
`--refresh` to check S3 again and re-download. The ETag condition guards against a
remote object changing during the download; ETags are not treated as file hashes.
New report editions need an intentionally updated manifest checksum.

`ingest --source s3` reads only the verified S3 cache and fails if it is missing;
it does not silently fall back to the original local documents. Because the content
and source filenames remain the same, choosing S3 should not duplicate the rebuilt
index. The default `ingest` still uses your local PDFs.

These commands perform identity and S3 reads only. The live connection was verified
on October 13, 2026 using the finance-rag-dev IAM user's temporary browser credentials:

- All three PDFs were readable and encrypted with SSE-S3 (`AES256`).
- Every downloaded SHA-256 matched the verified local PDF.
- A second sync used all three cache entries without re-downloading.
- Ingestion from the S3 cache retained 1,627 chunks, with zero new chunks or removals.
- All four bucket public-access blocks were enabled; ACLs were disabled through
  BucketOwnerEnforced object ownership.

Evidence is stored in `reports/s3-access.json`, `s3-download.json`,
`s3-cache-repeat.json`, `s3-ingestion.json`, and `s3-security.json`.
Nine local tests also passed, including corrupt-download preservation, caching,
oversized-object rejection, missing-cache handling, and rejection of root profiles.
This verifies the storage connection, not answer accuracy.

Small S3 reads consume metered usage/credits; the app makes no Bedrock calls here.

References: [AWS CLI browser login](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html),
[Boto3 console credentials](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html#login-with-console-credentials),
[Local sign-in policy](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/SignInLocalDevelopmentAccess.html).

## Verification

```powershell
# Use your local .venv Python, or the workspace's work/.venv Python.
python -m unittest discover -s tests -v
python scripts/verify_baseline.py
```

Use the environment's Python explicitly if it is not activated. Verification checks
the saved record count and metadata, reproduces an embedding from each publisher,
tests self-retrieval and publisher-filtered questions, and tests the graph without
calling an external model. None of these tests measures answer quality.

## Retrieval evaluation: first baseline

Evaluation means checking the system against questions with known evidence. We
measure retrieval separately from generation so a missing passage is not confused
with a model's reasoning or citation error.

`evaluation/benchmark.json` contains 40 assistant-authored draft cases: 24 development
questions and 16 reserved test questions. The cases cover numbers, qualitative
statements, forecast-vintage comparisons, historical facts, assumptions, definitions,
and two unanswerable questions. Forty-two evidence annotations were checked against
the original PDF pages and the rebuilt Chroma passages. Independent human review
is still pending. Evidence is concentrated on a small number of pages; this is a
starter benchmark, not a representative corpus-wide assessment.

Test cases use different primary evidence pages from development cases. We ran
only the development questions. Keep the test-query results unseen until retrieval
choices are fixed. The dataset is corpus-aware and some topics have been explored
in earlier smoke tests, so this is not a blind external benchmark.

```powershell
.\run.ps1 evaluate --split dev --filter-mode none --output reports/retrieval-dev-dense.json
.\run.ps1 evaluate --split dev --filter-mode publisher --output reports/retrieval-dev-filtered.json
```

These commands use local MiniLM and Chroma; they make zero AWS/model API calls.
The default is the rebuilt index and top six passages. Publisher filtering is
provided from the named source in single-source questions. Multi-source questions
still use global retrieval; separate retrieval per requested publisher has not yet
been added to this evaluator.

| Development metric, top six passages | Dense search | Dense + publisher filter |
|---|---:|---:|
| Mean required-evidence recall | 39.6% | 52.1% |
| Questions with all required evidence | 9/24 (37.5%) | 12/24 (50.0%) |
| Mean reciprocal rank | 0.240 | 0.275 |
| Comparisons with both evidence items | 0/4 | 0/4 |

Required-evidence recall is the fraction of annotated evidence items found, averaged
per question. Full success requires all evidence, including both sources for a
comparison. MRR rewards finding the first required evidence item near the top;
it does not measure completion of a comparison.

Matches require the correct source filename and a normalized evidence excerpt.
Equivalent wording on another page is allowed. Duplicate chunks cannot inflate
recall. The labels do not cover every valid paraphrase or alternative passage, so
a valid answer-supporting retrieval can still be counted as a miss. Human review
and expanded alternative evidence should precede strong resume accuracy claims.

This directly evaluates question retrieval, not the agent's rewritten tool queries
or final answers. Unanswerable cases are excluded from retrieval metrics; an
empty gold set does not mean similarity search must return zero passages. Actual
abstention, answer correctness, numeric/unit interpretation, and citation support
need generation evaluation. Dedicated table/visual extraction tests are not yet
included. Warm query timing excludes model startup and should not be treated as
a production latency benchmark.

See `reports/benchmark-audit.json` for annotation checks and
`reports/retrieval-comparison.json` for the two development configurations.
The full per-question reports retain retrieved passages and missing evidence for
error analysis. Twenty-two local tests passed, including multi-source scoring,
wrong-source rejection, duplicate-hit handling, and unanswerable-case exclusion.

## Hybrid search and local reranking experiment

The original Chroma/MiniLM index is reused. No re-embedding or AWS inference is
needed. `src/finance_rag/retrieval.py` now offers three strategies:

- `dense`: the preserved semantic-similarity baseline.
- `hybrid`: semantic search plus an in-memory BM25 keyword index over the same
  chunks. Reciprocal rank fusion adds `1/(60 + rank)` from each ranking, so unlike
  raw distances and BM25 scores, the values can be combined consistently.
- `hybrid-rerank`: a local cross-encoder reads each question/passage pair and
  sorts up to 40 fused candidates per requested source. It uses CPU, a pinned
  model revision, and a 512-token input limit. Scores are relevance rankings,
  not calibrated probabilities. Long pairs can be truncated.

For the two new strategies, explicitly named publishers are detected from the
question text (never expected answers). Publisher names are removed from the
content query so repeated document headers do not dominate. Single-source
questions search that publisher; comparisons search each requested publisher
separately and share six slots equally, then interleave the results. An explicit
`--publisher` override takes precedence. Unnamed questions search globally.
Routing does not infer unnamed sources, resolve publication vintages, or ensure
that each returned passage supports the answer.

BM25 uses k1=1.5, b=.75, and corpus-wide document frequencies. Token normalization
handles line-break hyphenation and a few geography aliases. The in-memory keyword
index is suitable for this 1,627-chunk learning corpus; larger collections will
need a persistent search index. Rank fusion considers 40 candidates from each
method and keeps 40 fused candidates before optional reranking.

| Same 24 development questions; final top six | Dense + publisher filter | Hybrid + query routing | Hybrid + query routing + reranking |
|---|---:|---:|---:|
| Mean required-evidence recall | 52.1% | 75.0% | 70.8% |
| Questions with all annotated evidence | 12/24 | 17/24 | 16/24 |
| Mean reciprocal rank | 0.275 | 0.477 | 0.590 |
| Comparisons with both annotated items | 0/4 | 1/4 | 1/4 |

Hybrid is the default for the rebuilt index because it preserves more required
evidence. Reranking remains opt-in: it improves the first relevant passage's rank
but loses some evidence, notably the Goldman Eurozone/UK cases. On this local run,
warm-query median was about .04 seconds for hybrid and 2.18 seconds with reranking;
this is an indicative single-machine observation, not a production benchmark.
The baseline index continues to default to dense search.

This comparison measures a combined change (routing, query cleanup, and hybrid
retrieval), not the isolated benefit of BM25. No gold labels or questions changed.
Some new strategies regress on IMF global forecasts. Comparisons still fail in
three of four cases. Read `evaluation/development-review.md`: there are valid
alternative passages and partial chunks that exact-excerpt scoring does not
fully represent. Labels remain assistant-authored and await independent review.

```powershell
.\run.ps1 evaluate --strategy hybrid --split dev --output reports/retrieval-dev-hybrid.json
.\run.ps1 evaluate --strategy hybrid-rerank --split dev --output reports/retrieval-dev-reranked.json
.\run.ps1 retrieve "Compare Goldman Sachs and IMF US growth forecasts for 2025" --index rebuilt --strategy hybrid
```

The reranker downloads once from Hugging Face, then runs locally from the model
cache. It needs no API key and makes no AWS calls. Existing Sentence Transformers
dependencies suffice. Model and library revisions are pinned. See the official
[CrossEncoder usage guide](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
and [model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

`reports/retrieval-improvement.json` records the shared benchmark hash and all
four configurations. Twenty-six local tests passed, including keyword filtering,
fusion deduplication, query-based source routing and comparison quotas. No AWS
credits were used. The 16 reserved test queries remain unrun; generated-answer
accuracy, citation support and abstention have not been evaluated.

## Surrounding-page context and comparison query experiments

The rebuilt index now defaults to hybrid retrieval with the original content query
and bounded same-page context. The original matching chunk remains in `text`.
`context_text` contains up to 400 characters before and 800 after that chunk,
capped at 2,200 characters total and confined to the same PDF page. Page labels
and chunk IDs stay attached. The source PDF is checked against the manifest's
SHA-256 before its verified bytes are parsed. Missing PDFs or unmatched anchors
are marked unavailable; a changed PDF fails rather than supplying unverified text.
Readers and extracted pages are cached for the lifetime of the retriever.

This expands text at query time. It does not rebuild Chroma, re-embed documents,
download from S3, or change benchmark questions/labels. It currently needs local
manifest PDFs; a future cloud worker will need its own verified document cache.
The baseline index still defaults to dense retrieval without expansion.

| Same 24 development questions | Chunk-only recall | Recall including page context | Complete evidence with context | Comparison success with context |
|---|---:|---:|---:|---:|
| Hybrid, original query | 75.0% | **83.3%** | **19/24** | 1/4 |
| Hybrid, shorter comparison query | 72.9% | 81.3% | 19/24 | 1/4 |
| Hybrid, original + shorter query | 75.0% | 83.3% | 19/24 | 1/4 |
| Original + shorter query, reranked | 70.8% | 70.8% | 16/24 | 1/4 |

`focused` removes publisher names and comparison/publication boilerplate from
explicitly multi-publisher queries, while retaining years, topic and basis words.
`multi-view` fuses rankings from both original and focused views. Neither improved
comparison completion here; they remain experimental CLI options, not defaults.
They do not select among multiple publication vintages. Document-vintage filters
will be needed if the corpus gains several reports from the same publisher.

We separately score `metrics` on original chunk text and `context_metrics` on
expanded text. Bigger context makes more evidence available, so 83.3% must not
be described as a ranking-only improvement or final-answer accuracy. All 144
results in the selected development run had verified same-page context. The
largest six-result text payload, counting original plus expanded text, was 18,630
characters; metadata/JSON/prompt increase the actual request size. Bedrock's
existing request-size and model-call limits still apply. More context can also
increase input tokens and distract the generator; live answer quality and cost
were not tested in this local stage.

```powershell
.\run.ps1 evaluate --strategy hybrid --query-mode original --context-mode page-window --output reports/retrieval-dev-original-context.json
.\run.ps1 retrieve "Compare Goldman Sachs and IMF US growth forecasts for 2025" --index rebuilt
# Reproduce the original chunk-only experiment:
.\run.ps1 evaluate --strategy hybrid --query-mode original --context-mode none
```

Thirty-three local tests passed, including context size/anchor preservation,
page boundaries, PDF checksum rejection, separate scoring, and query-view
retention. `reports/retrieval-context-comparison.json` records all four experiments.
The 16 reserved test queries remain unrun. Zero AWS calls were made.

Comparison retrieval is still incomplete, and two IMF global-growth questions
remain misses. Review those failures and alternative evidence before fixing a
retrieval configuration for the reserved test set or making resume accuracy claims.

## Answer diagnostics: additions in this stage

The goal is to inspect generated answers separately from passage retrieval. This
stage adds local checks around the existing Bedrock/LangGraph flow. It creates
no AWS resources, calls no evaluation API, and makes no new inference calls.

| Step | Addition and purpose | AWS technology | Important code |
|---|---|---|---|
| 1 | Match each citation ID, publisher, year and page against evidence actually retrieved. | Existing Bedrock generates citations; Python checks them locally. | `answer_checks.py`: `check_answer()` |
| 2 | Flag percentage values absent from validly cited text; test a few expected concepts with draft case-specific patterns. | No additional AWS service. | `percentages()`, `fact_pattern_checks()`, `evaluation/answer-cases.json` |
| 3 | Allow the exact “Not found in retrieved documents.” response after a retrieval attempt returns no evidence, while still rejecting uncited invented answers. | Existing Bedrock model/tool loop; mocked models used for this stage's tests. | `agent.py`: `route()`, `answer_checks.py`: `is_abstention()` |
| 4 | Verify historical saved evidence against Chroma and, if applicable, verified PDF page context before auditing saved answers. | Reuses saved Bedrock outputs; no new inference. | `verify_saved_evidence()`, `audit_saved_answers()` |
| 5 | Add diagnostics and a review status to new CLI answer reports; preserve the generated answer for inspection. | Existing `bedrock-runtime` client and `Converse`, with unchanged request/call limits. | `cli.py`: `answer` and `audit-answers`; `bedrock.py`: `invoke()` |

Citation metadata integrity does NOT establish semantic citation support. A number
can appear in a cited passage but describe a different country, year or metric.
There is an explicit test demonstrating that this can pass the presence check.
The checker also does not identify every factual claim or verify every uncited
sentence, arithmetic result, unit conversion or table relationship. Reports
always say semantic review is required. `automated_checks_passed` is a diagnostic
status, not a declaration that the answer is factually correct.

Two historical smoke reports were replayed: the cleaned JP Morgan resilience
answer and the original IMF growth answer. Evidence IDs/text/provenance were
verified against the rebuilt index. Both pass citation metadata checks and their
draft expected fact patterns. The older IMF response is flagged for an unwanted
output wrapper, so only one of two passes every automated diagnostic. The existing
Bedrock cleaner already removes complete wrappers from new model responses; the
historical file was left intact. These are not fresh answers using the improved
retriever and cannot measure its end-to-end accuracy.

```powershell
.\run.ps1 audit-answers --output reports/answer-audit.json
```

The saved-answer audit costs zero new inference usage. A normal `answer` command
still invokes Bedrock and can consume credits; adding the local diagnostics does
not add model calls. Forty-one local tests passed, including fake citations,
wrong page/publisher/year, unsupported percentages, altered saved evidence and
strict abstention. Live abstention behaviour was not measured. The 16 reserved
retrieval test queries remain unrun. No AWS permissions were expanded.

For AWS learning: `session.client("bedrock-runtime")` chooses the inference API,
and `client.converse(...)` sends messages/tool definitions to Nova. IAM's existing
`bedrock:InvokeModel` permission authorizes that request. AWS generates the answer;
the local checker inspects the returned string. See the official
[Converse API documentation](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html).

## AWS document registration: deployed and live-verified

This cloud stage was deployed on October 13, 2026 in us-east-1. CloudFormation
reported CREATE_COMPLETE, the scoped S3 upload trigger was installed, and all
three original PDFs were re-uploaded with identical verified contents. Real S3
events invoked Lambda and produced three UPLOADED records in DynamoDB. A live
duplicate/older-event replay returned two ignored updates and left the current
record unchanged. CloudWatch application logs were read successfully.

The account-plan API confirmed an active Free plan with $106.19 in remaining
credits at deployment preflight, expiring December 1, 2026 in Asia/Calcutta.
That balance is an observation at that time, not a current billing guarantee.
These live checks made real S3/Lambda/DynamoDB/CloudWatch requests; eligible usage
can draw on allowances/credits. Bedrock was not invoked in this stage. Existing
S3 storage usage continues while files remain in the bucket.

See `reports/registration-verification.json`, `registration-check.json`,
`registration-live-replay.json`, `registration-live-logs.json` and
`registration-trigger-uploads.json` for the results. The setup instructions below
document how the deployment was prepared and can be reproduced; the named IAM
role and customer-managed deployment policy have already been created.

The deployed workflow is:

1. A lowercase `.pdf` is uploaded under this bucket's `documents/` prefix.
2. S3 sends a JSON notification containing the bucket, object key, size and event
   sequence. It sends metadata, not the PDF bytes.
3. Lambda checks that metadata belongs to the expected bucket/region/prefix,
   is a PDF filename and is between 1 byte and 50 MiB. It does not inspect PDF
   contents or verify a SHA-256 fingerprint.
4. A conditional DynamoDB update records one current item per bucket/key with
   `status=UPLOADED`, `content_verified=false`, ETag, size and timestamps.
5. Structured log messages identify the document by a hash and record outcomes.
   CloudWatch logs expire after seven days; document contents and whole event
   payloads are not logged by our handler.

Important code and concepts:

| Code | Purpose and AWS concept |
|---|---|
| `infra/registration_handler.py`: `lambda_handler()` | Entry point AWS invokes. Uses temporary credentials supplied through the Lambda execution role; it needs no browser login or access keys. |
| `upload_metadata()` | Decodes S3's URL-encoded object key and validates event fields. |
| `register_records()` | Writes metadata using `table.update_item()`. A DynamoDB condition permits a new record or a strictly newer sequencer. Duplicate/older events do not reset status. |
| `registration.py`: `build_template()` | Defines the function, table, log group, retry settings and bucket-only invocation permission in CloudFormation: infrastructure as code. The existing bucket and separately bootstrapped IAM role are not resources owned by this stack. |
| `runtime_policy()` / `deployment_policy()` | Separates the program's permissions from the human developer's deployment permissions. The developer cannot create or edit IAM roles or upgrade the plan through this policy. |
| `require_free_plan()` | Reads the Free Tier account-plan response and refuses missing, Paid, expired, zero-credit or malformed plan state. |
| `notification_plan()` | Adds our trigger while preserving other destinations. Refuses overlapping filters or a conflicting notification ID. |
| `check_registered_documents()` | Reads registration records for the three manifest PDFs; it does not run ingestion or Bedrock. |

```python
key = unquote_plus(obj["key"])
# S3 sends "annual+report.pdf"; this gives "annual report.pdf".
```

```python
ConditionExpression = (
    "attribute_not_exists(document_id) OR #length < :length "
    "OR (#length = :length AND #seq < :seq)"
)
```

The condition executes atomically inside DynamoDB. We normalize the hexadecimal
sequencer by removing leading zeros, then compare length before text, equivalent
to numeric ordering. Compare only events for the same bucket/key. This follows
AWS's recommendation because notifications can be duplicated or delivered out
of order. See [S3 event structure](https://docs.aws.amazon.com/AmazonS3/latest/userguide/notification-content-structure.html)
and [event ordering](https://docs.aws.amazon.com/AmazonS3/latest/userguide/notification-how-to-event-types-and-destinations.html).

### Cost and settings before deployment

The function uses Python 3.13, 128 MB and a 10-second timeout, with one retry for
function errors and a one-hour event-age limit. It has no provisioned concurrency,
VPC/NAT gateway, Bedrock access or S3 download/write permission. DynamoDB uses
Standard provisioned capacity at **1 RCU and 1 WCU**, with no autoscaling, replicas,
streams, backups or customer-managed encryption keys configured. CloudWatch has
seven-day log retention; no paid dashboard/custom metric/alarm is added.

At published US East rates, 1 RCU plus 1 WCU for 730 hours is about **$0.57/month
before allowances**. For 100 Lambda invocations averaging one second and 1 MiB of
logs, the added compute/request/log ingestion costs are fractions of a cent.
Plan under $1/month for this small registration workload before allowances;
existing S3/Bedrock usage, much larger volume, retries and unusual activity are
outside that estimate. This is not a bill or a hard spending cap. Available
monthly free allowances may cover this usage; they are shared with other usage.
The deployment also requires an active Free plan with remaining USD credits and
future expiry, verified through the API. It never calls the plan-upgrade API.

See `reports/registration-cost-plan.json` and official
[DynamoDB pricing](https://aws.amazon.com/dynamodb/pricing/),
[Lambda pricing](https://aws.amazon.com/lambda/pricing/),
[CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/),
and [GetAccountPlanState](https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_freetier_GetAccountPlanState.html).
Service access can still be restricted by account eligibility; failures stop
deployment rather than upgrading the account.

### Required IAM setup

These actions require an administrative sign-in because the development user
cannot grant itself new permissions. Use that sign-in only for the IAM setup,
then return to `finance-rag-dev`. Do not attach AdministratorAccess for this stage.

1. In IAM → Roles → Create role, choose **Custom trust policy** and paste
   `infra/registration-role-trust.json`. Create the role named exactly
   `finance-rag-registration-role`, without attaching managed permission policies.
2. On that role, add an inline JSON policy from
   `reports/registration-runtime-policy.json`; name it exactly
   `FinanceRagRegistrationRuntime`. It allows only DynamoDB UpdateItem on this
   table and CreateLogStream/PutLogEvents in this function's log group.
3. In IAM → Policies → Create policy, use JSON from
   `reports/registration-deploy-policy.json`; name it `FinanceRagRegistrationDeploy`.
   Attach this **customer-managed policy** to `finance-rag-dev`. Do not put it in
   the user's inline policies: their aggregate size limit is smaller.

The generated policies contain account-specific ARNs and are kept in ignored
reports, not source control. Regenerate them with `prepare-registration` if needed.
`iam:PassRole` lets the developer assign this exact execution role to Lambda;
it does not let the developer edit its permissions. The role verifier refuses
extra managed/inline policies or a different trust policy before deployment.

### Deployment and live verification after IAM setup

```powershell
# Local file generation and deployment preview; no AWS service calls:
.\run.ps1 prepare-registration
.\run.ps1 deploy-registration

# Real AWS operations, after the role/policy setup:
.\run.ps1 deploy-registration --apply --output reports/registration-deployment.json
.\run.ps1 registration-status --output reports/registration-status.json
# Wait for CREATE_COMPLETE. On failure, inspect stack events; do not upgrade.
.\run.ps1 connect-registration --output reports/registration-notification-plan.json
.\run.ps1 connect-registration --apply --output reports/registration-notification.json
```

The preview `connect-registration` makes read-only AWS calls; it does not change
the bucket until `--apply`. The S3 PUT replaces the notification configuration,
so the code preserves other destinations and blocks overlapping filters. S3 has
no conditional update for this operation: do not edit notifications concurrently.
Existing CloudFormation stacks are left unchanged; automatic updates are not
implemented. The S3 invocation permission is restricted to this bucket and account.

Upload notifications are not retroactive. After connecting, deliberately re-upload
the same three original PDFs under `documents/`, then run:

```powershell
.\run.ps1 registration-check --output reports/registration-check.json
```

The function's CloudWatch log group was checked, all three DynamoDB records were
read, and live duplicate/older replay left the latest row unchanged. The live
trigger, IAM execution and actual DynamoDB condition passed these checks;
53 local tests also passed, including the simulated event and failure cases.
`UPLOADED` means metadata registered, not
parsed/indexed. New files are not automatically added to `documents.json` or Chroma.
Bedrock is not invoked by the registration function.

This stage does not yet track S3 deletes or provide a failure queue. Failed events
can be lost after the retry/event-age limits, so inspect errors and re-upload/replay
when necessary. A production extension needs recovery and reconciliation.
For cleanup, remove only our notification ID from the existing bucket first,
then delete this stack. That deletes its table/log/function resources and their
demo metadata; the original S3 bucket/PDFs are not stack resources. The separately
created IAM role and customer-managed policy need separate administrative cleanup.

## Registered documents to the local index

Implemented and verified live on October 13, 2026. This connects the existing
registration service to our existing parser and Chroma ingestion; it does not
deploy a cloud vector database or change the original notebook/index.

1. S3 stores the PDF. Its upload event makes Lambda register `UPLOADED` in DynamoDB.
2. The manual laptop worker reads the registry and S3 metadata. It checks the
   active Free plan, then claims the current upload as `INDEXING` for 15 minutes.
3. It downloads the three approved manifest PDFs and checks their SHA-256 hashes.
4. The existing parser splits PDF text into chunks locally. The existing local
   embedding model creates vectors only for missing chunks; Chroma stores them.
5. After verifying stored text/metadata and checking that the S3 upload has not
   changed, it records `INDEXED`, chunk count, document hash and local index ID.
6. Later runs skip current documents when the registry, local cache and local
   index metadata agree. Querying still uses local retrieval followed by Bedrock.

```powershell
# Local policy generation; add UpdateItem to the existing named-table permissions:
.\run.ps1 prepare-index-permissions
# Save the generated JSON as the default version of FinanceRagRegistrationDeploy.

# Real AWS reads only; no downloads or status changes:
.\run.ps1 index-registered --dry-run
# Real S3 downloads and DynamoDB status writes, with local parsing/indexing:
.\run.ps1 index-registered
.\run.ps1 registration-check
```

Important code: `src/finance_rag/indexing_worker.py` connects these steps;
`src/finance_rag/storage.py` verifies downloads; `src/finance_rag/ingestion.py`
parses and verifies the index. The worker uses the laptop's verified AWS identity;
Lambda continues using its separate execution role. Only the laptop's existing
policy gains `dynamodb:UpdateItem` for the existing table.

The key guard is a DynamoDB conditional update: finishing requires the same
upload sequencer, ETag, worker run ID, `INDEXING` status and unexpired lease.
An old worker therefore cannot mark a newer upload indexed. A local file lock
also prevents two ingestion commands from writing this project simultaneously.
Failures mark only rows still owned by that run `FAILED`; a crashed worker's
lease can be reclaimed after expiry. S3 and DynamoDB are separate services, so
this is not an atomic cross-service transaction. Upload events can be delayed;
reconciliation and S3 versioning remain future improvements.
See [DynamoDB conditional updates](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html).

Live result: Goldman Sachs 520 chunks, JPMorgan 202, IMF 905 (1,627 total).
All three rows were independently read back as `INDEXED` with verified content
and current sequencers. Existing chunk IDs matched: zero new or removed chunks.
The repeat run reported zero downloads, status writes or new chunks. Reports:
`reports/registered-indexing.json`, `reports/registered-indexing-check.json`,
`reports/registered-indexing-repeat.json`. There are 63 passing local tests,
including simulated concurrent workers, expired leases, changed uploads and
parser failures. These tests use fakes; the live reports verify actual AWS access.

No new AWS resources or Bedrock requests are added in this stage. The first live
run downloaded about 30 MB and made six DynamoDB status updates plus reads.
These operations can consume free allowances/credits; the repeat still performs
AWS reads. Existing S3 storage and provisioned table capacity continue accruing
eligible usage. The Free plan API reported ACTIVE and $126.19 remaining at the
test time; that is a balance snapshot, not a guarantee of zero service usage.

`INDEXED` means indexed on this laptop, identified by a local UUID. Lambda does
not start the laptop, and newly uploaded filenames are not automatically approved
or indexed: add reviewed documents to the manifest before running this worker.
It does not measure answer quality; retrieval and answer evaluation stay separate.

## Learning sequence

1. Understand and verify the local baseline and ingestion.
2. Store PDFs privately in S3; learn IAM permissions and caching.
3. Add an evaluation benchmark, then measure hybrid retrieval and local reranking.
4. Integrate Bedrock with bounded model calls; compare grounded answers and cost.
5. Add S3-triggered Lambda registration, DynamoDB document status, and CloudWatch logs.
6. Connect registered uploads to verified local indexing and guarded status updates.

AWS stages use your Free plan only after checking remaining credits, expiry, and
service eligibility. As shown in your screenshot, access ends December 1, 2026,
or earlier if credits run out. Keep local copies and do not upgrade the account
as part of this project without an explicit decision.
