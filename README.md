# Agentic Financial RAG on AWS

Ask questions about financial reports and get answers with source citations. The project keeps PDF parsing, MiniLM embeddings, Chroma, and hybrid retrieval local, while using AWS for document storage, upload tracking, and answer generation.

## How it works

```text
PDF upload → S3 → Lambda → DynamoDB upload status
                    ↓
       Manually run local indexing worker
                    ↓
      PDF text → chunks → MiniLM → Chroma
                    ↓
Question → hybrid search → LangGraph → Bedrock Nova Micro → cited answer
```

- **S3** stores the source PDFs. An upload notification invokes **Lambda**, which validates the event and registers the current object in **DynamoDB**. **CloudWatch** records Lambda logs.
- The **local worker** runs only when you start it. It downloads approved PDFs, verifies their SHA-256 hashes, splits their text into chunks, creates MiniLM embeddings, and stores chunks and vectors in local Chroma. It then marks the document `INDEXED` in DynamoDB.
- For a question, **hybrid search** combines Chroma similarity with BM25 keyword matching. **LangGraph** lets the model request another search when needed. **Amazon Bedrock Nova Micro** receives the question and selected text passages, then writes the answer; it does not access Chroma or the full PDFs directly.
- Local checks validate citation IDs and metadata and flag some suspicious numbers. A small benchmark measures passage retrieval, separately from answer quality.

The AWS upload trigger **does not automatically index documents**. This is a learning prototype with a local index, not a fully hosted service. The three original PDFs and generated index are excluded from this public repository.

## Run locally

Tested on Windows with Python 3.13. From PowerShell in this folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Provide PDFs you are permitted to use and update their entries and hashes in `documents.json`. To use your own AWS resources, copy `aws.example.json` to the ignored `aws.json` and set your private bucket and IAM user. The public repository does not deploy or include the existing AWS resources.

```powershell
.\run.ps1 aws-login
.\run.ps1 index-registered --dry-run
.\run.ps1 index-registered
.\run.ps1 answer "What does the IMF say about global growth?"
.\run.ps1 answer "Compare Goldman Sachs and IMF forecasts for 2025" --json
```

`answer` calls Bedrock and uses AWS credits; S3 and DynamoDB operations can also incur charges. Local retrieval evaluation does not call Bedrock:

```powershell
.\run.ps1 evaluate --strategy hybrid --split dev
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Key files

| Path | Purpose |
|---|---|
| `src/finance_rag/` | Parsing, indexing, retrieval, agent, Bedrock adapter, answer checks, and CLI |
| `infra/registration_handler.py` | Lambda upload-event validation and DynamoDB registration |
| `evaluation/benchmark.json` | Draft questions and expected passages for retrieval evaluation |
| `documents.json`, `config.json`, `bedrock.json` | Document manifest and local/model settings |
| `run.ps1` | PowerShell command entry point |
| `docs/implementation-notes.md` | Detailed architecture, AWS setup, experiments, and limitations |

On 24 draft development questions, hybrid search found an average of **75% of the annotated passages in the top six**; adding nearby page context raised the context-coverage measure to **83%**. These are small, draft retrieval results, **not answer-accuracy scores**. Citation checks also cannot prove every claim is correct. See the [implementation notes](docs/implementation-notes.md) for methodology and tradeoffs.
