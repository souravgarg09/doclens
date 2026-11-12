# DocLens

DocLens is a multimodal document question-answering system for quantitative information in charts and tables within born-digital PDF documents. It parses each page into typed regions, retrieves evidence for a question, asks a multimodal reasoner to answer only from that evidence, displays region-grounded citations, and checks numeric claims against extracted values.

The active application has one backend, one worker, and one frontend:

```text
PDF upload
  -> FastAPI stores the file and queues a Redis job
  -> parsing worker renders pages and detects regions
  -> table/chart/equation specialists extract structured evidence
  -> SigLIP embeddings are stored in Qdrant; metadata is stored in PostgreSQL
  -> LangGraph retrieves, re-ranks, grounds, answers, and verifies
  -> React displays the answer and highlights cited page regions
```

## Repository structure

```text
DocLens/
|-- backend/
|   |-- app/                 FastAPI, database models, agent, and services
|   |-- workers/             Redis parsing worker and specialist parsers
|   |-- tests/               lightweight verifier regression tests
|   `-- requirements.txt
|-- frontend/                Vite, React, and TypeScript interface
|-- samples/                 demonstration input only
|-- .env.example             safe configuration template
|-- docker-compose.yml       PostgreSQL, Redis, and Qdrant
|-- SETUP_DOCLENS.cmd     one-time Windows setup
|-- START_DOCLENS.cmd     normal Windows launcher
|-- STOP_DOCLENS.cmd      normal Windows shutdown
`-- README.md
```

Generated uploads, rendered pages, crops, virtual environments, dependencies, and secrets are excluded from source control.

## Prerequisites

- `uv` 0.12 or later, or an existing Python 3.12 installation
- Node.js 20.19 or later
- Docker Desktop with Compose
- At least 16 GB RAM; the ML models require several gigabytes of disk space
- A Gemini API key for the question-answering endpoint

The worker selects CUDA first, then Apple MPS, then CPU when `TORCH_DEVICE=auto`. CPU works but parsing is considerably slower.

## First-time setup

Run all commands from this `DocLens` directory unless a step says otherwise.

### Windows: one-time setup

After extracting or cloning DocLens, double-click `SETUP_DOCLENS.cmd`. It checks the prerequisites, creates a project-specific Python 3.12 environment using `uv` when available or global Python 3.12 as a fallback, installs the backend and frontend dependencies, creates `.env`, and opens it so you can add your own Gemini API key. It does not install Python packages globally.

When setup reports success, use `START_DOCLENS.cmd` for every normal run. The first setup and first document parse require an internet connection and can take several minutes because dependencies and machine-learning model weights must be downloaded.

### Manual setup (all platforms)

1. Create the environment file.

   Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   macOS or Linux:

   ```bash
   cp .env.example .env
   ```

   Open `.env`, set `GEMINI_API_KEY`, and confirm `GEMINI_MODEL`. Never commit `.env`.

2. Start PostgreSQL, Redis, and Qdrant.

   ```bash
   docker compose up -d
   docker compose ps
   ```

3. Create the backend environment and install dependencies.

   Windows PowerShell:

   ```powershell
   Set-Location backend
   uv venv .venv --python 3.12
   .\.venv\Scripts\Activate.ps1
   uv pip install -r requirements.txt
   ```

   macOS or Linux:

   ```bash
   cd backend
   uv venv .venv --python 3.12
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

4. Install frontend dependencies.

   ```bash
   cd frontend
   npm ci
   ```

The first parse downloads the configured Hugging Face model weights. Keep the demonstration machine online during the first run and complete this download before presenting the application.

## Run the application

### Windows: every normal run

No reinstall or environment activation is needed after the first-time setup.

1. Open the `DocLens` folder in File Explorer and double-click `START_DOCLENS.cmd`.

   That is the normal startup method. It starts Docker Desktop when necessary, then starts the database services, API, parsing worker, and frontend in the background. It waits for them to become ready and opens the browser. Runtime logs are stored under `.runtime\logs\`.

   Alternatively, right-click inside the folder, choose **Open in Terminal**, and run:

   ```powershell
   .\start.ps1
   ```

2. When finished, double-click `STOP_DOCLENS.cmd`. The terminal equivalent is:

   ```powershell
   .\stop.ps1
   ```

`stop.ps1` stops the application and Docker containers but preserves documents, database records, and vectors. Use `.\stop.ps1 -KeepInfrastructure` if another local project still needs the Docker services.

To inspect or edit the project in VS Code, run `code .`; the editor does not need to remain open for DocLens to run. The launcher terminal and browser may also be closed after startup because the services run in the background. Existing successfully parsed documents remain available.

For troubleshooting, inspect `.runtime\logs\api.err.log`, `worker.err.log`, or `frontend.err.log`. Do not run `docker compose down -v` unless you intentionally want to erase stored database and vector data.

### Parsing performance

The worker loads each machine-learning model once and reuses it. Region embeddings are processed in page-sized batches, and region IDs are allocated with one database flush per page instead of one flush per region. This removes repeated SigLIP and database overhead without changing the parser's output format or retrieval behaviour.

`EMBEDDING_BATCH_SIZE=16` is a conservative default for CPU operation. A larger value can be faster on a dedicated GPU but uses more memory. `TORCH_DEVICE=auto` automatically selects CUDA, Apple MPS, or CPU. The first-ever parse remains slower because model weights may need to be downloaded and loaded into memory; later documents in the same worker session avoid that startup cost.

### macOS or Linux

Use three terminals.

Terminal 1 - API, from `backend/` with the virtual environment active:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 - parsing worker, from `backend/` with the same environment active:

```bash
python -m workers.parse_worker
```

Terminal 3 - frontend, from `frontend/`:

```bash
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/`. The API health endpoint is `http://127.0.0.1:8000/api/health`.

The configured storage directory is resolved from the project root, so API and worker data remain in the same `DocLens/data/` directory even though both processes start from `backend/`.

## Demonstration flow

1. Confirm the health badge reports the configured model and actual Torch device.
2. Upload `samples/muscle-hypertrophy-meta-analysis.pdf`.
3. Wait until the document status becomes `ready`.
4. Inspect detected table, chart, and text regions.
5. Ask a question tied to a visible table or figure.
6. Click each `[R id]` citation and confirm that the correct region is highlighted.
7. Explain the grounding score, numeric verification result, and agent trace.

Do not describe an answer as numerically verified unless the verifier status is `consistent`. `skipped` means the answer contained no numeric claim, while `unverifiable`, `contradiction`, and `citation_gap` require review.

## Checks before a demo

Backend syntax and verifier tests on Windows, from `backend/`:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app workers tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Backend syntax and verifier tests on macOS or Linux, from `backend/` with the virtual environment active:

```bash
python -m compileall -q app workers tests
python -m unittest discover -s tests -v
```

Frontend production build, from `frontend/`:

```bash
npm run build
```

Infrastructure:

```bash
docker compose ps
```

## Current implementation boundary

The current implementation targets single, born-digital English PDFs with ordinary tables and simple bar or line charts. It supports region citations, retrieval traces, and a conservative numeric-consistency check. Equation OCR is optional because `pix2tex` is not installed by the default requirements file. Scanned documents, large cross-document corpora, production deployment, and benchmark-scale evaluation are outside the currently implemented scope.

Numeric verification checks arithmetic consistency with extracted values; it does not prove that the parser selected the correct unit, year, series, or operand. All preliminary results must therefore be reported with the source region and any observed failure cases.

## Troubleshooting

- `Could not reach the parse queue`: verify Redis is healthy with `docker compose ps`, then re-parse the failed document.
- Document remains `parsing`: check the worker terminal for model-download, memory, or parser errors.
- Qdrant connection failure: verify port `6333` is available and the `qdrant` container is healthy.
- PostgreSQL connection failure: confirm `DATABASE_URL` and the `postgres` container.
- Slow first run: allow model downloads and caches to complete before timing the demo.
- MPS operator failure: set `TORCH_DEVICE=cpu` in `.env` and restart the API and worker.
- Missing equation OCR: keep it documented as an optional module, or install and test `pix2tex` before claiming it works.
