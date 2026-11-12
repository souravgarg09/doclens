# DocLens

DocLens answers questions about born-digital PDF documents — research papers, reports — with
**evidence you can click**. It splits every page into typed regions (text, tables, charts,
equations), finds the regions relevant to a question, asks Gemini to answer *only* from those
regions, highlights the cited regions on the page, and re-checks the numbers in the answer
against the values extracted from the PDF.

B.Tech CSE major project, Amity University Punjab — Vidushi Chauhan, Anmol Sharma, Sourav Garg ·
supervisor Dr. Vishwanath Bijalwan.

```text
PDF upload ─▶ FastAPI stores it and queues a Redis job
            ─▶ parsing worker (all local, no cloud calls):
                 PyMuPDF render ─▶ DocLayout-YOLO regions
                   ├─ tables   ─▶ Table Transformer ─▶ typed cells
                   ├─ charts   ─▶ DePlot            ─▶ data series
                   └─ text     ─▶ PDF text layer
                 SigLIP embeddings ─▶ Qdrant · rows ─▶ PostgreSQL
Question   ─▶ LangGraph agent:
                 hybrid retrieval (SigLIP vectors + BM25 keywords)
               ─▶ rerank (priors + optional Gemini listwise)
               ─▶ context expansion (named tables/figures)
               ─▶ grounding guardrail ─▶ Gemini answer with [R id] citations
               ─▶ numeric verifier ─▶ one repair pass if a claim is unsupported
```

Gemini is used **only** at question time. Parsing never calls a cloud model.

## Folder layout

```text
DocLens/
├── backend/
│   ├── app/            FastAPI API, database models, LangGraph agent, services
│   ├── workers/        Redis parsing worker and the table/chart/layout parsers
│   ├── scripts/        download_models.py, benchmark.py
│   └── tests/          unit tests (verifier, embeddings, BM25, security guards)
├── frontend/           React + Vite + TypeScript interface
├── samples/            demonstration PDF
├── benchmarks/         benchmark reports (one .md + .json per run)
├── Documents/          midterm report/synopsis and university formats
├── docker-compose.yml  PostgreSQL, Redis, Qdrant (bound to this computer only)
├── SETUP_DOCLENS.cmd · START_DOCLENS.cmd · STOP_DOCLENS.cmd     Windows
└── setup.sh · start.sh · stop.sh                                macOS / Linux
```

## Requirements

- Docker Desktop (running)
- Node.js 20.19 or newer
- [uv](https://docs.astral.sh/uv/) — setup installs it on macOS/Linux; on Windows `winget install astral-sh.uv`
- A Gemini API key from <https://aistudio.google.com/apikey>
- 16 GB RAM recommended; about 6 GB of disk for packages and models
- An NVIDIA GPU (Windows/Linux) or Apple Silicon (Mac) is used automatically; CPU works but parses slower

## First-time setup (once per computer)

| Windows | macOS / Linux |
|---|---|
| Double-click `SETUP_DOCLENS.cmd` | `chmod +x *.sh && ./setup.sh` |

Setup creates `.env` (and asks for your Gemini key), creates the Python 3.12 environment inside
`backend/.venv`, installs the right PyTorch build for the machine (CUDA for NVIDIA, MPS for
Apple Silicon, CPU otherwise), installs the frontend, and **downloads all AI models once**
(about 2 GB) so the first document is not slowed down by downloads.

Do not copy `backend/.venv` between computers or operating systems — run setup on each machine.

## Every run

| | Windows | macOS / Linux |
|---|---|---|
| Start | double-click `START_DOCLENS.cmd` | `./start.sh` |
| Stop  | double-click `STOP_DOCLENS.cmd`  | `./stop.sh` |

Start launches Docker (if needed), the databases, the API, the parsing worker and the frontend in
the background, then opens <http://127.0.0.1:5173/>. The badge at the top left shows the state:
**Loading AI models…** for the first ~10–40 seconds after start, then **Ready**, and
**Parsing page X of Y** while a document is processed. Stop keeps all documents and answers.

Logs are in `.runtime/logs/`.

## Using it

1. Upload a PDF. A progress bar shows the page being analysed.
2. When it is ready, click **Parsed in …** above the page to see how long each stage took.
3. Ask a question (Enter). The answer cites regions like `R81`; click a citation to jump to and
   highlight that region on the page.
4. Read the verification line: `consistent` means every number in the answer matched an
   extracted value; `unverifiable`, `contradiction` and `citation_gap` need a human look.
   `Show processing trace` lists every agent step with its time.

## Measuring performance (benchmark)

With DocLens running, from `backend/`:

```bash
# Windows
.\.venv\Scripts\python.exe -m scripts.benchmark
# macOS / Linux
.venv/bin/python -m scripts.benchmark
```

It uploads the sample PDF three times, asks three questions per run, and writes
`benchmarks/<date>_<computer>.md` and `.json` with the machine, the device (CUDA/MPS/CPU),
upload-to-ready time, per-stage parse timings, per-step question timings and any Gemini quota
errors. Use `--no-questions` to measure parsing without spending Gemini quota, `--runs N` for
more repetitions, `--pdf` and `-q "question"` for other inputs.

To compare two laptops fairly, run the benchmark on each one after setup has finished (models
downloaded) and DocLens has started (models loaded). Setup time, model download and model
loading are deliberately kept out of the parse numbers and reported separately.

BENCHMARK_RESULTS_PLACEHOLDER

## Configuration (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — | required for questions |
| `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL` | `gemini-3.8-flash` / `gemini-3.5-flash-lite` | fallback is used when the main model is rate-limited (HTTP 429) |
| `GEMINI_RERANK` | `true` | one extra Gemini call per question to score retrieved regions; `false` halves Gemini usage |
| `TORCH_DEVICE` | `auto` | `auto` = CUDA → Apple MPS → CPU; force with `cuda`, `mps`, `cpu` |
| `CUDA_FP16` | `true` | half-precision inference on NVIDIA GPUs |
| `PRELOAD_MODELS` | `true` | load models when the worker starts instead of on the first upload |
| `LEXICAL_WEIGHT` | `1.0` | weight of BM25 keyword ranking in hybrid retrieval (0 = vectors only) |
| `TOP_K_RETRIEVE` / `TOP_K_CONTEXT` | `20` / `6` | candidates retrieved / regions sent to Gemini |
| `GROUNDING_THRESHOLD` | `0.35` | below this the agent refuses instead of guessing |
| `RENDER_DPI` | `144` | page render resolution |

## Checks

From `backend/` (use `.venv/bin/python` on macOS/Linux):

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

From `frontend/`:

```bash
npx tsc --noEmit -p . && npm run build
```

## Scope and honest limitations

- Built for born-digital English PDFs with ordinary tables and simple bar/line charts. Scanned
  PDFs (no text layer) are not supported.
- Equation OCR (`pix2tex`) is optional and not installed by default; equations keep their PDF text.
- The numeric verifier checks that numbers in the answer match extracted values or simple
  arithmetic on them. A `consistent` result does not prove the answer is correct in every
  respect (wrong series or unit can still match).
- Gemini's free tier has per-minute and per-day limits. Each question uses one or two calls
  (plus one repair call when verification fails).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Badge says **Parsing worker offline** | restart DocLens; read `.runtime/logs/worker*.log` |
| Badge stays on **Loading AI models…** for minutes | models are still downloading — run setup again while online |
| Document stuck in `queued` | the worker is not running (see above) |
| Answer says model unavailable / rate limited | Gemini quota reached; wait a minute or set `GEMINI_RERANK=false` |
| Port 8000 or 5173 already in use | run Stop first, then Start |
| Apple MPS operator error | set `TORCH_DEVICE=cpu` in `.env` and restart |
| Never run `docker compose down -v` | it erases all stored documents and vectors |
