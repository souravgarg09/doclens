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
                 PyMuPDF render ─▶ DocLayout-YOLO regions ─▶ drop empty/duplicate boxes
                   ├─ tables   ─▶ Table Transformer ─▶ typed cells
                   ├─ charts   ─▶ DePlot            ─▶ data series  (background phase, after "ready")
                   └─ text     ─▶ PDF text layer
                 SigLIP 2 embeddings ─▶ Qdrant · rows ─▶ PostgreSQL
Question   ─▶ LangGraph agent:
                 hybrid retrieval (SigLIP vectors + BM25 keywords)
               ─▶ rerank (similarity + keyword + structure priors; optional Gemini)
               ─▶ context expansion (named tables/figures)
               ─▶ grounding guardrail ─▶ Gemini answer with [R id] citations
               ─▶ numeric verifier ─▶ one repair pass if a claim is unsupported
```

Gemini is used **only** at question time. Parsing never calls a cloud model.

## Folder layout

```text
DocLens/
├── Setup.cmd · Start.cmd · Stop.cmd   double-click on Windows
├── setup.sh · start.sh · stop.sh      run on macOS (./start.sh)
├── brand.json                         product name, tagline, logo letter
├── docker-compose.yml                 PostgreSQL, Redis, Qdrant (bound to this computer only)
├── .env.example                       settings template (setup copies it to .env)
├── backend/
│   ├── app/            FastAPI API, database models, LangGraph agent, services
│   ├── workers/        Redis parsing worker and the table/chart/layout parsers
│   ├── scripts/        download_models.py, benchmark.py
│   └── tests/          unit tests
├── frontend/           React + Vite + TypeScript interface
├── scripts/windows/    PowerShell scripts behind the .cmd launchers
├── samples/            demonstration PDF
├── benchmarks/         benchmark reports (one .md + .json per run)
└── Documents/          Midterm/, Endterm/, University formats/

Created at run time and never committed: .env (your key), backend/.venv, frontend/node_modules,
data/ (uploads, page images, crops), .runtime/ (logs).
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
| Double-click `Setup.cmd` | `chmod +x *.sh && ./setup.sh` |

Setup creates `.env` (and asks for your Gemini key), creates the Python 3.12 environment inside
`backend/.venv`, installs the right PyTorch build for the machine (CUDA for NVIDIA, MPS for
Apple Silicon, CPU otherwise), installs the frontend, and **downloads all AI models once**
(about 2 GB) so the first document is not slowed down by downloads.

Do not copy `backend/.venv` between computers or operating systems — run setup on each machine.

## Every run

| | Windows | macOS / Linux |
|---|---|---|
| Start | double-click `Start.cmd` | `./start.sh` |
| Stop  | double-click `Stop.cmd`  | `./stop.sh` |

Start launches Docker (if needed), the databases, the API, the parsing worker and the frontend in
the background, then opens <http://127.0.0.1:5173/>. The badge at the top left shows the state:
**Loading AI models…** for the first ~30–60 seconds after start, then **Ready**, and
**Parsing page X of Y** while a document is processed. Stop keeps all documents and answers.

Logs are in `.runtime/logs/`.

## Using it

1. Upload a PDF. A progress bar shows the page being analysed.
2. As soon as it says ready you can ask questions; a "Reading chart values…" chip shows while chart
   numbers are still being decoded in the background. Click **Ready in …** above the page to see how
   long each stage took.
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

### Measured results (12 November 2026)

Windows 11 laptop · AMD Ryzen 5 4600H · 16 GB RAM · NVIDIA GTX 1650 (4 GB) · sample paper
(10 pages; 89 regions after removing duplicates: 5 tables, 4 charts, 2 equations, text and titles).
Full reports are in `benchmarks/`.

| Step | Before this overhaul | Now |
|---|---|---|
| Start-up until "Ready" (models loaded and warmed up) | models loaded inside the first parse | about 60 s, once per start |
| Upload → ready to answer questions | ~80 s (CPU-only PyTorch; GPU kernels untuned) | **6.8 s** median (6.8–10.9 s) |
| Upload → chart values decoded (background) | included above | ~18 s |
| Answer a question (median) | often minutes or no answer (Gemini model hung, 5× retry loops) | **5.8–11 s** across runs (fastest 2.2 s) |
| Tables with real cell structure | 0 of 5 (Table Transformer crashed silently) | 5 of 5 |
| Correct region among the 6 sent to Gemini (16-question test) | 8 of 16 (SigLIP + BM25) | 10 of 16 (SigLIP 2 + BM25) |

Where parse time goes now: layout 2.4–4.6 s, embeddings ~1.5–2 s, tables ~1–2 s, everything else
under 1 s; then chart decoding (DePlot, ~11 s for 4 charts) runs in the background while questions
can already be asked. Question time is almost entirely
the Gemini call, whose latency varies from 2 s to 30 s for the same prompt; hedged requests cut
most of the slow cases.

What changed to get there: CUDA build of PyTorch; cuDNN auto-tuning (layout 4.6 s → 0.15 s per
page); models preloaded and warmed up at start; fp16 disabled on GTX 16-series cards, where it
returns empty detections; Table Transformer processor fixed for transformers ≥ 4.52; Gemini model
switched from `gemini-3.8-flash` (no reply within 45 s) to `gemini-3.5-flash-lite` (2–3 s), with
hard timeouts, two tries per model, a fallback model and hedged requests; the extra Gemini rerank
call replaced by local hybrid SigLIP + BM25 scoring.

These numbers are for this laptop only. They say nothing about Apple Silicon; run the benchmark
on the Mac to get its own numbers.

## Models and versions (and why)

Every component was checked against the newest release on 12 November 2026 and measured on
DocLens's own sample. Rule: take the newest version unless it breaks something or measures slower.

| Job | Model / library | Why this one |
|---|---|---|
| Layout | DocLayout-YOLO DocStructBench **1280 (Jan 2025)** | newest weights; same regions as the 2024 weights with higher confidence and half the duplicate boxes, +65 ms/page |
| Tables | Table Transformer v1.1-all | newest release (the repo is now archived); runs on transformers 5 with a config repair |
| Charts | DePlot | no newer Google release; decoded in the background so it never delays "ready" |
| Search | **SigLIP 2** base + BM25 keywords | newest SigLIP; 10/16 vs 8/16 correct regions in the top 6 on our retrieval test |
| Answers | Gemini 3.5 Flash-Lite → 3.5 Flash → 3.6 Flash | fastest model first, the fast Flash next, the officially stable Flash as the last safety net |
| Runtime | torch 2.15 · transformers 5.22 · numpy 2.5 · PyMuPDF 2.1 | identical output to the old stack, chart decoding 12.2 s vs 18.1 s |

Changing the embedding model is safe: each embedding model has its own vector collection and the
worker re-parses documents indexed with an old model automatically on start.

Open decision: a dedicated text embedder (arctic-embed-s or bge-small) put the correct region in the
top 6 for 13/16 questions, against 10/16 for SigLIP 2 + BM25. It is a different model, so it waits for
the team's approval.

## Renaming the product or moving the folder

- **Name:** edit `brand.json` (`name`, `tagline`, `logo_letter`) and restart. The page title,
  sidebar, API title, launcher messages, benchmark reports and the Gemini prompt all read it.
  Nothing else needs changing; the launchers have generic names (`Start.cmd`, `./start.sh`).
- **Folder:** rename or move the project folder freely. Every path is computed relative to the
  project, and setup creates a relocatable Python environment. Stop the app before renaming.
- **Internal storage names stay fixed** (`docqa` database and queue, `regions` collection,
  `docqa_*` Docker volumes). They are invisible to users and deliberately not tied to the
  brand, so a rename can never orphan stored documents.

## Configuration (`.env`)

| Setting | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — | required for questions |
| `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash-lite` / `gemini-3.5-flash,gemini-3.6-flash` | fallbacks (comma list, tried in order) are used when a model is rate-limited, overloaded, times out or no longer exists |
| `GEMINI_THINKING` | `minimal` | model thinking level; higher is slower |
| `GEMINI_TIMEOUT_S` / `GEMINI_BUDGET_S` | `40` / `90` | wall-clock limit per attempt (request + backup) / per call; a slow model is abandoned for the next one |
| `GEMINI_HEDGE_AFTER_S` | `6` | send a duplicate request if no reply after this many seconds (0 = off) |
| `GEMINI_RERANK` | `false` | `true` adds one Gemini call per question to re-score retrieved regions |
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
| Answer says model unavailable / rate limited | Gemini quota reached or overloaded; wait a minute and ask again |
| Port 8000 or 5173 already in use | run Stop first, then Start |
| Apple MPS operator error | set `TORCH_DEVICE=cpu` in `.env` and restart |
| Never run `docker compose down -v` | it erases all stored documents and vectors |
