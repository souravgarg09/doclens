# DocLens

**Grounded answers from visual PDFs.** DocLens answers questions about research papers and
reports using only what is in the document, and shows the exact evidence: every answer cites the
page regions it came from, highlighted on the page, and every number in the answer is checked
against the values extracted from the PDF.

## Features

- **Layout-aware parsing.** Each page is split into typed regions (text, titles, tables, charts,
  figures, equations) with their position on the page.
- **Tables and charts as data.** Tables become structured cells; bar and line charts are decoded
  into data series, so answers can quote and compute with real values.
- **Hidden PDF links.** Citation numbers that jump to the reference list, "see Table 2"
  cross-references, and words linked to URLs or DOIs are read from the PDF. Ask "Which paper is
  reference 11?" or "Where is the code?", or click a link on the page to follow it. For PDFs
  without links, numbered citations such as "(11)" are matched to the reference list.
- **Clickable evidence.** Answers cite regions like `[R81]`; clicking one jumps to the page and
  highlights the region.
- **Numeric verification.** Numbers in an answer are matched against extracted cells, chart points
  and simple arithmetic on them; an unsupported claim triggers one automatic repair pass.
- **Refuses rather than guesses.** Questions the document cannot answer are declined.
- **Private by design.** Parsing runs entirely on your machine; only the question and the
  selected regions are sent to Gemini.

## How it works

```text
PDF upload ─▶ FastAPI stores it and queues a Redis job
            ─▶ parsing worker (local):
                 PyMuPDF render ─▶ DocLayout-YOLO regions
                   ├─ tables   ─▶ Table Transformer ─▶ cells
                   ├─ charts   ─▶ DePlot            ─▶ data series (background, after "ready")
                   └─ text     ─▶ PDF text layer
                 PDF link layer ─▶ citations, cross-references, URLs/DOIs, reference list
                 text + visual embeddings ─▶ Qdrant · regions and links ─▶ PostgreSQL
Question   ─▶ LangGraph agent:
                 hybrid retrieval (text vectors + BM25 keywords + visual vectors)
               ─▶ rerank ─▶ expand to named tables/figures ─▶ follow PDF links
               ─▶ grounding guardrail ─▶ Gemini answer with region citations
               ─▶ numeric verifier ─▶ repair pass if needed
```

| Component | Technology |
|---|---|
| API and agent | FastAPI, LangGraph |
| Parsing worker | PyTorch, PyMuPDF, Redis queue |
| Layout | DocLayout-YOLO (DocStructBench) |
| Tables | Microsoft Table Transformer v1.1 |
| Charts | Google DePlot |
| Text search | Snowflake arctic-embed-s + BM25 |
| Visual search | Google SigLIP 2 |
| Storage | PostgreSQL, Qdrant |
| Answers | Google Gemini (Flash-Lite, with automatic fallback) |
| Frontend | React, Vite, TypeScript |

## Requirements

- Docker Desktop (running)
- Node.js 20.19 or newer
- [uv](https://docs.astral.sh/uv/) (setup installs it on macOS/Linux; on Windows run `winget install astral-sh.uv`)
- A Gemini API key from <https://aistudio.google.com/apikey>
- 16 GB RAM recommended and about 6 GB of disk for packages and models
- An NVIDIA GPU (Windows/Linux) or Apple Silicon (macOS) is used automatically; CPU also works

## Quick start

**1. Set up (once per computer)**

| Windows | macOS / Linux |
|---|---|
| Double-click `Setup.cmd` | `chmod +x *.sh && ./setup.sh` |

Setup creates `.env` and asks for your Gemini key, creates the Python environment, installs the
right PyTorch build for your hardware (CUDA, Apple MPS or CPU), installs the frontend and
downloads the models (about 2 GB).

**2. Start and stop**

| | Windows | macOS / Linux |
|---|---|---|
| Start | double-click `Start.cmd` | `./start.sh` |
| Stop | double-click `Stop.cmd` | `./stop.sh` |

Start launches the databases, API, parsing worker and frontend, then opens
<http://127.0.0.1:5173/>. The status badge shows **Loading AI models…** for the first minute,
then **Ready**. Stopping keeps all documents and answers. Logs are in `.runtime/logs/`.

## Usage

1. **Upload a PDF.** A progress bar follows the parse; you can ask questions as soon as it is
   ready, while chart values are still being decoded in the background.
2. **Ask a question.** The answer cites regions such as `R81`; click a citation to see it on the
   page.
3. **Follow links.** Underlines mark the PDF's own links (yellow: citations, blue: web addresses).
   Click one to jump to its target or open the address; select a region to list its links.
4. **Check the verification.** `consistent` means every number matched an extracted value;
   `unverifiable`, `contradiction` and `citation_gap` deserve a closer look. **Show processing
   trace** lists every agent step with its timing.

## Configuration

Settings live in `.env` (created from `.env.example`).

| Setting | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — | required for questions |
| `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL` | `gemini-3.1-flash-lite` / `gemini-3.5-flash-lite,gemini-3.6-flash` | primary model and fallbacks, tried in order when a model is rate-limited, overloaded, slow or retired |
| `GEMINI_THINKING` | `minimal` | model thinking level; higher is slower |
| `GEMINI_TIMEOUT_S` / `GEMINI_BUDGET_S` | `40` / `90` | time limit per attempt / per question |
| `GEMINI_HEDGE_AFTER_S` | `0` (off) | send a backup request after this many seconds (useful on a paid tier) |
| `GEMINI_RERANK` | `false` | add a Gemini call to re-score retrieved regions |
| `TORCH_DEVICE` | `auto` | CUDA → Apple MPS → CPU; force with `cuda`, `mps` or `cpu` |
| `TEXT_EMBED_MODEL` | `Snowflake/snowflake-arctic-embed-s` | text embedder; empty = visual vectors + BM25 only |
| `LEXICAL_WEIGHT` / `VISUAL_WEIGHT` | `1.0` / `0.25` | weight of keyword and visual rankings in hybrid retrieval |
| `TOP_K_RETRIEVE` / `TOP_K_CONTEXT` | `20` / `6` | regions retrieved / regions sent to the model |
| `GROUNDING_THRESHOLD` | `0.35` | below this the agent declines to answer |
| `RENDER_DPI` | `144` | page render resolution |

Changing an embedding model is safe: documents are re-indexed automatically on the next start.

**Renaming.** The product name lives only in `brand.json` (name, tagline, logo letter); edit it
and restart. The project folder can be renamed or moved freely.

## Project layout

```text
DocLens/
├── Setup.cmd · Start.cmd · Stop.cmd   Windows launchers
├── setup.sh · start.sh · stop.sh      macOS / Linux launchers
├── brand.json                         product name and tagline
├── docker-compose.yml                 PostgreSQL, Redis, Qdrant (bound to localhost)
├── .env.example                       settings template
├── backend/
│   ├── app/          API, database models, agent, retrieval and verification
│   ├── workers/      parsing worker and the layout, table, chart and link parsers
│   ├── scripts/      model download and benchmark tools
│   └── tests/        unit tests
├── frontend/         React interface
├── scripts/windows/  PowerShell scripts behind the launchers
└── samples/          place demo PDFs here (see samples/README.md)
```

## Development

Run the tests from `backend/` (use `.venv/bin/python` on macOS/Linux):

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Type-check and build the frontend from `frontend/`:

```bash
npx tsc --noEmit -p . && npm run build
```

Measure parse and answer times on your machine with DocLens running (from `backend/`):

```bash
.\.venv\Scripts\python.exe -m scripts.benchmark --pdf ..\samples\your-paper.pdf -q "Your question?"
```

The report (`benchmarks/<date>_<computer>.md`) lists per-stage parse timings and per-step
question timings.

## Limitations

- Built for born-digital English PDFs. Scanned PDFs without a text layer are not supported.
- Chart decoding covers simple bar and line charts; other figures are passed to Gemini as images.
- Author-year citations ("Smith et al., 2019") are resolved only when the PDF links them;
  numbered citations are always matched.
- Equation OCR (`pix2tex`) is optional and not installed by default.
- A `consistent` verification means the numbers match extracted values; it does not prove every
  aspect of the answer (for example, the right series or unit).
- Gemini's free tier has per-minute and per-day request limits.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Badge says **Parsing worker offline** | restart DocLens and check `.runtime/logs/worker*.log` |
| Badge stays on **Loading AI models…** | models are still downloading; run setup again while online |
| Document stuck in `queued` | the worker is not running (see above) |
| Answer says the model is unavailable | Gemini quota reached or overloaded; wait a minute and retry |
| Port 8000 or 5173 already in use | run Stop, then Start |
| Apple MPS operator error | set `TORCH_DEVICE=cpu` in `.env` and restart |
| Data disappeared | never run `docker compose down -v`; it erases stored documents |

## Author

Sourav Garg
