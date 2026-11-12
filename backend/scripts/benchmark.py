"""End-to-end benchmark: parse a PDF, ask questions, record where time goes.

Start the application first (Start.cmd or ./start.sh), then from backend/:

    python -m scripts.benchmark                      # sample PDF, 3 runs
    python -m scripts.benchmark --runs 5 --pdf ../samples/other.pdf -q "Question 1?" -q "Question 2?"

It records, per run:
  * upload → ready wall-clock time, and the worker's own per-stage timings
    (render, layout, tables, charts, embeddings, vector upsert)
  * for every question: total latency and the time of each agent step
    (retrieve, rerank, expand, answer, verify), answer status, and any
    Gemini quota/availability errors
  * the machine: OS, CPU, RAM, PyTorch device

Results are written to benchmarks/<timestamp>_<host>.json and .md so runs on
different laptops (for example Windows vs an M-series Mac) can be compared fairly.
The first run after starting the worker is labelled; model loading happens at
worker start-up and is reported separately from parsing.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "samples" / "muscle-hypertrophy-meta-analysis.pdf"
DEFAULT_QUESTIONS = [
    "How many studies were included in the meta-analysis?",
    "What was the effect size difference between multiple sets and single sets for hypertrophy?",
    "What does Table 1 report?",
]
STAGE_ORDER = ["render", "index_setup", "layout", "text", "tables", "charts", "equations", "embeddings", "vector_upsert"]


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {"host": socket.gethostname(), "os": f"{platform.system()} {platform.release()}",
                            "arch": platform.machine(), "python": platform.python_version(),
                            "cpu": platform.processor() or "unknown"}
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["cpu_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return info


def wait_for_worker(client: httpx.Client, timeout: float) -> tuple[dict[str, Any], float]:
    """Wait until the worker has loaded its models; return (health, seconds waited)."""
    start = time.perf_counter()
    while True:
        health = client.get("/api/health").raise_for_status().json()
        state = health.get("worker", {}).get("state")
        if state in ("idle", "parsing"):
            return health, time.perf_counter() - start
        if time.perf_counter() - start > timeout:
            raise SystemExit(f"worker not ready after {timeout:.0f}s (state={state}). Is the worker running?")
        time.sleep(1)


def parse_once(client: httpx.Client, pdf: Path, timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    with pdf.open("rb") as fh:
        doc = client.post("/api/documents", files={"file": (pdf.name, fh, "application/pdf")}).raise_for_status().json()["document"]
    ready_s = None
    while True:
        detail = client.get(f"/api/documents/{doc['id']}").raise_for_status().json()["document"]
        if detail["status"] == "ready" and ready_s is None:
            ready_s = round(time.perf_counter() - start, 2)  # questions can be asked from here
        if detail["status"] == "failed" or (detail["status"] == "ready" and not detail.get("progress")):
            break  # failed, or ready with the background chart phase finished
        if time.perf_counter() - start > timeout:
            raise SystemExit(f"document {doc['id']} not finished after {timeout:.0f}s")
        time.sleep(0.5)
    return {"document_id": doc["id"], "status": detail["status"], "error": detail.get("error"),
            "upload_to_ready_s": ready_s, "upload_to_charts_done_s": round(time.perf_counter() - start, 2),
            "parse_stats": detail.get("parse_stats")}


def ask(client: httpx.Client, document_id: int, question: str) -> dict[str, Any]:
    start = time.perf_counter()
    response = client.post("/api/ask", json={"document_id": document_id, "question": question}, timeout=300)
    elapsed = round(time.perf_counter() - start, 2)
    if response.status_code != 200:
        return {"question": question, "http_status": response.status_code, "seconds": elapsed, "error": response.text[:300]}
    result = response.json()["result"]
    steps: dict[str, int] = {}
    for step in result.get("trace", []):
        steps[step["node"]] = steps.get(step["node"], 0) + int(step["ms"])
    trace_text = " ".join(step.get("detail", "") for step in result.get("trace", []))
    return {"question": question, "seconds": elapsed, "status": result["status"],
            "verification": (result.get("verification") or {}).get("status"),
            "citations": len(result.get("citations", [])), "steps_ms": steps,
            "quota_or_unavailable": any(w in trace_text for w in ("429", "rate limited", "UNAVAILABLE", "unavailable")),
            "answer": result["answer"][:400]}


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [r["parse"]["upload_to_ready_s"] for r in runs if r["parse"]["status"] == "ready"]
    done = [r["parse"]["upload_to_charts_done_s"] for r in runs if r["parse"]["status"] == "ready"]
    q_times = [a["seconds"] for r in runs for a in r["answers"] if "status" in a]
    return {"parse_median_s": round(statistics.median(ready), 2) if ready else None,
            "charts_done_median_s": round(statistics.median(done), 2) if done else None,
            "parse_min_s": min(ready) if ready else None, "parse_max_s": max(ready) if ready else None,
            "question_median_s": round(statistics.median(q_times), 2) if q_times else None,
            "answered": sum(1 for r in runs for a in r["answers"] if a.get("status") == "answered"),
            "questions": sum(len(r["answers"]) for r in runs),
            "quota_errors": sum(1 for r in runs for a in r["answers"] if a.get("quota_or_unavailable"))}


def to_markdown(report: dict[str, Any]) -> str:
    m, h, s = report["machine"], report["health"], report["summary"]
    lines = [f"# {h.get('app', {}).get('name', 'DocLens')} benchmark — {report['started_at']}", "",
             f"- Machine: {m['host']} · {m['os']} · {m['arch']} · {m.get('cpu')} · RAM {m.get('ram_gb', '?')} GB",
             f"- Device: {h.get('device_detail', h.get('device'))} · Gemini {h.get('model')} (fallback {h.get('fallback_model')})"
             f" · Gemini rerank {'on' if h.get('gemini_rerank') else 'off'}",
             f"- PDF: {report['pdf']} · runs: {len(report['runs'])} · waited {report['worker_wait_s']:.1f}s for the worker's models",
             "", "## Summary", "",
             f"- Upload → ready (questions possible): median **{s['parse_median_s']} s** (min {s['parse_min_s']}, max {s['parse_max_s']})",
             f"- Upload → chart values decoded (background phase): median {s.get('charts_done_median_s')} s",
             f"- Question latency: median **{s['question_median_s']} s**",
             f"- Answered {s['answered']}/{s['questions']} · Gemini quota/unavailable errors: {s['quota_errors']}",
             "", "## Parse stages (seconds)", "",
             "| run | ready | charts done | " + " | ".join(STAGE_ORDER) + " | pages | regions |",
             "|---|---|---|" + "---|" * len(STAGE_ORDER) + "---|---|"]
    for i, r in enumerate(report["runs"], 1):
        ps = r["parse"].get("parse_stats") or {}
        st = ps.get("stages", {})
        lines.append(f"| {i} | {r['parse']['upload_to_ready_s']} | {r['parse'].get('upload_to_charts_done_s', '–')} | "
                     + " | ".join(str(st.get(k, "–")) for k in STAGE_ORDER)
                     + f" | {ps.get('pages', '–')} | {ps.get('regions', '–')} |")
    lines += ["", "## Questions", "", "| run | question | seconds | status | verify | retrieve | rerank | answer | verify ms |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(report["runs"], 1):
        for a in r["answers"]:
            st = a.get("steps_ms", {})
            lines.append(f"| {i} | {a['question'][:60]} | {a['seconds']} | {a.get('status', a.get('http_status'))} | "
                         f"{a.get('verification') or '–'} | {st.get('retrieve', '–')} | {st.get('rerank', '–')} | "
                         f"{st.get('answer', '–')} | {st.get('verify', '–')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("-q", "--question", action="append", dest="questions")
    ap.add_argument("--no-questions", action="store_true", help="only measure parsing (no Gemini calls)")
    ap.add_argument("--keep", action="store_true", help="keep the benchmark documents instead of deleting them")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks")
    args = ap.parse_args()
    questions = [] if args.no_questions else (args.questions or DEFAULT_QUESTIONS)

    with httpx.Client(base_url=args.api, timeout=60) as client:
        health, waited = wait_for_worker(client, args.timeout)
        print(f"Device: {health.get('device_detail')} · waited {waited:.1f}s for the worker")
        report: dict[str, Any] = {"started_at": datetime.now().isoformat(timespec="seconds"), "machine": machine_info(),
                                  "health": health, "pdf": str(args.pdf.name), "worker_wait_s": round(waited, 2), "runs": []}
        for run in range(1, args.runs + 1):
            parse = parse_once(client, args.pdf, args.timeout)
            stats = parse.get("parse_stats") or {}
            print(f"run {run}: {parse['status']} · ready (can ask) after {parse['upload_to_ready_s']}s · "
                  f"charts done after {parse['upload_to_charts_done_s']}s · stages {stats.get('stages')}")
            answers = []
            if parse["status"] == "ready":
                for q in questions:
                    a = ask(client, parse["document_id"], q)
                    print(f"   {a['seconds']:>6}s  {a.get('status', a.get('http_status'))}  {q}")
                    answers.append(a)
            report["runs"].append({"run": run, "parse": parse, "answers": answers})
            if not args.keep:
                client.delete(f"/api/documents/{parse['document_id']}")
        report["summary"] = summarise(report["runs"])

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{socket.gethostname()}"
    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / f"{stem}.md").write_text(to_markdown(report), encoding="utf-8")
    print(f"\nSummary: {report['summary']}\nSaved {args.out / (stem + '.md')}")


if __name__ == "__main__":
    main()
