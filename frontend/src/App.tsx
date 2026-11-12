import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Answer, type Citation, type Doc, type Health, type ParseStats, type Region } from "./api";
import { CitationOverlay, TYPE_COLORS } from "./components/CitationOverlay";

const BRAND = __BRAND__;

type BusyState = "upload" | "ask" | "document" | "history" | null;

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [doc, setDoc] = useState<Doc | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [page, setPage] = useState(1);
  const [showAll, setShowAll] = useState(true);
  const [highlighted, setHighlighted] = useState<number[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState<BusyState>(null);
  const [history, setHistory] = useState<Answer[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const questionRef = useRef<HTMLTextAreaElement>(null);
  const loadSequence = useRef(0);

  const refreshDocs = async () => {
    const response = await api.listDocs();
    setDocs(response.documents);
    return response.documents;
  };

  const loadDoc = async (id: number) => {
    const sequence = ++loadSequence.current;
    const [detail, queries] = await Promise.all([api.getDoc(id), api.queries(id)]);
    if (sequence !== loadSequence.current) return;
    setDoc(detail.document);
    setRegions(detail.regions);
    setHistory(queries.queries);
  };

  // Health includes the worker heartbeat (loading models / idle / parsing / offline).
  // Poll quickly while the worker is busy or starting, slowly otherwise.
  const workerState = health?.worker?.state;
  const backendDown = health === null;

  // (Re)load the document list whenever the backend becomes reachable, so a page opened
  // while the API is still starting fills in by itself instead of staying empty.
  useEffect(() => {
    if (backendDown) return;
    refreshDocs()
      .then((list) => {
        setErr(null);
        setSel((current) => current ?? list[0]?.id ?? null);
      })
      .catch((ex: Error) => setErr(ex.message));
  }, [backendDown]);
  useEffect(() => {
    let cancelled = false;
    const load = () => api.health().then((h) => { if (!cancelled) setHealth(h); }).catch(() => { if (!cancelled) setHealth(null); });
    void load();
    const fast = backendDown || workerState === "loading" || workerState === "parsing" || workerState === "offline";
    const timer = window.setInterval(load, fast ? 2000 : 10000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [workerState, backendDown]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTyping = target?.matches("input, textarea, [contenteditable='true']");
      if (event.key === "/" && !isTyping) {
        event.preventDefault();
        questionRef.current?.focus();
      }
      if (event.key === "Escape" && !isTyping) {
        setHighlighted([]);
        setSelectedRegion(null);
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, []);

  useEffect(() => {
    if (sel == null) return;
    setPage(1);
    setHighlighted([]);
    setSelectedRegion(null);
    loadDoc(sel).catch((ex: Error) => setErr(ex.message));
  }, [sel]);

  useEffect(() => {
    if (!docs.some((item) => item.status === "queued" || item.status === "parsing")) return;
    const timer = window.setInterval(async () => {
      try {
        const list = await refreshDocs();
        const latest = list.find((item) => item.id === sel);
        if (latest && latest.status !== doc?.status) await loadDoc(latest.id);
        else if (latest) setDoc((current) => (current && current.id === latest.id ? { ...current, ...latest } : current));
      } catch (ex) {
        setErr((ex as Error).message);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [docs, sel, doc?.status]);

  const upload = async (event: React.FormEvent) => {
    event.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setBusy("upload");
    setErr(null);
    try {
      const response = await api.upload(file);
      await refreshDocs();
      setSel(response.document.id);
      setFileName("");
      if (fileRef.current) fileRef.current.value = "";
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const ask = async () => {
    if (!q.trim() || !sel) return;
    setBusy("ask");
    setErr(null);
    try {
      const response = await api.ask(sel, q.trim());
      setHistory((items) => [response.result, ...items]);
      focus(response.result);
      setQ("");
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const deleteDocument = async (item: Doc) => {
    if (!window.confirm(`Delete “${item.title}” and all of its parsed data?`)) return;
    setBusy("document");
    setErr(null);
    try {
      await api.deleteDoc(item.id);
      const remaining = await refreshDocs();
      if (item.id === sel) {
        const next = remaining[0]?.id ?? null;
        setSel(next);
        if (next == null) {
          setDoc(null);
          setRegions([]);
          setHistory([]);
        }
      }
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const reparseDocument = async (item: Doc) => {
    setBusy("document");
    setErr(null);
    try {
      await api.reparse(item.id);
      await refreshDocs();
      await loadDoc(item.id);
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const deleteAnswer = async (id: number) => {
    setBusy("history");
    setErr(null);
    try {
      await api.deleteQuery(id);
      setHistory((items) => items.filter((item) => item.id !== id));
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const clearHistory = async () => {
    if (!sel || !history.length || !window.confirm("Clear the complete answer history for this document?")) return;
    setBusy("history");
    setErr(null);
    try {
      await api.clearQueries(sel);
      setHistory([]);
    } catch (ex) {
      setErr((ex as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const focus = (answer: Answer) => {
    setHighlighted(answer.citations.map((citation) => citation.region_id));
    if (answer.citations[0]) setPage(answer.citations[0].page);
  };
  const goto = (citation: Citation) => {
    setPage(citation.page);
    setHighlighted([citation.region_id]);
  };
  const stats = useMemo(() => regions.reduce<Record<string, number>>((result, region) => {
    result[region.type] = (result[region.type] ?? 0) + 1;
    return result;
  }, {}), [regions]);

  return (
    <div className="layout">
      <aside className="panel sidebar">
        <div className="brand"><span className="logo">{BRAND.logo_letter}</span><div><h1>{BRAND.name}</h1><small>{BRAND.tagline}</small></div></div>
        <HealthBadge health={health} />

        <form onSubmit={upload} className="upload">
          <div><b>Add a PDF</b><small>Up to 25 MB · born-digital works best</small></div>
          <label className="file-picker"><input ref={fileRef} type="file" accept="application/pdf" onChange={(event) => setFileName(event.target.files?.[0]?.name ?? "")} /><span>{fileName || "Choose PDF"}</span></label>
          <button className="primary" disabled={busy !== null || !fileName}>{busy === "upload" ? "Uploading…" : "Parse document"}</button>
        </form>

        <div className="section-heading"><span>Documents</span><small>{docs.length}</small></div>
        <ul className="docs">
          {docs.map((item) => (
            <li key={item.id} className={item.id === sel ? "active" : ""} onClick={() => setSel(item.id)}>
              <div className="doc-copy"><b title={item.title}>{item.title}</b><small><i className={`dot ${item.status}`} /> {item.status === "parsing" && item.progress?.pages ? `parsing ${progressPercent(item)}%` : item.status}{item.pages ? ` · ${item.pages} pages` : ""}{item.status === "ready" && item.parse_stats ? ` · ${formatSeconds(item.parse_stats.total_seconds)}` : ""}</small></div>
              <div className="doc-actions">
                {item.status === "failed" && <button className="icon-button" title="Retry parsing" aria-label={`Retry parsing ${item.title}`} disabled={busy !== null} onClick={(event) => { event.stopPropagation(); void reparseDocument(item); }}>↻</button>}
                <button className="icon-button danger" title="Delete document" aria-label={`Delete ${item.title}`} disabled={busy !== null || item.status === "parsing"} onClick={(event) => { event.stopPropagation(); void deleteDocument(item); }}>×</button>
              </div>
            </li>
          ))}
          {!docs.length && <li className="list-empty">No documents yet</li>}
        </ul>
        <small className="sidebar-note">Start and stop with <b>Start.cmd</b> / <b>Stop.cmd</b> (Windows) or <b>./start.sh</b> / <b>./stop.sh</b>. Logs are in <code>.runtime/logs</code>.</small>
      </aside>

      <main>
        {doc ? (
          <>
            <div className="toolbar">
              <div className="document-title"><small>DOCUMENT</small><h2 title={doc.title}>{doc.title}</h2></div>
              <div className="page-controls"><button aria-label="Previous page" disabled={page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>‹</button><code>{page} / {doc.pages || "?"}</code><button aria-label="Next page" disabled={page >= doc.pages} onClick={() => setPage((value) => Math.min(doc.pages || value + 1, value + 1))}>›</button></div>
              <label className="toggle"><input type="checkbox" checked={showAll} onChange={(event) => setShowAll(event.target.checked)} /> show regions</label>
              <button className="secondary" disabled={!highlighted.length} onClick={() => { setHighlighted([]); setSelectedRegion(null); }}>Clear highlight</button>
            </div>
            <div className="stats">{Object.entries(stats).map(([type, count]) => <span key={type} style={{ borderColor: TYPE_COLORS[type], color: TYPE_COLORS[type] }}>{type} {count}</span>)}{doc.status !== "ready" && <em>{doc.status === "failed" ? doc.error : "Parsing document…"}</em>}{doc.status === "ready" && doc.parse_stats && <ParseReport stats={doc.parse_stats} />}</div>
            {doc.status === "ready" ? <CitationOverlay src={api.pageUrl(doc.id, page)} regions={regions.filter((region) => region.page === page)} highlighted={highlighted} showAll={showAll} onSelect={(region) => { setSelectedRegion(region); setHighlighted([region.id]); }} /> : (
              <div className={`empty-state ${doc.status}`}><span className="empty-icon">{doc.status === "failed" ? "!" : "↻"}</span><h3>{doc.status === "failed" ? "Parsing failed" : "Building the region index"}</h3><p>{doc.status === "failed" ? doc.error : parsingMessage(doc, health)}</p>{doc.status !== "failed" && <div className="parse-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progressPercent(doc)}><span style={{ width: `${Math.max(4, progressPercent(doc))}%` }} /></div>}{doc.status === "failed" && <button className="primary" disabled={busy !== null} onClick={() => void reparseDocument(doc)}>Retry parsing</button>}</div>
            )}
            {selectedRegion && <div className="card region-detail"><div className="row"><b style={{ color: TYPE_COLORS[selectedRegion.type] }}>R{selectedRegion.id} · {selectedRegion.type}</b><small>page {selectedRegion.page}</small><button className="icon-button" aria-label="Close region details" onClick={() => setSelectedRegion(null)}>×</button></div><pre>{selectedRegion.content || "No extracted text for this visual region."}</pre>{selectedRegion.structured && <details><summary>Structured extraction</summary><pre className="dim">{JSON.stringify(selectedRegion.structured, null, 2).slice(0, 3000)}</pre></details>}</div>}
          </>
        ) : <div className="empty-state welcome"><span className="empty-icon">{BRAND.logo_letter}</span><h2>Explore evidence, not guesses</h2><p>Upload a PDF to detect its visual regions, ask questions, and jump directly to cited evidence.</p></div>}
      </main>

      <aside className="panel answer-panel">
        <div className="ask-box">
          <div><b>Ask this document</b><small>Answers are restricted to retrieved page regions.</small></div>
          <textarea ref={questionRef} value={q} onChange={(event) => setQ(event.target.value)} rows={4} maxLength={2000} placeholder="Ask about a finding, table, chart, or numerical result…" onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(); } }} />
          <div className="ask-footer"><small>{q.length}/2000 · <kbd>/</kbd> focus · Enter run · Shift+Enter new line</small><button className="primary" disabled={busy !== null || !q.trim() || !doc || doc.status !== "ready"} onClick={() => void ask()}>{busy === "ask" ? "Reasoning…" : "Run agent"}</button></div>
          {busy === "ask" && <div className="progress" role="status" aria-live="polite"><span /><small>Retrieve → rerank → answer → verify</small></div>}
        </div>

        {err && <div className="error-banner" role="alert"><div><b>Something went wrong</b><p>{err}</p></div><button className="icon-button" aria-label="Dismiss error" onClick={() => setErr(null)}>×</button></div>}
        <div className="history-heading"><div><b>Answer history</b><small>{history.length ? `${history.length} saved` : "No questions yet"}</small></div>{history.length > 0 && <button className="link danger-text" disabled={busy !== null} onClick={() => void clearHistory()}>Clear history</button>}</div>
        <div className="answers">{history.map((answer) => <AnswerCard key={answer.id} a={answer} onFocus={() => focus(answer)} onCite={goto} onDelete={() => void deleteAnswer(answer.id)} disabled={busy !== null} />)}{!history.length && <div className="answer-empty"><span>⌁</span><p>Your grounded answers and verification results will appear here.</p></div>}</div>
      </aside>
    </div>
  );
}

function AnswerCard({ a, onFocus, onCite, onDelete, disabled }: { a: Answer; onFocus: () => void; onCite: (citation: Citation) => void; onDelete: () => void; disabled: boolean }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const renderedAnswer = normalizeCitationGroups(a.answer);
  const parts = renderedAnswer.split(/(\[R\d+\])/g);
  const traceDuration = a.trace.reduce((total, step) => total + step.ms, 0);
  const copyAnswer = async () => {
    await navigator.clipboard.writeText(`${a.question}\n\n${renderedAnswer}`);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };
  return (
    <article className={`card answer-card ${a.status}`}>
      <div className="row answer-title"><b>{a.question}</b><span className={`badge ${a.status}`}>{answerLabel(a)}</span><button className="mini-action" title="Copy question and answer" aria-label="Copy question and answer" onClick={() => void copyAnswer()}>{copied ? "Copied" : "Copy"}</button><button className="icon-button danger" title="Delete answer" aria-label="Delete answer" disabled={disabled} onClick={onDelete}>×</button></div>
      <p className="answer-copy">{parts.map((part, index) => { const match = part.match(/^\[R(\d+)\]$/); if (!match) return <span key={index}>{part}</span>; const citation = a.citations.find((item) => item.region_id === +match[1]); return <button key={index} className="cite" style={{ background: TYPE_COLORS[citation?.type ?? "text"] }} onClick={() => citation && onCite(citation)}>R{match[1]}</button>; })}</p>
      <div className="row metadata"><span>Grounding <b>{a.grounding.toFixed(2)}</b></span>{a.citations.map((citation) => <button key={citation.region_id} className="chip" style={{ borderColor: TYPE_COLORS[citation.type], color: TYPE_COLORS[citation.type] }} onClick={() => onCite(citation)}>R{citation.region_id} · {citation.type} · p{citation.page}</button>)}{a.citations.length > 0 && <button className="link" onClick={onFocus}>Highlight all</button>}</div>
      {a.verification && <div className="verify"><small>NUMERIC CONSISTENCY · {a.verification.status}</small><p>{a.verification.summary}</p><ul>{a.verification.checks.slice(0, 8).map((check, index) => <li key={index}><span className={check.matched ? "ok" : "bad"}>{check.matched ? "✓" : "✕"}</span> <code>{check.value}</code> <small>{check.evidence ?? check.note}{check.recomputed != null && !check.matched ? ` (recomputed ${check.recomputed})` : ""}</small></li>)}</ul></div>}
      {!!a.trace.length && <button className="link trace-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? "Hide" : "Show"} processing trace <span>· {formatDuration(traceDuration)}</span></button>}
      {open && <ol className="trace">{a.trace.map((step, index) => <li key={index}><div><code>{traceLabel(step.node)}</code><small>{formatDuration(step.ms)}</small></div><span title={step.detail}>{step.detail}</span></li>)}</ol>}
    </article>
  );
}

function traceLabel(node: string) {
  const labels: Record<string, string> = {
    parse: "Document ready",
    retrieve: "Evidence retrieved",
    rerank: "Evidence reranked",
    expand: "Context expanded",
    answer: "Answer generated",
    verify: "Numbers verified",
    guardrail: "Safety checked",
  };
  return labels[node.toLowerCase()] ?? node.replace(/_/g, " ");
}

function formatDuration(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function normalizeCitationGroups(text: string) {
  return text.replace(/\[\s*R\d+(?:\s*[,;]\s*R?\d+)+\s*\]/gi, (group) => {
    const ids = Array.from(group.matchAll(/(?:R\s*)?(\d+)/gi), (match) => match[1]);
    return ids.map((id) => `[R${id}]`).join(" ");
  });
}

function answerLabel(answer: Answer) {
  if (answer.status === "refused") return "Refused";
  if (answer.status === "flagged") return answer.verification?.status === "citation_gap" ? "Citation gap" : "Needs review";
  if (answer.verification?.status === "consistent") return "Verified";
  return "Answered";
}

function HealthBadge({ health }: { health: Health | null }) {
  const worker = health?.worker;
  let tone = "warn";
  let title = "Backend offline";
  let detail = "Run Start.cmd (Windows) or ./start.sh (macOS/Linux)";
  if (health) {
    detail = `${health.device_detail ?? health.device.toUpperCase()} · ${health.model}${health.fallback_model ? ` → ${health.fallback_model}` : ""}`;
    if (!health.gemini) title = "Gemini key missing";
    else if (!worker || worker.state === "offline" || worker.state === "unreachable") title = "Parsing worker offline";
    else if (worker.state === "loading") title = "Loading AI models…";
    else if (worker.state === "parsing") { tone = "ok"; title = worker.pages ? `Parsing page ${worker.page} of ${worker.pages}` : "Parsing a document"; }
    else { tone = "ok"; title = "Ready"; }
    if (worker?.state === "loading" && worker.detail) detail = `${worker.detail} · ${detail}`;
  }
  return (
    <div className={`health ${tone}`} role="status">
      <span className="health-dot" />
      <div><b>{title}</b><small>{detail}</small></div>
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  render: "Render pages", index_setup: "Index setup", layout: "Layout detection", text: "Text extraction", tables: "Table structure",
  charts: "Chart decoding", equations: "Equation OCR", embeddings: "Embeddings", vector_upsert: "Vector index",
};

function ParseReport({ stats }: { stats: ParseStats }) {
  const entries = Object.entries(stats.stages).filter(([, seconds]) => seconds >= 0.005).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, seconds]) => seconds), 0.001);
  return (
    <details className="parse-report">
      <summary>Parsed in <b>{formatSeconds(stats.total_seconds)}</b> · {formatSeconds(stats.seconds_per_page)}/page · {stats.device}</summary>
      <ul>{entries.map(([stage, seconds]) => (
        <li key={stage}><span>{STAGE_LABELS[stage] ?? stage}{(stats.counts[stage] ?? 0) > 1 ? ` ×${stats.counts[stage]}` : ""}</span><i style={{ width: `${(seconds / max) * 100}%` }} /><code>{formatSeconds(seconds)}</code></li>
      ))}</ul>
    </details>
  );
}

function progressPercent(item: Doc) {
  const progress = item.progress;
  if (!progress || !progress.pages) return item.status === "ready" ? 100 : 0;
  if (progress.stage === "indexing") return 97;
  return Math.min(95, Math.round(((progress.page - 1) / progress.pages) * 95));
}

function parsingMessage(item: Doc, health: Health | null) {
  if (health?.worker?.state === "loading") return "The worker is still loading its AI models; parsing starts as soon as they are ready.";
  if (health && (!health.worker || health.worker.state === "offline")) return `The parsing worker is not running. Restart ${BRAND.name} to process this document.`;
  const progress = item.progress;
  if (!progress || item.status === "queued") return "Waiting for the parsing worker…";
  if (progress.stage === "rendering") return "Rendering pages…";
  if (progress.stage === "indexing") return "Writing the vector index…";
  return `Detecting text, tables, charts and equations · page ${progress.page} of ${progress.pages}`;
}

function formatSeconds(seconds: number) {
  if (seconds >= 60) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return seconds >= 10 ? `${seconds.toFixed(0)}s` : `${seconds.toFixed(seconds >= 1 ? 1 : 2)}s`;
}
