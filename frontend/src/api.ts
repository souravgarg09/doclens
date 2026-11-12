export type BBox = [number, number, number, number]; // x0,y0,x1,y1 normalized 0..1
export type Region = { id: number; page: number; type: string; bbox: BBox; content: string; structured: Record<string, unknown> | null; confidence: number };
export type Doc = { id: number; title: string; pages: number; status: string; error: string | null };
export type Citation = { region_id: number; page: number; type: string; bbox: BBox; score: number; snippet: string };
export type Verification = { status: string; summary: string; checks: { claim: string; value: number; matched: boolean; evidence?: string; recomputed?: number | null; note?: string }[] };
export type Answer = { id: number; question: string; answer: string; status: "answered" | "refused" | "flagged"; grounding: number; citations: Citation[]; verification: Verification | null; trace: { node: string; detail: string; ms: number }[] };
export type Health = { ok: boolean; gemini: boolean; model: string; fallback_model: string; device: string };

const BASE = import.meta.env.VITE_API_URL ?? "";

async function j<T>(res: Response): Promise<T> {
  const raw = await res.text();
  let data: Record<string, unknown> = {};
  try { data = raw ? JSON.parse(raw) as Record<string, unknown> : {}; } catch { /* handled below */ }
  if (!res.ok) {
    const detail = data.detail ?? data.error;
    throw new Error(typeof detail === "string" ? detail : raw || res.statusText);
  }
  return data as T;
}

export const api = {
  health: () => fetch(`${BASE}/api/health`).then((r) => j<Health>(r)),
  listDocs: () => fetch(`${BASE}/api/documents`).then((r) => j<{ documents: Doc[] }>(r)),
  getDoc: (id: number) => fetch(`${BASE}/api/documents/${id}`).then((r) => j<{ document: Doc; regions: Region[] }>(r)),
  upload: (file: File) => { const fd = new FormData(); fd.append("file", file); return fetch(`${BASE}/api/documents`, { method: "POST", body: fd }).then((r) => j<{ document: Doc }>(r)); },
  deleteDoc: (id: number) => fetch(`${BASE}/api/documents/${id}`, { method: "DELETE" }).then((r) => j<{ ok: boolean }>(r)),
  reparse: (id: number) => fetch(`${BASE}/api/documents/${id}/reparse`, { method: "POST" }).then((r) => j<{ ok: boolean }>(r)),
  queries: (id: number) => fetch(`${BASE}/api/documents/${id}/queries`).then((r) => j<{ queries: Answer[] }>(r)),
  deleteQuery: (id: number) => fetch(`${BASE}/api/queries/${id}`, { method: "DELETE" }).then((r) => j<{ ok: boolean }>(r)),
  clearQueries: (id: number) => fetch(`${BASE}/api/documents/${id}/queries`, { method: "DELETE" }).then((r) => j<{ ok: boolean }>(r)),
  ask: (document_id: number, question: string) => fetch(`${BASE}/api/ask`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ document_id, question }) }).then((r) => j<{ result: Answer }>(r)),
  pageUrl: (id: number, page: number) => `${BASE}/api/documents/${id}/pages/${page}.png`,
};
