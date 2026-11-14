import { useState } from "react";
import type { PdfLink, Region } from "../api";

export const LINK_COLORS: Record<string, string> = { citation: "#fbbf24", url: "#38bdf8", doi: "#38bdf8", email: "#38bdf8", table: "#f59e0b", figure: "#22d3ee", equation: "#f472b6", section: "#a78bfa", footnote: "#94a3b8" };

/** What a link points to, for its tooltip and accessible name. */
export function linkSummary(link: PdfLink): string {
  if (link.uri) return `${link.kind.toUpperCase()}: ${link.uri}`;
  const where = link.target_page ? `page ${link.target_page}` : "";
  const what = link.kind === "citation" && link.label ? `Reference [${link.label}]` : link.kind;
  return `${what}${where ? ` · ${where}` : ""}${link.target_text ? ` · ${link.target_text.slice(0, 220)}` : ""}${link.inferred ? " (matched from the reference list)" : ""}`;
}

export const TYPE_COLORS: Record<string, string> = { text: "#64748b", title: "#a78bfa", table: "#f59e0b", figure: "#22d3ee", chart: "#34d399", equation: "#f472b6" };

/**
 * Pixel-level citation overlay: draws normalized bboxes on top of the rendered
 * page image. Because bboxes are in 0..1 page coordinates, they stay aligned at
 * any zoom level / device pixel ratio.
 */
export function CitationOverlay({ src, regions, highlighted, showAll, onSelect, links = [], showLinks = false, onFollow, flash }: {
  src: string; regions: Region[]; highlighted: number[]; showAll: boolean; onSelect?: (r: Region) => void;
  links?: PdfLink[]; showLinks?: boolean; onFollow?: (link: PdfLink) => void; flash?: PdfLink["bbox"] | null;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const visible = regions.filter((r) => showAll || highlighted.includes(r.id));
  return (
    <div className="page">
      <img src={src} alt="page" draggable={false} />
      {visible.map((r) => {
        const [x0, y0, x1, y1] = r.bbox;
        const hl = highlighted.includes(r.id);
        const color = TYPE_COLORS[r.type] ?? "#94a3b8";
        return (
          <button key={r.id} className={`box ${hl ? "hl" : ""}`} title={`R${r.id} ${r.type}`} onClick={() => onSelect?.(r)}
            onMouseEnter={() => setHover(r.id)} onMouseLeave={() => setHover(null)}
            style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%`,
              borderColor: color, background: hl ? `${color}1f` : hover === r.id ? `${color}16` : "transparent", boxShadow: hl ? `0 0 0 2px ${color}45, 0 0 12px ${color}55` : undefined }}>
            <span className="tag" style={{ background: color }}>R{r.id} {r.type}</span>
          </button>
        );
      })}
      {showLinks && links.map((link) => {
        const [x0, y0, x1, y1] = link.bbox;
        const color = LINK_COLORS[link.kind] ?? "#94a3b8";
        const summary = linkSummary(link);
        return (
          <button key={`l${link.id}`} className={`link-box ${link.inferred ? "inferred" : ""}`} title={summary} aria-label={`Follow link: ${summary}`}
            onClick={(event) => { event.stopPropagation(); onFollow?.(link); }}
            style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%`, borderColor: color, ["--link" as string]: color }} />
        );
      })}
      {flash && <span className="link-flash" aria-hidden="true" style={{ left: `${flash[0] * 100}%`, top: `${flash[1] * 100}%`, width: `${(flash[2] - flash[0]) * 100}%`, height: `${(flash[3] - flash[1]) * 100}%` }} />}
    </div>
  );
}
