import { useState } from "react";
import type { Region } from "../api";

export const TYPE_COLORS: Record<string, string> = { text: "#64748b", title: "#a78bfa", table: "#f59e0b", figure: "#22d3ee", chart: "#34d399", equation: "#f472b6" };

/**
 * Pixel-level citation overlay: draws normalized bboxes on top of the rendered
 * page image. Because bboxes are in 0..1 page coordinates, they stay aligned at
 * any zoom level / device pixel ratio.
 */
export function CitationOverlay({ src, regions, highlighted, showAll, onSelect }: {
  src: string; regions: Region[]; highlighted: number[]; showAll: boolean; onSelect?: (r: Region) => void;
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
    </div>
  );
}
