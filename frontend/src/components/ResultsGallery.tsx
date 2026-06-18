import { useState } from "react";
import type { SearchHit } from "../api";
import { ImageModal } from "./ImageModal";

export function ResultsGallery({ hits }: { hits: SearchHit[] }) {
  const [selected, setSelected] = useState<SearchHit | null>(null);
  if (!hits || hits.length === 0) return null;
  return (
    <>
      <div className="gallery">
        {hits.map((h) => (
          <div className="card" key={`${h.image_id}-${h.rank}`} onClick={() => setSelected(h)}>
            <span className="rank-badge">#{h.rank}</span>
            <span className="score-badge">{h.score.toFixed(3)}</span>
            <img src={h.image_url} alt={`#${h.rank}`} loading="lazy" />
            <div className="meta">{h.caption ? h.caption.slice(0, 60) : `image ${h.image_id}`}</div>
          </div>
        ))}
      </div>
      <ImageModal hit={selected} onClose={() => setSelected(null)} />
    </>
  );
}
