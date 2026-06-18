import { useEffect } from "react";
import type { SearchHit } from "../api";

export function ImageModal({ hit, onClose }: { hit: SearchHit | null; onClose: () => void }) {
  useEffect(() => {
    if (!hit) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hit, onClose]);

  if (!hit) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ position: "relative" }}>
        <img src={hit.image_url} alt={`#${hit.rank}`} />
        <button className="close" onClick={onClose} aria-label="Close">
          ×
        </button>
        <div className="sidebar">
          <h3>
            #{hit.rank} match · score {hit.score.toFixed(3)}
          </h3>
          <dl>
            <dt>Image ID</dt>
            <dd>{hit.image_id}</dd>
            <dt>Filename</dt>
            <dd style={{ wordBreak: "break-all", fontSize: 11 }}>
              {hit.file_path.split(/[\\/]/).pop()}
            </dd>
            {hit.caption && (
              <>
                <dt>Caption</dt>
                <dd>{hit.caption}</dd>
              </>
            )}
            {hit.language && (
              <>
                <dt>Language</dt>
                <dd>{hit.language}</dd>
              </>
            )}
          </dl>
        </div>
      </div>
    </div>
  );
}
