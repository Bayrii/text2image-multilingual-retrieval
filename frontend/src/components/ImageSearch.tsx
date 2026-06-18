import { useEffect, useRef, useState } from "react";
import { searchImage, type SearchHit } from "../api";
import { ResultsGallery } from "./ResultsGallery";

export function ImageSearch() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function setSelectedFile(f: File | null) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (!f) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    if (!f.type.startsWith("image/")) {
      setStatus(`Not an image: ${f.type || "unknown"}`);
      return;
    }
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setStatus(`Selected ${f.name} (${(f.size / 1024).toFixed(1)} KB)`);
    setHits([]);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setSelectedFile(f);
  }

  async function run() {
    if (!file) return;
    setLoading(true);
    setStatus("");
    setHits([]);
    setSearched(true);
    try {
      const data = await searchImage(file, topK);
      setHits(data.results);
      setStatus(`${data.results.length} visually-similar images · ${data.latency_ms.toFixed(0)} ms`);
    } catch (e) {
      setStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div
        className={`drop-zone ${dragOver ? "drag-over" : ""} ${previewUrl ? "has-image" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
        />
        {previewUrl ? (
          <div className="preview-wrap">
            <img src={previewUrl} alt="query preview" className="preview" />
            <button
              className="primary"
              onClick={(e) => {
                e.stopPropagation();
                setSelectedFile(null);
                setHits([]);
                setStatus("");
              }}
              style={{ background: "#5f6368", marginTop: 10 }}
            >
              Clear
            </button>
          </div>
        ) : (
          <>
            <div className="drop-icon">📷</div>
            <div className="drop-text">
              <strong>Drop an image here</strong> or click to browse
            </div>
            <div className="drop-hint">
              JPEG / PNG / WebP — finds visually similar images in the 5,000-image gallery
            </div>
          </>
        )}
      </div>

      <div className="row" style={{ marginTop: 16 }}>
        <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
          {[3, 5, 10, 20].map((n) => (
            <option key={n} value={n}>
              top-{n}
            </option>
          ))}
        </select>
        <button className="primary" onClick={run} disabled={!file || loading}>
          {loading ? "Searching…" : "Find similar"}
        </button>
        <div style={{ fontSize: 12, color: "#9aa0a6", marginLeft: 4 }}>
          Uses CLIP ViT-B/32 visual encoder · cosine similarity in 512-d space
        </div>
      </div>

      <div className={`status ${status.startsWith("Error") || status.startsWith("Not") ? "error" : ""}`}>
        {loading && <span className="spinner" />}
        {status}
      </div>

      {!searched && !file && (
        <div className="empty-state">
          <div className="icon">🖼️</div>
          <div>Drop or pick an image to find visually-similar gallery items.</div>
        </div>
      )}

      <ResultsGallery hits={hits} />
    </div>
  );
}
