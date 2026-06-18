import { useEffect } from "react";

export function AboutModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal about"
        onClick={(e) => e.stopPropagation()}
        style={{ gridTemplateColumns: "1fr", maxWidth: 720, position: "relative" }}
      >
        <button className="close" onClick={onClose} aria-label="Close">×</button>
        <div className="sidebar" style={{ maxHeight: "88vh" }}>
          <h3 style={{ fontSize: 22, marginBottom: 4 }}>Multilingual Text-to-Image Retrieval</h3>
          <div style={{ color: "#9aa0a6", fontSize: 13, marginBottom: 18 }}>
            Final-year project · fine-tuned mCLIP + Whisper · React + FastAPI
          </div>

          <h4 style={section}>Architecture</h4>
          <ul style={ul}>
            <li><strong>Text/Image encoder:</strong> M-CLIP/XLM-RoBERTa-Large-Vit-B-32 (110M trainable parameters after partial freezing of XLM-R layers 0–15)</li>
            <li><strong>Speech encoder:</strong> OpenAI Whisper-small (250 MB, 99 languages, runs on CPU)</li>
            <li><strong>Retrieval:</strong> FAISS flat IP index + exact re-rank (two-stage); 5,000 images × 512-d</li>
            <li><strong>Loss (training):</strong> InfoNCE + ContrastiveWithHardNegatives, temperature 0.07</li>
          </ul>

          <h4 style={section}>Three input modalities</h4>
          <ul style={ul}>
            <li><strong>Text</strong> — type in en/ar/zh/fr/de/ru or any XLM-R-supported language</li>
            <li><strong>Voice</strong> — Whisper transcribes → mCLIP embeds the transcript</li>
            <li><strong>Image</strong> — direct visual-encoder query, find similar images</li>
          </ul>

          <h4 style={section}>Fine-tuning result (5K test set, 15,012 queries)</h4>
          <table style={tbl}>
            <thead>
              <tr>
                <th>Metric</th><th>Pretrained</th><th>Fine-tuned</th><th>Δ</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>R@1 overall</td><td>0.2649</td><td>0.2645</td><td style={{ color: "#f28b82" }}>−0.04 pp</td></tr>
              <tr><td>R@5 overall</td><td>0.5007</td><td>0.5079</td><td style={{ color: "#81c995" }}>+0.72 pp</td></tr>
              <tr><td>mAP@10</td><td>0.3677</td><td>0.3684</td><td style={{ color: "#81c995" }}>+0.07 pp</td></tr>
            </tbody>
          </table>
          <div style={{ fontSize: 12, color: "#9aa0a6", marginTop: 6 }}>
            5× more data than v2 produced essentially no metric lift — the contrastive loss
            plateaus at epoch 1 on this scale. Documented as a real finding (see comparison.md).
          </div>

          <h4 style={section}>Notable limitations (documented in report)</h4>
          <ul style={ul}>
            <li><strong>Cross-lingual homograph collision</strong> in Azerbaijani — short queries like <code>it</code> (dog) collide with English pronouns in the shared embedding space</li>
            <li><strong>Frozen visual encoder</strong> — all gains depend on text-side adaptation</li>
            <li><strong>MT-generated captions</strong> — multilingual captions translated from English; trivially perfect text_bridge baseline reflects this</li>
          </ul>

          <h4 style={section}>Keyboard shortcuts</h4>
          <ul style={ul}>
            <li><code>/</code> — focus text query</li>
            <li><code>Escape</code> — close modal</li>
            <li><code>Enter</code> — run text search</li>
          </ul>

          <div style={{ marginTop: 18, fontSize: 12, color: "#9aa0a6" }}>
            Report: <code>reports/comparison/comparison.md</code> · Source:{" "}
            <code>backend/</code> + <code>frontend/</code>
          </div>
        </div>
      </div>
    </div>
  );
}

const section: React.CSSProperties = {
  marginTop: 18,
  marginBottom: 8,
  fontSize: 14,
  color: "#8ab4f8",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
};
const ul: React.CSSProperties = { margin: 0, paddingLeft: 18, color: "#cfd5dc", fontSize: 13, lineHeight: 1.6 };
const tbl: React.CSSProperties = { width: "100%", borderCollapse: "collapse", fontSize: 13, marginTop: 6 };
