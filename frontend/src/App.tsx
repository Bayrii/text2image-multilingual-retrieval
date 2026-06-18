import { useEffect, useState } from "react";
import { getStats, type StatsResponse } from "./api";
import { AboutModal } from "./components/AboutModal";
import { ImageSearch } from "./components/ImageSearch";
import { TextSearch } from "./components/TextSearch";
import { VoiceSearch } from "./components/VoiceSearch";

type Tab = "text" | "voice" | "image";

export default function App() {
  const [tab, setTab] = useState<Tab>("text");
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);

  useEffect(() => {
    getStats()
      .then((s) => {
        setStats(s);
        setHealthy(true);
      })
      .catch(() => setHealthy(false));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA") return;
      if (e.key === "/" && tab === "text") {
        e.preventDefault();
        document.querySelector<HTMLInputElement>('input[type="text"]')?.focus();
      }
      if (e.key === "?") {
        e.preventDefault();
        setAboutOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tab]);

  return (
    <div className="app">
      <div className="hero">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 280 }}>
            <h1>Multilingual Text-to-Image Retrieval</h1>
            <div className="tagline">
              Search a 5,000-image gallery by <strong>typing</strong>, <strong>speaking</strong>, or
              dropping in an <strong>image</strong>. Powered by fine-tuned <strong>mCLIP</strong>{" "}
              (XLM-RoBERTa-Large + CLIP ViT-B/32) and <strong>Whisper</strong>. Cross-modal cosine
              similarity in a shared 512-d semantic space.
            </div>
          </div>
          <button className="about-btn" onClick={() => setAboutOpen(true)} title="Press ? to open">
            ℹ︎ About
          </button>
        </div>
        <div className="stats">
          <div className="stat">
            <div className="value">{stats?.total_images?.toLocaleString() ?? "—"}</div>
            <div className="label">Images</div>
          </div>
          <div className="stat">
            <div className="value">{stats?.languages ?? "—"}</div>
            <div className="label">Fine-tuned Langs</div>
          </div>
          <div className="stat">
            <div className="value">{stats ? `${stats.index_size_mb.toFixed(1)} MB` : "—"}</div>
            <div className="label">Index size</div>
          </div>
          <div className="stat">
            <div className="value" style={{ color: healthy === false ? "#f28b82" : healthy ? "#81c995" : "#9aa0a6" }}>
              {healthy === null ? "…" : healthy ? "Online" : "Offline"}
            </div>
            <div className="label">Backend</div>
          </div>
        </div>
      </div>

      <div className="tabs">
        <button className={`tab ${tab === "text" ? "active" : ""}`} onClick={() => setTab("text")}>
          <span className="emoji">⌨︎</span>Text
        </button>
        <button className={`tab ${tab === "voice" ? "active" : ""}`} onClick={() => setTab("voice")}>
          <span className="emoji">🎙</span>Voice
        </button>
        <button className={`tab ${tab === "image" ? "active" : ""}`} onClick={() => setTab("image")}>
          <span className="emoji">🖼</span>Image
        </button>
      </div>

      {tab === "text" && <TextSearch />}
      {tab === "voice" && <VoiceSearch />}
      {tab === "image" && <ImageSearch />}

      <AboutModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
}
