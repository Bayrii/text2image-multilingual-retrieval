import { useState } from "react";
import { isProbablyShortAze, searchText, type SearchHit } from "../api";
import { ResultsGallery } from "./ResultsGallery";

const EXAMPLES = [
  { lang: "en", text: "a man riding a bicycle on a city street" },
  { lang: "de", text: "Ein Hund spielt im Park" },
  { lang: "zh", text: "猫在窗边坐着" },
  { lang: "ar", text: "رجل يقود دراجة في الشارع" },
  { lang: "ru", text: "две девочки играют в мяч" },
  { lang: "fr", text: "Un chat noir assis sur un canapé" },
  { lang: "az", text: "küçədə velosiped sürən kişi" },
];

export function TextSearch() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [retriever, setRetriever] = useState<"two_stage" | "text_bridge">("two_stage");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const showAzWarning = isProbablyShortAze(query);

  async function run(q?: string) {
    const finalQuery = (q ?? query).trim();
    if (!finalQuery) return;
    setLoading(true);
    setStatus("");
    setHits([]);
    setSearched(true);
    try {
      const data = await searchText(finalQuery, topK, retriever);
      setHits(data.results);
      setStatus(`${data.results.length} hits · ${data.latency_ms.toFixed(0)} ms`);
    } catch (e) {
      setStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="row">
        <input
          className="grow"
          type="text"
          placeholder="Type your query in any language…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
          {[3, 5, 10, 20].map((n) => (
            <option key={n} value={n}>
              top-{n}
            </option>
          ))}
        </select>
        <select
          value={retriever}
          onChange={(e) => setRetriever(e.target.value as "two_stage" | "text_bridge")}
        >
          <option value="two_stage">Two-stage</option>
          <option value="text_bridge">Text bridge</option>
        </select>
        <button className="primary" onClick={() => run()} disabled={loading || !query.trim()}>
          {loading ? "Searching…" : "Search"}
        </button>
      </div>

      {showAzWarning && (
        <div className="warning">
          <strong>Short Azerbaijani query detected.</strong> Single-token words like{" "}
          <code>it</code> (dog) collide with English homographs in mCLIP's shared embedding space.
          Try a longer phrase like <code>it tullanır</code> or <code>parkda it</code> for better
          results — context disambiguates the language.
        </div>
      )}

      <div className="examples">
        <span className="label">Examples:</span>
        {EXAMPLES.map((ex) => (
          <span
            key={ex.text}
            className="chip"
            title={ex.lang.toUpperCase()}
            onClick={() => {
              setQuery(ex.text);
              run(ex.text);
            }}
          >
            <span style={{ opacity: 0.6, marginRight: 6 }}>[{ex.lang}]</span>
            {ex.text}
          </span>
        ))}
      </div>

      <div className={`status ${status.startsWith("Error") ? "error" : ""}`}>
        {loading && <span className="spinner" />}
        {status}
      </div>

      {searched && hits.length === 0 && !loading && !status.startsWith("Error") && (
        <div className="empty-state">
          <div className="icon">🔍</div>
          <div>No results returned for that query.</div>
        </div>
      )}

      {!searched && (
        <div className="empty-state">
          <div className="icon">🌐</div>
          <div>Try a query in any of the 7 demonstrated languages above.</div>
        </div>
      )}

      <ResultsGallery hits={hits} />
    </div>
  );
}
