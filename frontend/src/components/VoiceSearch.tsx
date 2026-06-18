import { useEffect, useRef, useState } from "react";
import { isProbablyShortAze, searchVoice, type SearchHit } from "../api";
import { ResultsGallery } from "./ResultsGallery";
import { WaveformVisualizer } from "./WaveformVisualizer";

const LANGUAGES = [
  { code: "auto", label: "Auto-detect" },
  { code: "az", label: "Azerbaijani" },
  { code: "en", label: "English" },
  { code: "ar", label: "Arabic" },
  { code: "zh", label: "Chinese" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "ru", label: "Russian" },
  { code: "tr", label: "Turkish" },
];

const TRAINED_LANGS = new Set(["en", "ar", "zh", "fr", "de", "ru"]);

export function VoiceSearch() {
  const [recording, setRecording] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [language, setLanguage] = useState("auto");
  const [topK, setTopK] = useState(5);
  const [retriever, setRetriever] = useState<"two_stage" | "text_bridge">("two_stage");
  const [transcript, setTranscript] = useState("");
  const [detectedLang, setDetectedLang] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      stream?.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  async function startRecording() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      setStream(s);
      const mr = new MediaRecorder(s);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        setAudioBlob(blob);
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioUrl(URL.createObjectURL(blob));
        s.getTracks().forEach((t) => t.stop());
        setStream(null);
      };
      mr.start();
      recorderRef.current = mr;
      setRecording(true);
      setStatus("Recording… click Stop when done.");
      setTranscript("");
      setDetectedLang("");
    } catch (e) {
      setStatus(`Mic error: ${e instanceof Error ? e.message : String(e)}. Check Chrome's mic permission.`);
    }
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
    setStatus("Recorded. Press Search.");
  }

  async function search() {
    if (!audioBlob) return;
    setLoading(true);
    setStatus("");
    setHits([]);
    setTranscript("");
    setSearched(true);
    try {
      const data = await searchVoice(audioBlob, topK, retriever, language);
      setTranscript(data.transcript);
      setDetectedLang(data.detected_language);
      setHits(data.results);
      setStatus(
        data.transcript
          ? `transcribe ${data.transcribe_ms.toFixed(0)} ms · retrieve ${data.retrieve_ms.toFixed(0)} ms · ${data.results.length} hits`
          : "No speech detected.",
      );
    } catch (e) {
      setStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  const preAzWarning = language === "az" && (
    <div className="warning">
      <strong>Azerbaijani is outside our fine-tuned set</strong> (en/ar/zh/fr/de/ru). Whisper-small's
      AZ accuracy is limited, and short AZ words collide with English homographs in mCLIP (e.g.{" "}
      <code>it</code> = dog in AZ but pronoun in EN). Speak clearly, use a full phrase.
    </div>
  );

  const detectedOutOfSet =
    detectedLang && !TRAINED_LANGS.has(detectedLang) && language === "auto" ? (
      <div className="warning">
        Whisper detected <strong>{detectedLang}</strong>, which is outside our fine-tuned set.
        Selecting your language explicitly usually improves results.
      </div>
    ) : null;

  const shortAzTranscript =
    transcript && (detectedLang === "az" || language === "az") && isProbablyShortAze(transcript) ? (
      <div className="warning">
        Short Azerbaijani transcript (<code>{transcript}</code>) is ambiguous in mCLIP — try a
        longer phrase next time.
      </div>
    ) : null;

  return (
    <div>
      <div className="row">
        {!recording ? (
          <button className="record" onClick={startRecording} disabled={loading}>
            ● Record
          </button>
        ) : (
          <button className="record recording" onClick={stopRecording}>
            ■ Stop
          </button>
        )}
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          {LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.label}
            </option>
          ))}
        </select>
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
        <button className="primary" onClick={search} disabled={!audioBlob || loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </div>

      {(recording || stream) && (
        <div className="waveform-wrap">
          <WaveformVisualizer stream={stream} active={recording} />
        </div>
      )}

      {preAzWarning}

      {audioUrl && !recording && <audio controls src={audioUrl} />}

      {transcript && (
        <div className="transcript">
          <span className="lang-badge">{detectedLang || "?"}</span>
          <span>{transcript}</span>
        </div>
      )}

      {detectedOutOfSet}
      {shortAzTranscript}

      <div className={`status ${status.startsWith("Error") || status.startsWith("Mic") ? "error" : ""}`}>
        {loading && <span className="spinner" />}
        {status}
      </div>

      {!searched && !audioBlob && (
        <div className="empty-state">
          <div className="icon">🎙️</div>
          <div>Click Record, speak a query in any language, then Search.</div>
        </div>
      )}

      <ResultsGallery hits={hits} />
    </div>
  );
}
