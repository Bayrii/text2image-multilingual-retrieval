export type SearchHit = {
  image_id: number;
  score: number;
  file_path: string;
  image_url: string;
  rank: number;
  caption?: string | null;
  language?: string | null;
};

export type SearchResponse = {
  query: string;
  results: SearchHit[];
  latency_ms: number;
};

export type VoiceSearchResponse = {
  transcript: string;
  detected_language: string;
  results: SearchHit[];
  transcribe_ms: number;
  retrieve_ms: number;
};

export type StatsResponse = {
  total_images: number;
  languages: number;
  index_size_mb: number;
};

const API = "/api";

export async function getStats(): Promise<StatsResponse> {
  const r = await fetch(`${API}/stats`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function searchText(
  query: string,
  topK: number,
  retriever: "two_stage" | "text_bridge",
): Promise<SearchResponse> {
  const r = await fetch(`${API}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK, retriever }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function searchVoice(
  audioBlob: Blob,
  topK: number,
  retriever: "two_stage" | "text_bridge",
  language: string | null,
): Promise<VoiceSearchResponse> {
  const fd = new FormData();
  fd.append("audio", audioBlob, "query.webm");
  fd.append("top_k", String(topK));
  fd.append("retriever", retriever);
  if (language && language !== "auto") fd.append("language", language);

  const r = await fetch(`${API}/search/voice`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function searchImage(file: File, topK: number): Promise<SearchResponse> {
  const fd = new FormData();
  fd.append("image", file, file.name);
  fd.append("top_k", String(topK));
  const r = await fetch(`${API}/search/image`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

const AZ_CHARS = /[əşğçıİÜüÖöĞŞÇƏ]/;
const COMMON_AZ_WORDS = new Set([
  "it", "ev", "at", "su", "yol", "qız", "oğlan", "kişi", "qadın",
  "uşaq", "kitab", "masa", "stol", "günəş", "ay", "ulduz",
]);

export function isProbablyShortAze(query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return false;
  const tokens = q.split(/\s+/);
  if (tokens.length > 2) return false;
  if (AZ_CHARS.test(q)) return true;
  return tokens.some((t) => COMMON_AZ_WORDS.has(t));
}
