"""Gradio UI that calls the FastAPI backend.

Two tabs:
  * Text query in any of 6 languages → /search
  * Voice query (microphone) → /search/voice (Whisper → retrieval)

Run the API first:
    python scripts/07_serve.py
Then in another terminal:
    python app/gradio_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap caches & sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts._bootstrap  # noqa: E402,F401

import gradio as gr
import requests

API_URL = os.environ.get("T2I_API_URL", "http://127.0.0.1:8000")

EXAMPLES = [
    "a man riding a bicycle on a city street",
    "Un chat noir assis sur un canapé",
    "Ein Hund spielt im Park",
    "猫が窓辺に座っている",
    "Una mujer sosteniendo una taza de café",
    "رجل يقود دراجة في الشارع",
    "две девочки играют в мяч",
    "一只狗在公园里跑",
    "Bir adam plajda yürüyor",
    "Il piatto di pasta è sul tavolo",
]

VOICE_LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("English", "en"),
    ("Arabic", "ar"),
    ("Chinese", "zh"),
    ("French", "fr"),
    ("German", "de"),
    ("Russian", "ru"),
    ("Azerbaijani", "az"),
    ("Turkish", "tr"),
]

RETRIEVER_LABELS = ["Two-stage (fine-tuned)", "Text bridge"]


def _hits_to_outputs(hits):
    images, table = [], []
    for hit in hits:
        path = hit.get("file_path", "")
        if path and Path(path).exists():
            images.append((path, f"#{hit['rank']}  score={hit['score']:.3f}"))
        table.append([
            hit["rank"],
            f"{hit['score']:.3f}",
            hit["image_id"],
            hit.get("language") or "",
            (hit.get("caption") or "")[:80],
        ])
    return images, table


def _retriever_value(label: str) -> str:
    return "two_stage" if label.startswith("Two") else "text_bridge"


def search_text(query: str, top_k: int, retriever_label: str):
    if not query or not query.strip():
        return [], [], "0.0 ms"
    try:
        r = requests.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": int(top_k),
                  "retriever": _retriever_value(retriever_label)},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [], [["—", "—", 0, "", f"API error: {e}"]], "—"
    images, table = _hits_to_outputs(data["results"])
    return images, table, f"{data['latency_ms']:.1f} ms"


def search_voice(audio_path, top_k: int, retriever_label: str, lang_choice: str):
    if not audio_path:
        return "", [], [], "—"
    lang = None if lang_choice == "auto" else lang_choice
    try:
        with open(audio_path, "rb") as af:
            files = {"audio": ("query.wav", af, "audio/wav")}
            data = {"top_k": str(int(top_k)),
                    "retriever": _retriever_value(retriever_label)}
            if lang:
                data["language"] = lang
            r = requests.post(f"{API_URL}/search/voice",
                              files=files, data=data, timeout=180)
            r.raise_for_status()
            result = r.json()
    except Exception as e:
        return f"API error: {e}", [], [], "—"

    transcript = result.get("transcript", "")
    detected = result.get("detected_language", "?")
    transcript_display = (
        f"Transcript [{detected}]: {transcript}"
        if transcript else f"(no speech detected — language guess: {detected})"
    )
    images, table = _hits_to_outputs(result.get("results", []))
    timing = (f"transcribe={result.get('transcribe_ms', 0):.0f} ms  "
              f"retrieve={result.get('retrieve_ms', 0):.0f} ms")
    return transcript_display, images, table, timing


def build_ui():
    with gr.Blocks(title="Multilingual Text-to-Image Retrieval") as demo:
        gr.Markdown("# Multilingual Text-to-Image Retrieval")
        gr.Markdown(f"Backend: `{API_URL}`")

        with gr.Tabs():
            # ------- Text tab -------
            with gr.Tab("Text"):
                with gr.Row():
                    query = gr.Textbox(label="Query (any language)", lines=1, scale=4)
                    top_k_t = gr.Slider(1, 10, value=5, step=1, label="Top-K", scale=1)
                    retriever_t = gr.Dropdown(
                        choices=RETRIEVER_LABELS, value=RETRIEVER_LABELS[0],
                        label="Retriever", scale=1,
                    )
                btn_t = gr.Button("Search", variant="primary")
                gallery_t = gr.Gallery(label="Top results", columns=5, height=320)
                table_t = gr.Dataframe(
                    headers=["rank", "score", "image_id", "lang", "caption"],
                    datatype=["number", "str", "number", "str", "str"],
                    interactive=False, label="Match details",
                )
                latency_t = gr.Textbox(label="Latency", interactive=False)
                gr.Examples(EXAMPLES, inputs=[query], label="Example queries")
                btn_t.click(search_text, [query, top_k_t, retriever_t],
                            [gallery_t, table_t, latency_t])
                query.submit(search_text, [query, top_k_t, retriever_t],
                             [gallery_t, table_t, latency_t])

            # ------- Voice tab -------
            with gr.Tab("Voice"):
                gr.Markdown(
                    "Click the mic, speak your query in any supported language, "
                    "then press **Search by voice**. Whisper transcribes the audio "
                    "and the transcript is fed into the same retrieval pipeline."
                )
                with gr.Row():
                    audio = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Speak query",
                        scale=3,
                    )
                    with gr.Column(scale=2):
                        lang_v = gr.Dropdown(
                            choices=VOICE_LANG_CHOICES,
                            value="auto",
                            label="Language hint",
                        )
                        top_k_v = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                        retriever_v = gr.Dropdown(
                            choices=RETRIEVER_LABELS, value=RETRIEVER_LABELS[0],
                            label="Retriever",
                        )
                btn_v = gr.Button("Search by voice", variant="primary")
                transcript_v = gr.Textbox(label="Transcript", interactive=False)
                gallery_v = gr.Gallery(label="Top results", columns=5, height=320)
                table_v = gr.Dataframe(
                    headers=["rank", "score", "image_id", "lang", "caption"],
                    datatype=["number", "str", "number", "str", "str"],
                    interactive=False, label="Match details",
                )
                timing_v = gr.Textbox(label="Latency", interactive=False)
                btn_v.click(
                    search_voice,
                    [audio, top_k_v, retriever_v, lang_v],
                    [transcript_v, gallery_v, table_v, timing_v],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,
                inbrowser=True)
