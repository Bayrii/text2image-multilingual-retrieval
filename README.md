# Multilingual Text-to-Image Retrieval

End-to-end **multilingual text-to-image retrieval** system: type, speak, or
drop an image and the system retrieves matching photos from a 5,000-image
gallery using fine-tuned multilingual CLIP. **No generative AI** — pure
retrieval.

Three input modalities, one model:

- **Text** in any of 6 trained languages (en/ar/zh/fr/de/ru) plus best-effort
  on Azerbaijani and Turkish via XLM-RoBERTa's broader coverage.
- **Voice** via OpenAI Whisper — speak in any language, the transcript flows
  into the same retrieval pipeline.
- **Image** — drop a photo, find visually similar gallery items using the CLIP
  visual encoder directly.

## Why this project matters

Most commercial visual search engines (Google Lens, Pinterest Lens,
e-commerce search bars) degrade sharply for low-resource languages. This
project demonstrates a compact, reproducible system that:

1. Serves three input modalities through a single shared embedding space.
2. **Quantifies the cross-lingual gap** between high-resource (en/de/fr) and
   low-resource (ar/zh/ru) languages on COCO retrieval.
3. **Documents a real failure mode** — cross-lingual homograph collisions —
   that is not visible in average metrics but appears at query time. See
   [`reports/comparison/comparison.md`](reports/comparison/comparison.md).

## Architecture

```
┌──────────────────────── frontend/ (React + Vite + TS) ────────────────────────┐
│                                                                                │
│   [⌨ Text]    [🎙 Voice]    [🖼 Image]                                          │
│      │            │              │                                             │
│      │     audio blob       image file                                         │
│      │            │              │                                             │
│      └──── HTTP (/api proxy) ────┘                                             │
└─────────────────────────────┬──────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────────────┐
│                                                                                │
│  backend/ (FastAPI)                                                            │
│                                                                                │
│  /search        ──> mCLIP text encoder ──> FAISS image index ──> re-rank       │
│  /search/voice  ──> Whisper-small ─> mCLIP text enc ─> FAISS image idx         │
│  /search/image  ──> mCLIP visual encoder ──> FAISS image index                 │
│  /images/*.jpg  ──> static                                                     │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

- **Text/Image encoder:** `M-CLIP/XLM-Roberta-Large-Vit-B-32` — 110M trainable
  parameters after freezing the bottom 16 of 24 XLM-R layers.
- **Speech encoder:** `openai/whisper-small` (~250 MB, 99 languages, runs on
  CPU at ~3–5 sec per query).
- **Retrieval:** FAISS flat-IP index over 5,000 image vectors (10 MB) +
  exact-cosine re-rank.
- **Training loss:** InfoNCE + ContrastiveWithHardNegatives, temperature 0.07.

## Results

Pretrained vs fine-tuned mCLIP on 5K test set (15,012 queries × 6 languages):

| Lang | R@1 (pre) | R@1 (fine) | Δ R@1 | R@5 (fine) | mAP@10 (fine) |
|---|---|---|---|---|---|
| ar | 0.2366 | 0.2382 | **+0.0016** | 0.4692 | 0.3364 |
| de | 0.2690 | 0.2714 | +0.0024 | 0.5160 | 0.3780 |
| en | 0.2902 | 0.2882 | −0.0020 | 0.5364 | 0.3933 |
| fr | 0.2766 | 0.2786 | +0.0020 | 0.5264 | 0.3827 |
| ru | 0.2670 | 0.2566 | **−0.0104** | 0.5064 | 0.3626 |
| zh | 0.2502 | 0.2542 | +0.0040 | 0.4928 | 0.3575 |
| **overall** | **0.2649** | **0.2645** | −0.0004 | **0.5079** | **0.3684** |

**Honest reading of the result:** scaling training data 5× from 1K to 5K
images did **not** lift overall R@1 — fine-tuning plateaus at epoch 1 on this
scale. The lift comes from architectural choices (which 8 layers are
trainable) rather than data. The full discussion, per-language story, and
documented limitations live in
[`reports/comparison/comparison.md`](reports/comparison/comparison.md).

## Setup

Requires Python 3.10+, Node 18+, and ~10 GB free disk.

```powershell
# 1. Python backend deps
copy .env.example .env
pip install -r requirements.txt

# 2. Frontend deps
cd frontend
npm install
cd ..
```

Caches (HuggingFace, pip, torch) are redirected to `.cache/` via env vars —
see `src/utils.py::setup_caches`. Pre-trained model weights download on first
use.

## Reproduce the pipeline

```powershell
python scripts/01_download_data.py     # data/images/, captions_en.json (~10 min for 5K imgs)
python scripts/02_translate.py         # captions_multilingual.json    (~60 min, MarianMT)
python scripts/03_build_dataset.py     # dataset.parquet, splits.json
python scripts/04_train.py             # checkpoints/best.pt           (~1.5–2 h on RTX 3060)
python scripts/05_build_index.py       # embeddings/{image,text}.index
python scripts/08_compare_models.py    # reports/comparison/           (pretrained vs fine-tuned)
```

## Run the demo

Two processes, one in each terminal:

```powershell
# Terminal 1: backend (FastAPI on :8000, loads mCLIP + Whisper + indexes)
python -m backend.main

# Terminal 2: frontend (Vite on :5173, hot-reload)
cd frontend
npm run dev
```

Open <http://localhost:5173> in Chrome. The Vite dev server proxies `/api/*`
to the backend so there are no CORS surprises.

## Frontend features

- Live stats from `/stats` (image count, language count, index size, backend
  health)
- **Text tab**: language-tagged example chips, live short-Azerbaijani
  homograph warning, `/` keyboard shortcut to focus.
- **Voice tab**: MediaRecorder capture, **live waveform** visualizer using
  Web Audio AnalyserNode, language hint dropdown, post-transcription
  out-of-set / short-AZ warnings.
- **Image tab**: drag-drop or click-to-pick, image preview, top-K
  visually-similar results.
- **Click any result** → lightbox with full image + score + caption.
- **About modal** (`?` key) — architecture, metric table, documented
  limitations.
- Result cards stagger-fade in.

## Documented limitations

1. **Cross-lingual homograph collision** in Azerbaijani — short queries like
   `it` (dog) collide with English pronouns. See the dedicated section in
   `reports/comparison/comparison.md`. The frontend includes a heuristic
   `isProbablyShortAze` warning at query time.
2. **Frozen visual encoder** — all gain depends on text-side adaptation.
3. **MT-generated captions** — multilingual captions translated from English;
   trivially perfect `text_bridge` baseline reflects this leak.
4. **5K-data plateau** — more data did not help because contrastive loss
   converges at epoch 1 with our trainable-layer budget.

## Tests

```powershell
pytest tests/ -v
```

Covers: image-id-level splits (no leakage), InfoNCE behavior, recall + mAP
computation, retriever ordering + cardinality, FastAPI route schemas
including voice transcription stubs.

## Project layout

```
backend/         FastAPI app (main.py, routes.py, schemas.py)
frontend/        React + Vite + TypeScript (src/App.tsx, components/, api.ts)
src/             ML library used by backend + scripts
  data/          download, translate, dataset, augment
  models/        mCLIP wrapper, contrastive loss
  training/      trainer, warmup-cosine scheduler
  retrieval/     FAISS indexer, two-stage + text-bridge retrievers
  evaluation/    metrics, evaluator, report generators
  voice/         Whisper transcription handler
scripts/         01..08 numbered pipeline entry points
tests/           pytest suites
config/          config.yaml — single source of hyperparameters
reports/         comparison.md + plots (committed deliverables)
```

## License

MIT — see [LICENSE](LICENSE).
