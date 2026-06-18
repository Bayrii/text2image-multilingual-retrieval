"""Pretrained vs fine-tuned comparison.

Runs the same evaluation pipeline twice — once with raw pretrained mCLIP
weights, once with the fine-tuned checkpoint — on the same test split, and
generates a side-by-side report that quantifies the impact of fine-tuning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..models.mclip import MultilingualCLIPModel
from ..retrieval.indexer import FAISSIndexer
from ..retrieval.retriever import TwoStageRetriever
from ..utils import ensure_dir, get_logger, save_json
from .evaluator import Evaluator
from .report import RESOURCE_LEVEL_M

logger = get_logger("compare")


@torch.no_grad()
def _embed_images(model, file_paths: List[str], device, batch_size: int = 32) -> np.ndarray:
    embs = []
    for i in tqdm(range(0, len(file_paths), batch_size), desc="img-embed"):
        chunk = file_paths[i:i + batch_size]
        pil = [Image.open(p).convert("RGB") for p in chunk]
        pixels = model.image_processor(images=pil, return_tensors="pt")["pixel_values"].to(device)
        e = model.encode_image(pixels).float().cpu().numpy()
        embs.append(e)
    return np.concatenate(embs, axis=0)


def _build_index_from_model(model, rows: List[Dict[str, Any]],
                            embedding_dim: int, device) -> FAISSIndexer:
    seen = set()
    img_meta, img_files = [], []
    for r in rows:
        iid = int(r["image_id"])
        if iid in seen:
            continue
        seen.add(iid)
        img_meta.append({"image_id": iid, "file_path": r["file_path"]})
        img_files.append(r["file_path"])
    embs = _embed_images(model, img_files, device=device)
    indexer = FAISSIndexer(dim=embedding_dim, index_type="flat_ip")
    indexer.build(embs, img_meta)
    return indexer


def _load_pretrained(cfg, device) -> MultilingualCLIPModel:
    logger.info("Loading PRETRAINED mCLIP (no fine-tuning)...")
    model = MultilingualCLIPModel(cfg).to(device)
    model.eval()
    return model


def _load_finetuned(cfg, device, ckpt_path: Path) -> MultilingualCLIPModel:
    logger.info("Loading FINE-TUNED checkpoint: %s", ckpt_path)
    model = MultilingualCLIPModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def _eval_one(label: str, model, all_rows, test_rows, cfg, device) -> Dict[str, Any]:
    logger.info("=== Evaluating: %s ===", label)
    indexer = _build_index_from_model(model, all_rows, int(cfg["model"]["embedding_dim"]), device)
    retriever = TwoStageRetriever(model, indexer, cfg)
    out = Evaluator(retriever, test_rows, cfg).run()
    return out


def _comparison_table(pre_metrics: Dict, post_metrics: Dict) -> str:
    langs = sorted([k for k in pre_metrics if k != "overall"])
    if "overall" in pre_metrics:
        langs = langs + ["overall"]
    headers = ["lang", "R@1 (pre)", "R@1 (fine)", "Δ R@1",
               "R@5 (pre)", "R@5 (fine)", "Δ R@5",
               "mAP@10 (pre)", "mAP@10 (fine)", "Δ mAP", "n"]
    rows = []
    for lang in langs:
        p = pre_metrics.get(lang, {})
        f = post_metrics.get(lang, {})
        def _d(a, b): return f"{(b - a):+.4f}" if (a is not None and b is not None) else "—"
        rows.append([
            lang,
            f"{p.get('R@1', 0):.4f}", f"{f.get('R@1', 0):.4f}", _d(p.get("R@1", 0), f.get("R@1", 0)),
            f"{p.get('R@5', 0):.4f}", f"{f.get('R@5', 0):.4f}", _d(p.get("R@5", 0), f.get("R@5", 0)),
            f"{p.get('mAP@10', 0):.4f}", f"{f.get('mAP@10', 0):.4f}",
            _d(p.get("mAP@10", 0), f.get("mAP@10", 0)),
            str(p.get("n_queries", f.get("n_queries", 0))),
        ])
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _plot_grouped_bar(pre_metrics: Dict, post_metrics: Dict, out_path: Path,
                      metric_key: str = "R@1"):
    import matplotlib.pyplot as plt
    langs = sorted([k for k in pre_metrics if k != "overall"])
    if not langs:
        return
    pre_vals = [pre_metrics[l].get(metric_key, 0.0) for l in langs]
    post_vals = [post_metrics.get(l, {}).get(metric_key, 0.0) for l in langs]
    x = np.arange(len(langs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(langs) * 1.1), 4.5))
    b1 = ax.bar(x - width / 2, pre_vals, width, label="Pretrained", color="#9aa0a6")
    b2 = ax.bar(x + width / 2, post_vals, width, label="Fine-tuned", color="#1a73e8")
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} per language: pretrained vs fine-tuned")
    ax.set_ylim(0, max(max(pre_vals + post_vals) * 1.2, 0.05))
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_gain(pre_metrics: Dict, post_metrics: Dict, out_path: Path,
               metric_key: str = "R@1"):
    import matplotlib.pyplot as plt
    langs = sorted([k for k in pre_metrics if k != "overall"])
    if not langs:
        return
    gains = [post_metrics.get(l, {}).get(metric_key, 0) - pre_metrics[l].get(metric_key, 0)
             for l in langs]
    pairs = sorted(zip(langs, gains), key=lambda x: -x[1])
    labels, vals = zip(*pairs)
    colors = ["#34a853" if v >= 0 else "#ea4335" for v in vals]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4))
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel(f"Δ {metric_key} (fine-tuned − pretrained)")
    ax.set_title(f"Fine-tuning gain per language ({metric_key})")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_resource_gap(pre_metrics: Dict, post_metrics: Dict, out_path: Path):
    """Cross-lingual gap visualisation: x = MT corpus size (log), y = R@1, two series."""
    import matplotlib.pyplot as plt
    pts = []
    for lang in pre_metrics:
        if lang == "overall":
            continue
        if lang in RESOURCE_LEVEL_M:
            pts.append((RESOURCE_LEVEL_M[lang], lang,
                        pre_metrics[lang].get("R@1", 0.0),
                        post_metrics.get(lang, {}).get("R@1", 0.0)))
    if not pts:
        return
    pts.sort()
    xs = [p[0] for p in pts]
    pre_y = [p[2] for p in pts]
    post_y = [p[3] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, pre_y, "o--", color="#9aa0a6", label="Pretrained", markersize=8)
    ax.plot(xs, post_y, "o-", color="#1a73e8", label="Fine-tuned", markersize=8)
    for x, lang, py, fy in pts:
        ax.annotate(lang, (x, max(py, fy)), xytext=(5, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Helsinki-NLP en-X training data (M pairs, approx, log scale)")
    ax.set_ylabel("R@1")
    ax.set_title("Cross-lingual gap: pretrained vs fine-tuned")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _findings(pre: Dict, post: Dict) -> str:
    overall_pre = pre.get("overall", {})
    overall_post = post.get("overall", {})
    langs = [k for k in pre if k != "overall"]
    if not langs:
        return "_No per-language data._"
    gains = [(l, post.get(l, {}).get("R@1", 0) - pre[l].get("R@1", 0)) for l in langs]
    gains.sort(key=lambda x: -x[1])
    top_lang, top_gain = gains[0]
    bot_lang, bot_gain = gains[-1]
    pre_overall_r1 = overall_pre.get("R@1", 0)
    post_overall_r1 = overall_post.get("R@1", 0)
    delta = post_overall_r1 - pre_overall_r1
    rel = (delta / pre_overall_r1 * 100) if pre_overall_r1 > 0 else 0.0
    return "\n".join([
        f"- **Overall R@1** lifted from **{pre_overall_r1:.4f}** (pretrained) to "
        f"**{post_overall_r1:.4f}** (fine-tuned) — Δ = **{delta:+.4f}** "
        f"({rel:+.1f}% relative).",
        f"- **Largest gain**: `{top_lang}` ({top_gain:+.4f} R@1).",
        f"- **Smallest gain**: `{bot_lang}` ({bot_gain:+.4f} R@1).",
        f"- Pretrained overall R@5 = {overall_pre.get('R@5', 0):.4f}, "
        f"fine-tuned overall R@5 = {overall_post.get('R@5', 0):.4f}.",
        f"- Pretrained overall mAP@10 = {overall_pre.get('mAP@10', 0):.4f}, "
        f"fine-tuned overall mAP@10 = {overall_post.get('mAP@10', 0):.4f}.",
    ])


def generate_comparison_report(pre_metrics: Dict, post_metrics: Dict,
                               output_dir: Path, ckpt_info: str = "") -> Path:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    save_json({"pretrained": pre_metrics, "finetuned": post_metrics},
              output_dir / "comparison_metrics.json")

    bar_r1 = output_dir / "compare_R@1.png"
    bar_r5 = output_dir / "compare_R@5.png"
    gain_r1 = output_dir / "gain_R@1.png"
    res_gap = output_dir / "resource_gap_compare.png"
    _plot_grouped_bar(pre_metrics, post_metrics, bar_r1, "R@1")
    _plot_grouped_bar(pre_metrics, post_metrics, bar_r5, "R@5")
    _plot_gain(pre_metrics, post_metrics, gain_r1, "R@1")
    _plot_resource_gap(pre_metrics, post_metrics, res_gap)

    table = _comparison_table(pre_metrics, post_metrics)
    md = f"""# Pretrained vs Fine-tuned mCLIP — Comparison Report

{ckpt_info}

## Side-by-side metrics

{table}

## Key findings

{_findings(pre_metrics, post_metrics)}

## Plots

### R@1 per language
![compare R@1](compare_R@1.png)

### R@5 per language
![compare R@5](compare_R@5.png)

### Fine-tuning gain (Δ R@1)
![gain R@1](gain_R@1.png)

### Cross-lingual gap (MT-corpus size vs R@1)
![resource gap](resource_gap_compare.png)
"""
    out_md = output_dir / "comparison.md"
    out_md.write_text(md, encoding="utf-8")
    logger.info("Comparison report -> %s", out_md)
    return out_md


def run_comparison(cfg: Dict[str, Any], all_rows: List[Dict],
                   test_rows: List[Dict], device, ckpt_path: Path,
                   output_dir: Path) -> Dict[str, Any]:
    pre_model = _load_pretrained(cfg, device)
    pre_out = _eval_one("pretrained", pre_model, all_rows, test_rows, cfg, device)
    del pre_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fine_model = _load_finetuned(cfg, device, ckpt_path)
    fine_out = _eval_one("fine-tuned", fine_model, all_rows, test_rows, cfg, device)
    del fine_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pre_metrics = pre_out["metrics"]
    post_metrics = fine_out["metrics"]
    info = (f"_Checkpoint: `{ckpt_path}`_  \n"
            f"_Test queries: {pre_metrics.get('overall', {}).get('n_queries', 0)} "
            f"across {len([k for k in pre_metrics if k != 'overall'])} languages._")
    generate_comparison_report(pre_metrics, post_metrics, output_dir, ckpt_info=info)
    return {"pretrained": pre_metrics, "finetuned": post_metrics}
