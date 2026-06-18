"""Generate evaluation report: metrics.json, plots (PNG), report.md."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..utils import ensure_dir, get_logger

logger = get_logger("report")


# Approximate Helsinki-NLP en-{lang} OPUS training-data size in M sentence pairs.
# Used to plot resource-level vs R@1 (the cross-lingual gap).
RESOURCE_LEVEL_M = {
    "en": 100, "fr": 60, "de": 50, "es": 50, "ru": 30, "zh": 25,
    "ar": 20, "it": 30, "ja": 15, "tr": 12, "az": 1.5, "sw": 1.0,
}


def _to_markdown_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _plot_recall_bar(metrics: Dict[str, Any], out_path: Path):
    import matplotlib.pyplot as plt
    langs = [k for k in metrics if k != "overall"]
    if not langs:
        return
    pairs = sorted(((l, metrics[l].get("R@1", 0.0)) for l in langs),
                   key=lambda x: -x[1])
    labels, vals = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals)
    ax.set_ylabel("R@1")
    ax.set_title("Recall@1 per language (sorted)")
    ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_heatmap(metrics: Dict[str, Any], out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    langs = [k for k in metrics if k != "overall"]
    if not langs:
        return
    keys = ["R@1", "R@5", "R@10"]
    keys = [k for k in keys if all(k in metrics[l] for l in langs)]
    if not keys:
        return
    mat = np.array([[metrics[l][k] for k in keys] for l in langs])
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.4 * len(langs)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)
    for i in range(len(langs)):
        for j in range(len(keys)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.5 else "black", fontsize=8)
    ax.set_title("Recall@K heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_resource_scatter(metrics: Dict[str, Any], out_path: Path):
    import matplotlib.pyplot as plt
    pts = []
    for lang, m in metrics.items():
        if lang == "overall":
            continue
        if lang in RESOURCE_LEVEL_M:
            pts.append((RESOURCE_LEVEL_M[lang], m.get("R@1", 0.0), lang))
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys)
    for x, y, label in pts:
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Helsinki-NLP en-X training data (M pairs, approx, log scale)")
    ax.set_ylabel("R@1")
    ax.set_title("Cross-lingual gap: resource level vs R@1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _findings(metrics: Dict[str, Any]) -> str:
    langs = [k for k in metrics if k != "overall"]
    if not langs:
        return "No per-language results."
    sorted_langs = sorted(langs, key=lambda l: -metrics[l].get("R@1", 0.0))
    best = sorted_langs[0]
    worst = sorted_langs[-1]
    gap = metrics[best]["R@1"] - metrics[worst]["R@1"]
    overall = metrics.get("overall", {})
    bullets = [
        f"- Overall R@1 = **{overall.get('R@1', 0):.3f}**, R@5 = **{overall.get('R@5', 0):.3f}**, "
        f"R@10 = **{overall.get('R@10', 0):.3f}**.",
        f"- Best language: **{best}** (R@1 = {metrics[best]['R@1']:.3f}).",
        f"- Worst language: **{worst}** (R@1 = {metrics[worst]['R@1']:.3f}).",
        f"- Cross-lingual gap (best - worst R@1): **{gap:.3f}**.",
    ]
    return "\n".join(bullets)


def generate_report(metrics_payload: Dict[str, Any], output_dir: str | Path,
                    notes: str = "") -> Path:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    metrics = metrics_payload["metrics"]

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    bar = output_dir / "recall_bar.png"
    heat = output_dir / "recall_heatmap.png"
    scat = output_dir / "resource_vs_recall.png"
    _plot_recall_bar(metrics, bar)
    _plot_heatmap(metrics, heat)
    _plot_resource_scatter(metrics, scat)

    overall = metrics.get("overall", {})
    summary_rows = [[k, f"{overall.get(k, 0):.4f}"] for k in
                    ("R@1", "R@5", "R@10", "mAP@10")]
    summary_tbl = _to_markdown_table(["metric", "value"], summary_rows)

    per_lang_keys = ["R@1", "R@5", "R@10", "mAP@10", "n_queries"]
    per_rows = []
    for lang in sorted(k for k in metrics if k != "overall"):
        r = [lang]
        for k in per_lang_keys:
            v = metrics[lang].get(k)
            r.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        per_rows.append(r)
    per_lang_tbl = _to_markdown_table(["language"] + per_lang_keys, per_rows)

    md = f"""# Evaluation Report

{notes}

## Summary (overall)

{summary_tbl}

## Per-language

{per_lang_tbl}

## Plots

![Recall@1 per language](recall_bar.png)

![Recall@K heatmap](recall_heatmap.png)

![Resource level vs R@1](resource_vs_recall.png)

## Key findings

{_findings(metrics)}
"""
    out_md = output_dir / "report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Report written: %s", out_md)
    return out_md
