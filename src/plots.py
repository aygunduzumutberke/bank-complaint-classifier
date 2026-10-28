"""Matplotlib figures shared by the report and the Streamlit app."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import CATEGORIES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"
SERIES = {"llm": "#2a78d6", "baseline": "#eb6834"}  # blue, orange: a colour-blind-safe pair
NAMES = {"llm": "LLM (Gemini)", "baseline": "TF-IDF + Logistic Regression"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=9)


def f1_by_category(comparison: dict, llm_name: str | None = None):
    """Grouped horizontal bars: F1 per category for both models, weakest LLM class at the bottom."""
    llm, base = comparison["llm"]["per_class"], comparison["baseline"]["per_class"]
    cats = sorted(llm, key=lambda c: llm[c]["f1"])
    y = np.arange(len(cats))
    height = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(cats) + 1.4), facecolor=SURFACE)
    _style(ax)
    for offset, key, scores in ((height / 2, "llm", llm), (-height / 2, "baseline", base)):
        label = llm_name if (key == "llm" and llm_name) else NAMES[key]
        ax.barh(y + offset, [scores[c]["f1"] for c in cats], height=height - 0.04,
                color=SERIES[key], label=label, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, [CATEGORIES[c]["label_en"] for c in cats], color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("F1 score", color=INK_SECONDARY, fontsize=9)
    ax.set_title("F1 by complaint category", loc="left", color=INK, fontsize=12, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
              labelcolor=INK_SECONDARY, handlelength=1, borderaxespad=0.2)
    fig.tight_layout()
    return fig


def confusion_heatmap(metrics: dict, title: str):
    """Row-normalised confusion matrix: each row shows where one true category's complaints went."""
    labels = metrics["labels"]
    counts = np.array(metrics["confusion_matrix"], dtype=float)
    rows = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)
    names = [CATEGORIES[c]["label_en"] for c in labels]

    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=SURFACE)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j]:
                color = "#ffffff" if share[i, j] > 0.55 else INK
                ax.text(j, i, int(counts[i, j]), ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(labels)), names, rotation=35, ha="right", color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(range(len(labels)), names, color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("Predicted", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Actual", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    fig.tight_layout()
    return fig
