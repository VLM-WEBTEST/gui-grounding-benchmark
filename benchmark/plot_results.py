"""
Plot benchmark results as a horizontal bar chart.

Reads results/benchmark_table.csv (or --csv path) and saves a PNG.

Usage:
    python -m benchmark.plot_results
    python -m benchmark.plot_results --csv other/table.csv --out other/plot.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

CATEGORY_COLORS = {
    "closed_api":           "#9c6ade",
    "open_generalist":      "#4a7abf",
    "open_gui_specialist":  "#e28d2a",
    "ours":                 "#d9534f",
    "error":                "#999999",
}


def _parse_pct(s: str):
    if s in ("N/A", "—", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["Overall_f"] = _parse_pct(r.get("Overall", ""))
            r["Text_f"]    = _parse_pct(r.get("Text", ""))
            r["Icon_f"]    = _parse_pct(r.get("Icon", ""))
            rows.append(r)
    return rows


def plot_bars(rows, out_path: Path):
    # Highest Overall first; N/A at the bottom.
    def sort_key(r):
        return -(r["Overall_f"] if r["Overall_f"] is not None else -1)
    rows = sorted(rows, key=sort_key)

    labels = [r["Model"] for r in rows]
    vals   = [r["Overall_f"] or 0 for r in rows]
    colors = [CATEGORY_COLORS.get(r["Category"], "#999") for r in rows]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.5))
    bars = ax.barh(labels, vals, color=colors)

    for bar, r in zip(bars, rows):
        if r["Overall_f"] is None:
            txt = "N/A"
        else:
            text_part = (
                f" (T {r['Text_f']:.0f} / I {r['Icon_f']:.0f})"
                if r["Text_f"] is not None else ""
            )
            txt = f"{r['Overall_f']:.1f}%{text_part}"
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("ClickAcc Overall (%)")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("ScreenSpot-Web ClickAcc")

    present = [c for c in CATEGORY_COLORS if any(r["Category"] == c for r in rows)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c]) for c in present]
    ax.legend(handles, present, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/benchmark_table.csv")
    p.add_argument("--out", default="results/benchmark_plot.png")
    args = p.parse_args()

    rows = load_rows(Path(args.csv))
    plot_bars(rows, Path(args.out))


if __name__ == "__main__":
    main()
