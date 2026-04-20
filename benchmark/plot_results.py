"""
Generate benchmark plot from results/benchmark_table.csv.

Scatter: x = params (log scale), y = ClickAcc-Overall.
Colored by category. Ours highlighted with star marker.
Closed APIs as horizontal lines (params unknown).

Output: results/benchmark_plot.svg (vector, Arial 11pt, printable in grayscale).
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results", dest="results_dir")
    args = p.parse_args()

    raise NotImplementedError("Wire up in implementation step")


if __name__ == "__main__":
    main()
