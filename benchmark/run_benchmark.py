"""
Benchmark orchestrator.

Runs all specified models on ScreenSpot-Web, saves raw predictions to
results/raw/{model_name}.jsonl, generates results/benchmark_table.{md,csv}.

Usage:
    python -m benchmark.run_benchmark --models all
    python -m benchmark.run_benchmark --models moondream2_lora florence2_lora
    python -m benchmark.run_benchmark --resume
"""

import argparse
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

SEED = 0

MODELS_REGISTRY = {
    "gpt4o":           ("benchmark.models.gpt4o",           "GPT4o"),
    "claude_sonnet":   ("benchmark.models.claude_sonnet",   "ClaudeSonnet"),
    "qwen2_vl":        ("benchmark.models.qwen2_vl",        "Qwen2VL"),
    "moondream2_base": ("benchmark.models.moondream2_base", "Moondream2Base"),
    "moondream2_lora": ("benchmark.models.moondream2_lora", "Moondream2LoRA"),
    "florence2_base":  ("benchmark.models.florence2_base",  "Florence2Base"),
    "florence2_lora":  ("benchmark.models.florence2_lora",  "Florence2LoRA"),
    "seeclick":        ("benchmark.models.seeclick",        "SeeClick"),
    "os_atlas":        ("benchmark.models.os_atlas",        "OSAtlas"),
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_model(model_key: str, samples, results_dir: Path, resume: bool) -> dict:
    raise NotImplementedError("Wire up in implementation step")


def build_table(results: Dict[str, dict], out_dir: Path):
    raise NotImplementedError("Wire up in implementation step")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["all"],
                   help="Model keys or 'all'. Keys: " + ", ".join(MODELS_REGISTRY))
    p.add_argument("--resume", action="store_true",
                   help="Skip models with existing raw predictions")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--results-dir", type=str, default="results", dest="results_dir")
    args = p.parse_args()

    set_seed(SEED)

    keys = list(MODELS_REGISTRY) if args.models == ["all"] else args.models
    results_dir = Path(args.results_dir)
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)

    raise NotImplementedError("Wire up in implementation step")


if __name__ == "__main__":
    main()
