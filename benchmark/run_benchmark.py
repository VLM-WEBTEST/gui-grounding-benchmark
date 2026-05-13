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
import gc
import importlib
import json
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from benchmark.metrics.click_accuracy import click_accuracy
from benchmark.screenspot_loader import ScreenSpotSample, ScreenSpotWeb

SEED = 0

MODELS_REGISTRY = {
    "gpt4o":           ("benchmark.models.gpt4o",           "GPT4o"),
    "claude_sonnet":   ("benchmark.models.claude_sonnet",   "ClaudeSonnet"),
    "gemini":          ("benchmark.models.gemini",          "Gemini"),
    "qwen2_vl":        ("benchmark.models.qwen2_vl",        "Qwen2VL"),
    "moondream2_base": ("benchmark.models.moondream2_base", "Moondream2Base"),
    "moondream2_lora": ("benchmark.models.moondream2_lora", "Moondream2LoRA"),
    "florence2_base":  ("benchmark.models.florence2_base",  "Florence2Base"),
    "florence2_lora":  ("benchmark.models.florence2_lora",  "Florence2LoRA"),
    "florence2_large_lora": ("benchmark.models.florence2_large_lora", "Florence2LargeLoRA"),
    "florence2_base_v1_lora": ("benchmark.models.florence2_base_v1_lora", "Florence2BaseV1LoRA"),
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


def _get_model_class(model_key: str):
    module_path, class_name = MODELS_REGISTRY[model_key]
    return getattr(importlib.import_module(module_path), class_name)


def _read_prior(raw_path: Path) -> Dict[int, dict]:
    """Read any prior predictions from a model's jsonl file, keyed by sample idx."""
    if not raw_path.exists():
        return {}
    prior: Dict[int, dict] = {}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prior[row["idx"]] = row
    return prior


def _free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_model(
    model_key: str,
    samples: List[ScreenSpotSample],
    results_dir: Path,
    resume: bool,
    errors_log,
    concurrency: int = 1,
) -> dict:
    cls = _get_model_class(model_key)
    meta = {
        "key":      model_key,
        "name":     cls.name,
        "category": cls.category,
        "params":   cls.params,
    }

    raw_path = results_dir / "raw" / f"{model_key}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_prior(raw_path) if resume else {}

    predictions: List[Optional[tuple]] = [None] * len(samples)
    latencies: List[float] = [0.0] * len(samples)
    bboxes = [s.bbox for s in samples]
    types = [s.data_type for s in samples]

    for i, row in prior.items():
        if i < len(samples):
            predictions[i] = tuple(row["pred"]) if row["pred"] is not None else None
            latencies[i] = float(row.get("latency_s", 0.0))

    todo = [(i, s) for i, s in enumerate(samples) if i not in prior]
    model = None
    if todo:
        print(f"  [{model_key}] loading model (concurrency={concurrency}) ...", flush=True)
        model = cls()

    write_lock = threading.Lock()
    open_mode = "a" if resume and prior else "w"

    def _predict(i: int, s: ScreenSpotSample):
        try:
            t0 = time.perf_counter()
            pred = model.predict(s.image, s.instruction)
            dt = time.perf_counter() - t0
        except Exception as e:
            with write_lock:
                errors_log.write(
                    f"[{model_key}][idx={i}] predict error: {e}\n"
                    + traceback.format_exc() + "\n"
                )
                errors_log.flush()
            pred, dt = None, 0.0
        return i, s, pred, dt

    with open(raw_path, open_mode) as f:
        if concurrency <= 1:
            iterator = (_predict(i, s) for i, s in todo)
        else:
            pool = ThreadPoolExecutor(max_workers=concurrency)
            futures = [pool.submit(_predict, i, s) for i, s in todo]
            iterator = (fut.result() for fut in as_completed(futures))

        for i, s, pred, dt in tqdm(iterator, total=len(todo), desc=model_key, unit="sample"):
            predictions[i] = pred
            latencies[i] = dt
            row = {
                "idx": i,
                "instruction": s.instruction,
                "pred": list(pred) if pred is not None else None,
                "gt_bbox": list(s.bbox),
                "data_type": s.data_type,
                "latency_s": dt,
            }
            with write_lock:
                f.write(json.dumps(row) + "\n")
                f.flush()

        if concurrency > 1:
            pool.shutdown(wait=True)

    # Release GPU memory before the next model.
    del model
    _free_cuda()

    metrics = click_accuracy(predictions, bboxes, types)
    metrics["avg_latency_s"] = (
        float(np.mean([x for x in latencies if x > 0])) if any(latencies) else 0.0
    )
    return {**meta, **metrics}


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{100 * x:.1f}"


def build_table(results: Dict[str, dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = [
        "Model", "Category", "Params",
        "Overall", "Text", "Icon",
        "n", "n_failed", "avg_latency_s",
    ]

    rows = []
    for key, r in results.items():
        if r is None or "error" in r:
            rows.append([
                key, "error", "—",
                "N/A", "N/A", "N/A",
                "—", "—", "—",
            ])
            continue
        rows.append([
            r["name"],
            r["category"],
            r.get("params", "—"),
            _fmt_pct(r["overall"]),
            _fmt_pct(r["text"]),
            _fmt_pct(r["icon"]),
            str(r.get("n_overall", "—")),
            str(r.get("n_failed", 0)),
            f"{r.get('avg_latency_s', 0):.2f}",
        ])

    csv_path = out_dir / "benchmark_table.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")

    md_path = out_dir / "benchmark_table.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join("---" for _ in headers) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")

    return csv_path, md_path


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["all"],
                   help="Model keys or 'all'. Keys: " + ", ".join(MODELS_REGISTRY))
    p.add_argument("--resume", action="store_true",
                   help="Skip samples with existing raw predictions")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--results-dir", type=str, default="results", dest="results_dir")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel API workers. Use 5-10 for OpenAI/Anthropic, 1 for GPU models.")
    args = p.parse_args()

    set_seed(SEED)

    keys = list(MODELS_REGISTRY) if args.models == ["all"] else args.models
    for k in keys:
        if k not in MODELS_REGISTRY:
            raise SystemExit(
                f"Unknown model: {k!r}. Valid keys: {list(MODELS_REGISTRY)}"
            )

    results_dir = Path(args.results_dir)
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)

    print("Loading ScreenSpot-Web ...")
    ds = ScreenSpotWeb()
    samples = list(ds)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"  {len(samples)} samples")

    results: Dict[str, dict] = {}
    errors_log_path = results_dir / "errors.log"
    with open(errors_log_path, "a") as errlog:
        errlog.write(
            f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"models={keys}  resume={args.resume}  n={len(samples)} ===\n"
        )
        for key in keys:
            print(f"\n--- {key} ---")
            try:
                results[key] = run_model(
                    key, samples, results_dir, args.resume, errlog,
                    concurrency=args.concurrency,
                )
                r = results[key]
                print(
                    f"  overall={_fmt_pct(r['overall'])}  "
                    f"text={_fmt_pct(r['text'])}  "
                    f"icon={_fmt_pct(r['icon'])}  "
                    f"failed={r['n_failed']}/{r['n_overall']}  "
                    f"latency={r['avg_latency_s']:.2f}s"
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                errlog.write(
                    f"[{key}] model load/run failed: {e}\n"
                    + traceback.format_exc() + "\n"
                )
                errlog.flush()
                results[key] = {"key": key, "error": str(e)}
            _free_cuda()

    csv_path, md_path = build_table(results, results_dir)
    print(f"\nResults written to:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
