"""
Evaluate a model from MODELS_REGISTRY on WebClick (Hcompany/WebClick).

Usage:
    python eval_webclick.py --model moondream2_base
    python eval_webclick.py --model moondream2_lora --resume
    python eval_webclick.py --model all

Writes results/webclick_raw/{model}.jsonl + results/webclick_table.md.
"""

import argparse
import gc
import importlib
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from benchmark.metrics.click_accuracy import point_in_bbox
from benchmark.run_benchmark import MODELS_REGISTRY, set_seed
from benchmark.webclick_loader import WebClick, WebClickSample


def _load_cls(model_key: str):
    mod_path, cls_name = MODELS_REGISTRY[model_key]
    return getattr(importlib.import_module(mod_path), cls_name)


def _read_prior(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out[r["idx"]] = r
            except json.JSONDecodeError:
                continue
    return out


def _free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_one(
    model_key: str, samples: List[WebClickSample], results_dir: Path, resume: bool,
    errlog, concurrency: int = 1,
) -> dict:
    cls = _load_cls(model_key)
    meta = {"key": model_key, "name": cls.name, "category": cls.category, "params": cls.params}

    raw_path = results_dir / "webclick_raw" / f"{model_key}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    prior = _read_prior(raw_path) if resume else {}

    predictions: List[Optional[tuple]] = [None] * len(samples)
    latencies = [0.0] * len(samples)
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

    def _predict(i: int, s: WebClickSample):
        try:
            t0 = time.perf_counter()
            pred = model.predict(s.image, s.instruction)
            dt = time.perf_counter() - t0
        except Exception as e:
            with write_lock:
                errlog.write(
                    f"[{model_key}][idx={i}] predict error: {e}\n"
                    + traceback.format_exc() + "\n"
                )
                errlog.flush()
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
            with write_lock:
                f.write(json.dumps({
                    "idx": i,
                    "instruction": s.instruction,
                    "pred": list(pred) if pred is not None else None,
                    "gt_bbox": list(s.bbox),
                    "bucket": s.bucket,
                    "latency_s": dt,
                }) + "\n")
                f.flush()

        if concurrency > 1:
            pool.shutdown(wait=True)

    del model
    _free_cuda()

    # Metrics
    hits = [predictions[i] is not None and point_in_bbox(predictions[i], s.bbox)
            for i, s in enumerate(samples)]
    bucket_idx: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        bucket_idx.setdefault(s.bucket, []).append(i)

    def rate(ids):
        return sum(hits[i] for i in ids) / len(ids) if ids else float("nan")

    result = {
        **meta,
        "overall": float(np.mean(hits)) if hits else float("nan"),
        "n_overall": len(samples),
        "n_failed": sum(1 for p in predictions if p is None),
        "avg_latency_s": float(np.mean([x for x in latencies if x > 0])) if any(latencies) else 0.0,
    }
    for b, ids in bucket_idx.items():
        result[f"{b}"] = rate(ids)
        result[f"n_{b}"] = len(ids)
    return result


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{100 * x:.1f}"


def build_table(results: Dict[str, dict], out_dir: Path, buckets: List[str]):
    headers = ["Model", "Category", "Params", "Overall", *buckets, "n", "n_failed", "avg_latency_s"]
    rows = []
    for key, r in results.items():
        if "error" in r:
            rows.append([key, "error", "—", "N/A", *["N/A"]*len(buckets), "—", "—", "—"])
            continue
        bucket_cells = [fmt_pct(r.get(b)) for b in buckets]
        rows.append([
            r["name"], r["category"], r.get("params", "—"),
            fmt_pct(r["overall"]), *bucket_cells,
            str(r.get("n_overall", "—")),
            str(r.get("n_failed", 0)),
            f"{r.get('avg_latency_s', 0):.2f}",
        ])

    (out_dir / "webclick_table.csv").write_text(
        ",".join(headers) + "\n" + "\n".join(",".join(str(c) for c in r) for r in rows) + "\n"
    )
    (out_dir / "webclick_table.md").write_text(
        "| " + " | ".join(headers) + " |\n" +
        "|" + "|".join("---" for _ in headers) + "|\n" +
        "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"
    )


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=["moondream2_base", "moondream2_lora"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--results-dir", type=str, default="results", dest="results_dir")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel API workers. Use 5-10 for OpenAI/Anthropic, 1 for GPU models.")
    args = p.parse_args()

    set_seed(0)

    keys = list(MODELS_REGISTRY) if args.model == ["all"] else args.model
    for k in keys:
        if k not in MODELS_REGISTRY:
            raise SystemExit(f"Unknown model: {k!r}. Valid: {list(MODELS_REGISTRY)}")

    results_dir = Path(args.results_dir)
    (results_dir / "webclick_raw").mkdir(parents=True, exist_ok=True)

    print("Loading WebClick ...")
    ds = WebClick()
    samples = list(ds)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"  {len(samples)} samples")

    buckets = sorted({s.bucket for s in samples})
    print(f"  buckets: {buckets}")

    results: Dict[str, dict] = {}
    with open(results_dir / "webclick_errors.log", "a") as errlog:
        errlog.write(
            f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"models={keys}  n={len(samples)} ===\n"
        )
        for key in keys:
            print(f"\n--- {key} ---")
            try:
                results[key] = run_one(
                    key, samples, results_dir, args.resume, errlog,
                    concurrency=args.concurrency,
                )
                r = results[key]
                parts = [f"overall={fmt_pct(r['overall'])}"]
                for b in buckets:
                    parts.append(f"{b}={fmt_pct(r.get(b))}")
                parts.append(f"failed={r['n_failed']}/{r['n_overall']}")
                parts.append(f"latency={r['avg_latency_s']:.2f}s")
                print("  " + "  ".join(parts))
            except Exception as e:
                print(f"  ERROR: {e}")
                errlog.write(f"[{key}] model load/run failed: {e}\n" + traceback.format_exc() + "\n")
                errlog.flush()
                results[key] = {"key": key, "name": key, "category": "error", "error": str(e)}
            _free_cuda()

    build_table(results, results_dir, buckets)
    print(f"\nResults → {results_dir}/webclick_table.md")


if __name__ == "__main__":
    main()
