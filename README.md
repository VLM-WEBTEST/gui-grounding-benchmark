# GUI Grounding Benchmark

🇬🇧 README in English · 🇷🇺 [README на русском](README_RU.md)

---

Reproducible comparison of vision-language models (VLMs) on the task of localizing UI elements on web-page screenshots from natural-language instructions. The evaluation protocol follows ClickAcc from SeeClick (Cheng et al., ACL 2024).

Compares 11 models in 4 categories across three independent benchmarks.

## Benchmarks

| Benchmark | Source | Size | Purpose |
|---|---|---|---|
| **ScreenSpot-V2** | `lmms-lab/ScreenSpot-v2` | 437 web (1272 full) | Standard comparison with literature |
| **WebClick** | `Hcompany/WebClick` | 1639 (3 buckets) | Independent web-grounding benchmark |
| **Khabner test** | `Khabner/moondream-data` | 2183 (10 element types) | In-domain held-out (production-style UI) |

## Models compared

| Category | Model | Parameters |
|---|---|---|
| Closed API | GPT-4o, Claude Sonnet 4.6, Gemini 2.5 Pro | – |
| Open generalist | Qwen2-VL-7B, Moondream2, Florence-2-base | 0.23–7B |
| Open GUI specialist | OS-Atlas-Base-7B, SeeClick | 7–9.6B |
| **Ours (LoRA)** | **Moondream2 + LoRA, Florence-2-base + LoRA, Florence-2-large + LoRA** | **1.86B, 232M, 770M** |

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with API keys and paths to LoRA checkpoints
```

## Running

```bash
# ScreenSpot-V2 (web subset, 437 samples)
python -m benchmark.run_benchmark --models all --concurrency 8

# WebClick (1639 samples, 3 buckets)
python eval_webclick.py --model all --concurrency 8

# Khabner test (in-domain, 2183 samples, stratified by element_type)
python eval_khabner.py --model all --concurrency 8

# Resume (skip models that already have a jsonl)
python -m benchmark.run_benchmark --models all --resume

# Quick smoke test (10 samples, single model)
python -m benchmark.run_benchmark --models moondream2_lora --max-samples 10

# Concurrency=N — for API models (8 parallel requests ≈ 7× speedup)
# Concurrency=1 — required for GPU models (memory isn't shared between threads)
```

## Results layout

```
results/
├── raw/{model}.jsonl              # per-sample predictions, ScreenSpot-V2
├── webclick_raw/{model}.jsonl     # per-sample WebClick
├── khabner_raw/{model}.jsonl      # per-sample Khabner test
├── benchmark_table.{md,csv}       # summary table, V2
├── webclick_table.{md,csv}        # summary WebClick
└── khabner_table.{md,csv}         # summary Khabner
```

Each jsonl row: `{idx, instruction, pred: [x, y] | null, gt_bbox, ...meta..., latency_s}`.

## Metric — ClickAcc

A prediction counts as correct if the predicted point `(x, y)` falls inside the ground-truth bounding box `[x_min, y_min, x_max, y_max]` in normalized `[0, 1]` coordinates.

Stratification:
- ScreenSpot-V2: `text` / `icon`, by platform (web/mobile/desktop), by `data_source` (8 categories)
- WebClick: `agentbrowse` / `humanbrowse` / `calendars`
- Khabner test: `element_type` (button / input / link / checkbox / heading / menu_item / ...), `site_name` (40 site categories)

Failed predictions (None) count as misses.

## Reproducibility

- Seed = 0 (numpy, torch, random)
- Per-sample raw jsonl → metrics recomputed without re-running inference (`--resume`)
- Prompts for each model are in the adapter file (`benchmark/models/*.py`), quoted in the docstring with a link to the source
- `transformers==4.46.3` (Florence-2 breaks on 4.52+)

## Parser tests

```bash
pytest benchmark/parsing/tests/
```

## Model adapters

Every model in `benchmark/models/{model_key}.py` provides:
- `predict(image, instruction) → (x, y) | None` — unified interface
- A docstring quoting the prompt template from the original source (paper / official repo)
- Model-specific handling (image resize for Claude, max_pixels for Qwen2-VL, etc.)

## Project layout

```
benchmark/
├── run_benchmark.py            # orchestrator for ScreenSpot-V2
├── plot_results.py             # benchmark_plot.png generator
├── screenspot_loader.py        # ScreenSpot-V2 dataset loader
├── webclick_loader.py          # WebClick loader
├── khabner_loader.py           # Khabner test loader
├── models/                     # adapters (one file per model)
├── metrics/click_accuracy.py   # ClickAcc implementation
└── parsing/coordinate_parsers.py  # output parsers for each model
eval_webclick.py                # orchestrator for WebClick (with concurrency)
eval_khabner.py                 # orchestrator for Khabner test
```

## References

- SeeClick paper (ScreenSpot V1): [arXiv:2401.10935](https://arxiv.org/abs/2401.10935)
- ScreenSpot-V2 dataset: [lmms-lab/ScreenSpot-v2](https://huggingface.co/datasets/lmms-lab/ScreenSpot-v2)
- WebClick dataset: [Hcompany/WebClick](https://huggingface.co/datasets/Hcompany/WebClick)
- Training dataset: [Khabner/moondream-data](https://huggingface.co/datasets/Khabner/moondream-data)
- LoRA paper: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

## License

[MIT](LICENSE)
