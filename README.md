# GUI Grounding Benchmark

Reproducible evaluation of GUI grounding models on ScreenSpot-Web (Cheng et al., ACL 2024).

Compares 9 models across 4 categories on the standard **ClickAcc** metric (predicted click inside ground-truth bbox).

## Models

| Category | Model | Params |
|---|---|---|
| Closed API | GPT-4o | N/A |
| Closed API | Claude 3.5 Sonnet | N/A |
| Open generalist | Qwen2-VL-7B | 7B |
| Open generalist | Moondream2 (base) | 1.86B |
| Open generalist | Florence-2 (base) | 270M |
| Open GUI-specialist | SeeClick | 9.6B |
| Open GUI-specialist | OS-Atlas-Base-7B | 7B |
| **Ours** | **Moondream2 + LoRA** | **1.86B** |
| **Ours** | **Florence-2 + LoRA** | **270M** |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your keys and checkpoint paths
```

## Usage

```bash
# All models
python -m benchmark.run_benchmark --models all

# Specific models
python -m benchmark.run_benchmark --models moondream2_lora florence2_lora

# Resume (skip models with existing raw predictions)
python -m benchmark.run_benchmark --resume

# Smoke test
python -m benchmark.run_benchmark --models moondream2_base --max-samples 10

# Build plot after benchmark
python -m benchmark.plot_results
```

## Outputs

```
results/
├── raw/{model}.jsonl         # per-sample predictions (for reproducibility)
├── benchmark_table.md        # for paper
├── benchmark_table.csv
├── benchmark_plot.svg        # for poster
└── errors.log
```

## Metric: ClickAcc

A prediction is correct iff the predicted point (x, y) falls inside the ground-truth bounding box.

Reported separately for:
- **Text** elements (menu items, buttons with text labels)
- **Icon/Widget** elements (icons, widgets without text)
- **Overall** (combined)

Reference: Cheng et al., "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents", ACL 2024.

## Tests

```bash
pytest benchmark/parsing/tests/
```

## Reproducibility

- Seed fixed at 0 (numpy, torch, random)
- Raw predictions saved per sample → re-run metrics without inference
- Pinned dependency versions
- Per-model prompts copied verbatim from primary sources (cited in each adapter's docstring)
