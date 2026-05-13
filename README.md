# GUI Grounding Benchmark

Воспроизводимое сравнение визуально-языковых моделей (VLM) на задаче локализации UI-элементов веб-страниц по текстовой инструкции. Постановка соответствует ClickAcc-протоколу из SeeClick (Cheng et al., ACL 2024).

Сопоставляет 11 моделей в 4 категориях на трёх независимых бенчмарках.

## Бенчмарки

| Бенчмарк | Источник | Размер | Назначение |
|---|---|---|---|
| **ScreenSpot-V2** | `lmms-lab/ScreenSpot-v2` | 437 web (1272 full) | Стандартное сравнение с literature |
| **WebClick** | `Hcompany/WebClick` | 1639 (3 bucket'а) | Independent web-grounding бенчмарк |
| **Khabner test** | `Khabner/moondream-data` | 2183 (10 element types) | In-domain held-out (production-style UI) |

## Сравниваемые модели

| Категория | Модель | Параметры |
|---|---|---|
| Closed API | GPT-4o, Claude Sonnet 4.6, Gemini 2.5 Pro | – |
| Open generalist | Qwen2-VL-7B, Moondream2, Florence-2-base | 0.27–7B |
| Open GUI-specialist | OS-Atlas-Base-7B, SeeClick | 7–9.6B |
| **Ours (LoRA)** | **Moondream2 + LoRA, Florence-2-base + LoRA, Florence-2-large + LoRA** | **270M, 770M, 1.86B** |

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env с API-ключами и путями к LoRA-чекпоинтам
```

## Запуск

```bash
# ScreenSpot-V2 (web subset из 437 сэмплов)
python -m benchmark.run_benchmark --models all --concurrency 8

# WebClick (1639 сэмплов, 3 bucket'а)
python eval_webclick.py --model all --concurrency 8

# Khabner test (in-domain, 2183 сэмплов, стратификация по element_type)
python eval_khabner.py --model all --concurrency 8

# Resume (пропустить модели у которых уже есть jsonl)
python -m benchmark.run_benchmark --models all --resume

# Quick smoke (10 сэмплов одной моделью)
python -m benchmark.run_benchmark --models moondream2_lora --max-samples 10

# Concurrency=N — для API моделей (8 параллельных запросов ~7× ускоряет)
# Concurrency=1 — обязательно для GPU моделей (память не делится между потоками)
```

## Структура результатов

```
results/
├── raw/{model}.jsonl              # per-sample predictions ScreenSpot-V2
├── webclick_raw/{model}.jsonl     # per-sample WebClick
├── khabner_raw/{model}.jsonl      # per-sample Khabner test
├── benchmark_table.{md,csv}       # сводная таблица V2
├── webclick_table.{md,csv}        # сводная WebClick
└── khabner_table.{md,csv}         # сводная Khabner
```

Каждая jsonl-строка: `{idx, instruction, pred: [x, y] | null, gt_bbox, ...meta..., latency_s}`

## Метрика — ClickAcc

Предсказание считается корректным, если предсказанная точка `(x, y)` лежит внутри ground-truth bbox `[x_min, y_min, x_max, y_max]` в нормализованных координатах `[0, 1]`.

Стратификация:
- ScreenSpot-V2: `text` / `icon`, по платформам (web/mobile/desktop), по `data_source` (8 категорий)
- WebClick: `agentbrowse` / `humanbrowse` / `calendars`
- Khabner test: `element_type` (button / input / link / checkbox / heading / menu_item / ...), `site_name` (40 типов сайтов)

Failed predictions (None) считаются как miss.

## Воспроизводимость

- Seed=0 (numpy, torch, random)
- Per-sample raw jsonl → метрики пересчитываются без повторной инференции (`--resume`)
- Промпты для каждой модели в адаптерах (`benchmark/models/*.py`), процитированы в docstring со ссылкой на источник
- `transformers==4.46.3` (Florence-2 ломается на 4.52+)

## Тесты парсеров

```bash
pytest benchmark/parsing/tests/
```

## Адаптеры моделей

Каждая модель в `benchmark/models/{model_key}.py` имеет:
- `predict(image, instruction) → (x, y) | None` — единый интерфейс
- Docstring с цитатой prompt-template'а из первоисточника (paper / official repo)
- Обработку специфики модели (image resize для Claude, max_pixels для Qwen2-VL, и т.д.)

## Структура проекта

```
benchmark/
├── run_benchmark.py            # orchestrator для ScreenSpot-V2
├── plot_results.py             # генерация benchmark_plot.png
├── screenspot_loader.py        # ScreenSpot-V2 dataset loader
├── webclick_loader.py          # WebClick loader
├── khabner_loader.py           # Khabner test loader
├── models/                     # адаптеры (один файл на модель)
├── metrics/click_accuracy.py   # ClickAcc реализация
└── parsing/coordinate_parsers.py  # парсеры координат разных моделей
eval_webclick.py                # orchestrator для WebClick (с concurrency)
eval_khabner.py                 # orchestrator для Khabner test
SBAS_Khabner_results.md         # сводный документ с результатами для постера
```

## Ссылки

- SeeClick paper (ScreenSpot V1): [arXiv:2401.10935](https://arxiv.org/abs/2401.10935)
- ScreenSpot-V2 dataset: [lmms-lab/ScreenSpot-v2](https://huggingface.co/datasets/lmms-lab/ScreenSpot-v2)
- WebClick dataset: [Hcompany/WebClick](https://huggingface.co/datasets/Hcompany/WebClick)
- Training dataset: [Khabner/moondream-data](https://huggingface.co/datasets/Khabner/moondream-data)
- LoRA paper: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
