# LLM-Confidence-Vs.-Accuracy
# Do LLMs Know When They're Wrong?

**Measuring Calibration & Overconfidence in Large Language Models Across Question Categories — a BoolQ Benchmark Study**

Authors: Sahasra Chinthireddy, Zoran Tiganj
Indiana University Bloomington — Kelley School of Business / Luddy School of Computing, Engineering & Informatics / CEWIT

## Overview

Large language models are increasingly deployed in medicine, law, and finance, where knowing *when to trust the model* matters as much as the answer itself. This project investigates whether LLM confidence signals — both self-reported ("how confident are you, 0–100?") and token-level (LogProb entropy) — actually predict correctness on a standard yes/no QA benchmark.

We evaluate 10 model conditions across 5 model families on 100 BoolQ questions per run (1,700 total query–response pairs), then compute accuracy, mean self-reported confidence, overconfidence delta, and Expected Calibration Error (ECE) for each.

**Headline finding:** every model tested reports higher confidence than its accuracy warrants (overconfidence delta of +2 to +22 percentage points), and raw LogProb entropy is a weak predictor of correctness (r ≈ 0.11, p ≈ 0.29).

## Files in this repository

| File | Description |
| --- | --- |
| `run_experiment.py` | Main experiment script. Loads BoolQ, queries each model for an answer and a confidence estimate, and writes raw/aggregated results plus `poster_data.json` for the poster. |
| `boolq_100_questions.csv` | The 100-question stratified sample used as the evaluation set (question, gold answer, supporting passage). |
| `poster_data.csv` | Aggregated results used to populate the poster — per-model accuracy/confidence/ECE, category breakdowns, calibration-curve points, and the overconfidence histogram. |
| `poster.html` | Self-contained 1440×1080 research poster rendering abstract, methodology, charts, and findings. Open in any browser. |

## Methodology (summary)

1. **Dataset** — BoolQ (Clark et al., 2019), 100-question stratified sample covering geography, science, history, biology, law, and pop culture.
2. **Models** — GPT-4o, GPT-4o-mini (×5 runs each), GPT-3.5-turbo (×3), Claude 3 Haiku (×2), Llama-3-8B (×2).
3. **Answer prompt** — model reads the passage and answers "Yes" or "No" (temperature 0, max 10 tokens).
4. **Confidence prompt** — a second call asks the model to rate its confidence 0–100 on the answer it just gave.
5. **LogProb signal** — for OpenAI models, entropy of the top-5 tokens at the answer position is recorded.
6. **Metrics** — accuracy, mean self-reported confidence, overconfidence delta (confidence − accuracy), and ECE with 10 equal-width bins for both the self-reported and LogProb signals.

## Running the experiment

Requirements:

```bash
pip install openai anthropic datasets pandas numpy tqdm
```

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Run:

```bash
python run_experiment.py
```

Outputs (written to `./results/`):

- `results_raw.csv` — every question × model × run response.
- `results_summary.csv` — per-model aggregated stats (accuracy, confidence, ECE, correlations).
- `results_category.csv` — per-category breakdown.
- `poster_data.json` — drop-in replacement for the `DATA` object in `poster.html`.

If HuggingFace is unreachable, the script falls back to a baked-in 100-question sample embedded at the bottom of `run_experiment.py`.

## Key results

| Model | Accuracy | Self-confidence | Δ | ECE (self) | ECE (LogProb) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-4o | 91% | 93% | +2 | 0.018 | 0.071 |
| GPT-4o-mini | 86% | 88.2% | +2.2 | 0.051 | 0.119 |
| Claude 3 Haiku | 83% | 87% | +4 | 0.044 | 0.109 |
| GPT-3.5-turbo | 79% | 92% | +13 | 0.098 | 0.152 |
| Llama-3-8B | 72% | 94% | +22 | 0.134 | 0.198 |

Smaller and older models are both *less accurate* and *more overconfident*. Across all models, self-reported confidence is better calibrated than LogProb entropy. Miscalibration is category-dependent — for GPT-4o-mini, Law shows the largest gap (+17) while Pop Culture shows the smallest (+4).

## Viewing the poster

Open `poster.html` directly in a browser. It is a single standalone HTML file sized to 1440×1080 (print-ready 16:9).

## References

- Clark et al. (2019). BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. *NAACL*.
- Guo et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
- Kadavath et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.
- Xiong et al. (2023). Can LLMs Express Their Uncertainty? *arXiv:2306.13063*.

## Acknowledgments

IU CEWIT, Luddy School of Computing, Kelley School of Business.
