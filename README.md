# T04 Team Project 3 — Natural Language Processing with Disaster Tweets

Kaggle team: `CAU53744-T04`

## Setup

Put `train.csv`, `test.csv`, `sample_submission.csv` from
https://www.kaggle.com/competitions/nlp-getting-started/data into `data/`.

## Reproduce

```bash
python src/read_samples.py   # Step 1: 40 random tweets to read before any model code
python src/main.py           # duplicates, split, baseline, models A/B, submission.csv, results.json
```

`splits/split.json` holds the one 80/20 split (ids). `main.py` creates it only if it is missing and loads it otherwise.

## Code layout

| File | Role |
|---|---|
| `src/config.py` | seed 42, split ratio, team info, public F1 scores |
| `src/data.py` | loading, `clean_text()` shared by every model, duplicate handling, the one split |
| `src/baseline.py` | Step 2 TF-IDF + logistic regression |
| `src/models.py` | Step 3 model A and model B |
| `src/main.py` | runs everything and writes `submission.csv` and `results.json` |

## Environment

- Python 3.13.3
- numpy 2.2.4
- pandas 2.2.3
- scikit-learn 1.9.1
- Seed 42 everywhere

## Step 7 (relabelling with LLM labels)

`step7/llm_labels.csv` (no header: `id,label,confidence`) holds first-pass LLM labels for the training part only.
`python src/main.py` reads it, replaces the training label where the LLM disagrees with confidence 3
(`RELABEL_MIN_CONFS` in `src/config.py`), retrains the baseline, model A and model B, and writes the validation F1
against the unchanged validation labels to `results.json` under `step7`. If the file is missing the step is skipped.
