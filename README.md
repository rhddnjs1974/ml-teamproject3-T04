# T04 Team Project 3 — Natural Language Processing with Disaster Tweets

Kaggle team: `CAU53744-T04`

## Setup

Put `train.csv`, `test.csv`, `sample_submission.csv` from
https://www.kaggle.com/competitions/nlp-getting-started/data into `data/`.

## Reproduce

```bash
python src/read_samples.py   # Step 1: 40 random tweets to read before any model code
python src/main.py           # every step below in one run; writes submission.csv, results.json, figures/
```

The reported run used a Google Colab T4 GPU (models B and C need a GPU to finish inside 60 minutes).
On another GPU the last digits of models A, B and C may differ.

### Files the run reads if they exist

| File | Made by | If it is missing |
|---|---|---|
| `splits/split.json` | the first run (80/20, stratified, seed 42) | recreated; the split is deterministic, so it comes out the same |
| `step4/errors_30.csv` | the first run (15 FP + 15 FN of model B, seed 42) | a new 30 is drawn; on another GPU it may differ from the 30 we read |
| `step4/categories.csv` | the team, by hand | Step 4 category counts, Figure 2 and the Step 5 estimate are skipped |
| `step7/llm_labels.csv` | first-pass LLM labels, training part only | the Step 7 relabelling experiment is skipped |

The submitted zip holds only `src/`, `report.pdf`, `results.json`, `README.md`, `submission.csv`;
`results.json` keeps every number from the reported run, including the steps above.

## Code layout

| File | Role |
|---|---|
| `src/config.py` | seed 42, split ratio, team info, public F1 scores |
| `src/data.py` | loading, `clean_text()` shared by every model, duplicate handling, the one split |
| `src/baseline.py` | Step 2 TF-IDF + logistic regression |
| `src/read_samples.py` | Step 1: prints 40 random tweets to read before any model code |
| `src/models.py` | Step 3 model A (BiGRU) and model B (DistilBERT); Step 7 model C (bert-base) |
| `src/errors.py` | Step 4: the 30 errors, Figure 1, Figure 2; Step 5: label-problem estimate and F1 ceiling |
| `src/fixes.py` | Step 4: one fix for each of the two largest error categories |
| `src/gap.py` | Step 6: evidence for the local-vs-leaderboard gap |
| `src/relabel.py` | Step 7: relabelling experiment |
| `src/main.py` | runs everything and writes `submission.csv` and `results.json` |

## Environment

- Python 3.13.3
- numpy 2.2.4
- pandas 2.2.3
- scikit-learn 1.9.1
- matplotlib 3.10.1
- torch TODO (Colab)
- transformers TODO (Colab)
- Pretrained models: `distilbert-base-uncased` (model B), `bert-base-uncased` (model C), from the Hugging Face Hub
- Seed 42 everywhere (split, shuffling, weight initialisation, Step 4 selection, every `random_state`)

## Step 7 (relabelling with LLM labels)

`step7/llm_labels.csv` (no header: `id,label,confidence`) holds first-pass LLM labels for the training part only.
`python src/main.py` reads it, replaces the training label where the LLM disagrees with confidence 3
(`RELABEL_MIN_CONFS` in `src/config.py`), retrains the baseline and models A, B and C (model C = bert-base-uncased, Step 7), and writes the validation F1
against the unchanged validation labels to `results.json` under `step7`. If the file is missing the step is skipped.
