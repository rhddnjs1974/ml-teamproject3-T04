# T04 Team Project 3 — Natural Language Processing with Disaster Tweets

Kaggle team: `CAU53744-T04`

## Setup

Put `train.csv`, `test.csv`, `sample_submission.csv` from
https://www.kaggle.com/competitions/nlp-getting-started/data into `data/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy==2.2.4 pandas==2.2.3 scikit-learn==1.9.1 matplotlib==3.11.2 torch==2.11.0 transformers==5.23.0
```

`distilbert-base-uncased` (model B) and `bert-base-uncased` (model C) are downloaded from the Hugging Face Hub on the first run.

## Reproduce

```bash
python src/read_samples.py   # Step 1: the 40 tweets we read before writing model code (eda/samples_to_read.txt)
python src/main.py           # everything else: writes results.json and submission.csv
```

`python src/main.py` runs, in order: duplicate handling and the split, the TF-IDF baseline, models A and B,
the Step 4 selection and figures, the two Step 4 fixes, the Step 5 ceiling and the Step 6 gap measurements,
model C and the Step 7 relabelling. It took 7.2 minutes on one RTX 5070. `runtime_seconds` in `results.json`
has the exact time.

### Files the code reads besides the Kaggle data

These are in the zip next to `src/`. They hold ids, labels and our hand-assigned categories, plus the 30
Step 4 tweets copied from train.csv.

| File | What it is | If it is missing |
|---|---|---|
| `splits/split.json` | the one 80/20 split (ids), seed 42 | recreated from seed 42 (same split) |
| `step4/errors_30.csv` | the 30 Step 4 tweets, frozen after the first run | 30 are drawn again with seed 42; on another GPU they can differ |
| `step4/categories.csv` | the category each reader gave each of the 30 | Figure 2, the fix targets and the Step 5 ceiling are skipped |
| `step7/llm_labels.csv` | first-pass LLM labels for the training part only (`id,label,confidence`, no header) | the Step 7 relabelling is skipped |

Public F1 scores cannot be computed by the code. They are recorded in `PUBLIC_F1` in `src/config.py`.

## Code layout

| File | Role |
|---|---|
| `src/config.py` | paths, seed 42, split ratio, team info, public F1 scores, Step 7 confidence rule |
| `src/data.py` | loading, `clean_text()` shared by every model, duplicate handling, the one split |
| `src/read_samples.py` | Step 1: 20 positive + 20 negative tweets to read |
| `src/baseline.py` | Step 2: TF-IDF + logistic regression |
| `src/models.py` | Step 3: model A (BiGRU), model B (DistilBERT); Step 7: model C (bert-base-uncased) |
| `src/errors.py` | Step 4: the 30 errors, Figure 1, Figure 2; Step 5: label-problem estimate and F1 ceiling |
| `src/fixes.py` | Step 4: one fix for each of the two largest error categories |
| `src/gap.py` | Step 6: measurements behind the local-vs-leaderboard gap (no retraining) |
| `src/relabel.py` | Step 7: relabelling the training part with LLM labels |
| `src/main.py` | runs everything and writes `results.json`, `submission.csv`, `figures/` |

## Environment

- Python 3.13.3
- numpy 2.2.4
- pandas 2.2.3
- scikit-learn 1.9.1
- matplotlib 3.11.2
- torch 2.11.0 (CUDA 12.8)
- transformers 5.23.0
- Pretrained models: `distilbert-base-uncased` (model B), `bert-base-uncased` (model C), from the Hugging Face Hub
- GPU: NVIDIA GeForce RTX 5070 (12 GB). On another GPU the last digits of models A, B and C may differ.
- Seed 42 everywhere (split, shuffling, weight initialisation, Step 4 selection, every `random_state`).
  With `cudnn.deterministic`, two runs on the same machine give the same numbers apart from timings.

## Step 7 (relabelling with LLM labels)

`step7/llm_labels.csv` (no header: `id,label,confidence`) holds first-pass LLM labels for the training part only.
`python src/main.py` reads it, replaces the training label where the LLM disagrees with confidence 3
(`RELABEL_MIN_CONFS` in `src/config.py`), retrains the baseline and models A, B and C, and writes the validation F1
against the unchanged validation labels to `results.json` under `step7`. If the file is missing the step is skipped.
