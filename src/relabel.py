"""Step 7: relabelling experiment with first-pass LLM labels.

Idea: for training rows where the LLM disagrees with the current label AND its confidence is at least
RELABEL_MIN_CONFS (fixed in config.py before any result was seen), replace the training label by the LLM label.
Then retrain the same models and read the VALIDATION F1 against the UNCHANGED validation labels.

Rules this file keeps (assignment, Step 5 and Step 7):
  - only training labels change; validation labels and the validation part are never touched
  - the split, text cleaning, seed and every model setting stay the same (the models' own train functions are reused)
  - the LLM labels file covers training ids only; the code stops if it contains a validation id
Input: step7/llm_labels.csv, no header, columns id,label,confidence (confidence 1 to 3).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import LLM_LABELS_FILE, OUT_DIR, RELABEL_MIN_CONFS

RELABEL_MODELS = ("baseline", "model_a", "model_b")


def load_llm_labels(path=LLM_LABELS_FILE):
    """Read the LLM labels (id, llm, conf). Returns None when the file is missing."""
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=["id", "llm", "conf"], encoding="utf-8-sig")
    if not df["id"].is_unique:
        raise ValueError(f"{path} has repeated ids")
    if not df["llm"].isin([0, 1]).all() or not df["conf"].isin([1, 2, 3]).all():
        raise ValueError(f"{path} has values outside label 0/1 or confidence 1-3")
    return df.set_index("id")


def relabel_training(tr_ids, tr_y, llm, min_conf):
    """Return (new labels, boolean mask of the changed rows). Only rows the LLM labelled and disagrees on change."""
    ids = pd.Series(np.asarray(tr_ids))
    llm_label = ids.map(llm["llm"]).to_numpy(dtype=float)     # NaN where the LLM has no label
    conf = ids.map(llm["conf"]).to_numpy(dtype=float)
    change = (conf >= min_conf) & (llm_label != tr_y)          # NaN compares False on conf, so unknown rows stay
    new_y = tr_y.copy()
    new_y[change] = llm_label[change].astype(int)
    return new_y, change


def run_relabel(train_fns, runs, tr, tr_x, tr_y, va, va_x, va_y):
    """Retrain baseline, model A and model B on the relabelled training part for each confidence level."""
    llm = load_llm_labels()
    if llm is None:
        print(f"[step7] {LLM_LABELS_FILE} not found: relabelling experiment skipped")
        return None
    overlap = set(va["id"]) & set(llm.index)
    if overlap:
        raise ValueError(f"LLM labels contain {len(overlap)} validation ids; training part only")

    tr_ids = tr["id"].to_numpy()
    results = {}
    for min_conf in RELABEL_MIN_CONFS:
        new_y, change = relabel_training(tr_ids, tr_y, llm, min_conf)
        entry = {
            "rule": f"training rows where the LLM label differs from the label and LLM confidence >= {min_conf}; "
                    "the LLM label replaces the training label; validation labels unchanged",
            "rows_changed": int(change.sum()),
            "changed_0_to_1": int((change & (tr_y == 0)).sum()),
            "changed_1_to_0": int((change & (tr_y == 1)).sum()),
            "train_positive_rate_before": round(float(tr_y.mean()), 4),
            "train_positive_rate_after": round(float(new_y.mean()), 4),
            "models": {},
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": tr_ids[change], "old_label": tr_y[change], "new_label": new_y[change]}).to_csv(
            OUT_DIR / f"relabel_conf{min_conf}_changes.csv", index=False)
        for name in RELABEL_MODELS:
            if name not in runs:
                continue
            model = train_fns[name](tr_x, new_y, va_x, va_y)
            pred = np.asarray(model.predict(va_x)).astype(int)
            f1 = float(f1_score(va_y, pred))
            entry["models"][name] = {
                "val_f1_original_labels": runs[name]["val_f1"],
                "val_f1_relabelled": f1,
                "diff": f1 - runs[name]["val_f1"],
                "val_predicted_positive_rate_original": round(float(np.mean(runs[name]["val_pred"])), 4),
                "val_predicted_positive_rate_relabelled": round(float(pred.mean()), 4),
            }
            print(f"[step7 conf>={min_conf}] {name}: val F1 {runs[name]['val_f1']:.4f} -> {f1:.4f} "
                  f"({f1 - runs[name]['val_f1']:+.4f}), {entry['rows_changed']} rows relabelled")
        results[f"llm_conf_ge_{min_conf}"] = entry
    return results
