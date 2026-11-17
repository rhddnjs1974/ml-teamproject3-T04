"""한 번의 명령으로 중복 처리 → 분할 → 베이스라인/모델 A/모델 B → 제출 파일 → results.json.

사용법 (프로젝트 루트에서):
    python src/main.py
"""
import json
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from baseline import build_baseline
from config import ID_COL, MEMBERS, OUT_DIR, PUBLIC_F1, ROOT, SEED, TARGET, TEAM_ID, TEXT_COL
from data import clean_text, deduplicate, describe, load_raw, make_or_load_split
from errors import plot_categories, plot_confusion, save_errors, select_errors
from models import train_model_a, train_model_b


def write_submission(test, pred, name):
    sub = pd.DataFrame({"id": test[ID_COL], "target": np.asarray(pred).astype(int)})
    path = OUT_DIR / "submissions" / f"{name}.csv"
    sub.to_csv(path, index=False)
    return path


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    start = time.time()
    (OUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)

    # Step 1. 중복 처리 → 분할 (분할 파일이 있으면 그대로 읽는다)
    raw_train, test = load_raw()
    clean_train, dup_report = deduplicate(raw_train)
    tr, va = make_or_load_split(clean_train)
    data_table = [describe("train.csv as downloaded", raw_train),
                  describe("after duplicate handling", clean_train),
                  describe("training part", tr),
                  describe("validation part", va)]
    print("duplicates:", dup_report)
    for row in data_table:
        print(f"  {row['set']:26s} rows {row['rows']:5d}  positive rate {row['positive_rate']:.4f}")

    # 모든 모델이 같은 정제 결과를 쓴다
    tr_x = [clean_text(t) for t in tr[TEXT_COL]]
    va_x = [clean_text(t) for t in va[TEXT_COL]]
    te_x = [clean_text(t) for t in test[TEXT_COL]]
    tr_y = tr[TARGET].to_numpy()
    va_y = va[TARGET].to_numpy()

    # Step 2. TF-IDF 베이스라인 + Step 3. 모델 A, B
    runs = {}
    for name, train_fn in [("baseline", lambda *a: build_baseline().fit(a[0], a[1])),
                           ("model_a", train_model_a), ("model_b", train_model_b)]:
        t0 = time.time()
        try:
            model = train_fn(tr_x, tr_y, va_x, va_y)
        except NotImplementedError as e:
            print(f"[{name}] skipped: {e}")
            continue
        va_pred = np.asarray(model.predict(va_x)).astype(int)
        runs[name] = {
            "model": model,
            "val_pred": va_pred,
            "val_f1": float(f1_score(va_y, va_pred)),
            "train_seconds": round(time.time() - t0, 1),
        }
        write_submission(test, model.predict(te_x), name)
        print(f"[{name}] val F1 = {runs[name]['val_f1']:.4f} ({runs[name]['train_seconds']}s)")

    base_f1 = runs["baseline"]["val_f1"]
    best = max(runs, key=lambda k: runs[k]["val_f1"])
    pd.read_csv(OUT_DIR / "submissions" / f"{best}.csv").to_csv(ROOT / "submission.csv", index=False)
    print(f"best on validation: {best}")

    # Step 4. 최고 모델의 오류 30개 선택 + 그림 1, (카테고리가 다 채워졌으면) 그림 2
    picked, error_counts = select_errors(va, runs[best]["val_pred"])
    save_errors(picked)
    plot_confusion(va_y, runs[best]["val_pred"], best)
    category_counts = plot_categories()
    print(f"[step4] {len(picked)} errors saved to step4/errors_30.csv ({error_counts})")

    output = {
        "project": 3,
        "team_id": TEAM_ID,
        "members": MEMBERS,
        "seed": SEED,
        "split_sizes": {"train": len(tr), "val": len(va)},
        "data_table": data_table,
        "duplicates": dup_report,
        "models": {
            name: {"val_f1": r["val_f1"], "diff_vs_baseline": r["val_f1"] - base_f1,
                   "train_seconds": r["train_seconds"], "public_f1": PUBLIC_F1.get(name),
                   "info": getattr(r["model"], "info", None)}
            for name, r in runs.items()
        },
        "best_model": best,
        "step4": {
            "model": best,
            "rule": "15 false positives + 15 false negatives of the best model on validation, sampled with seed 42",
            **error_counts,
            "examples": [{"no": int(r.no), "id": int(r.id), "true": int(r.target), "pred": int(r.pred)}
                         for r in picked.itertuples()],
            "category_counts": category_counts,
        },
        "runtime_seconds": round(time.time() - start, 1),
    }
    with open(ROOT / "results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"done in {output['runtime_seconds']}s")


if __name__ == "__main__":
    main()
