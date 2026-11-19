"""Step 6: 로컬 val F1 과 Kaggle public F1 의 차이를 설명할 근거를 계산한다.

PDF 가 준 원인 후보를 하나씩 근거로 판정한다.
  1) test 의 양성 비율이 다르다          → 모델별 예측 양성 비율 (val vs Kaggle test)
  2) val 로 판정 기준값을 조정했다        → 아니오: 0.5 고정 (models.THRESHOLD, LogisticRegression 기본)
  3) 같거나 거의 같은 문장이 양쪽에 있다  → Kaggle test 중 training 부분과 dedup_key 가 같은 행의 수,
                                          그 행들에서 모델 예측이 training 쪽 라벨과 일치하는 비율
  4) training 부분보다 많은 데이터로 학습 → 아니오: Kaggle 파일도 training 부분으로 fit 한 모델에서 나옴
  5) 전처리가 다르다                      → 아니오: Kaggle test 도 같은 clean_text() 를 거침
"""
import numpy as np
import pandas as pd

from config import OUT_DIR, TARGET, TEXT_COL
from data import dedup_key


def analyse_gap(tr, va, test, runs, public_f1):
    tr_keys = tr[TEXT_COL].map(dedup_key)
    label_of_key = pd.Series(tr[TARGET].to_numpy(), index=tr_keys).groupby(level=0).first()
    te_keys = test[TEXT_COL].map(dedup_key)
    near_dup = te_keys.isin(label_of_key.index).to_numpy()
    va_near_dup = int(va[TEXT_COL].map(dedup_key).isin(label_of_key.index).sum())

    out = {
        "kaggle_test_rows": int(len(test)),
        "kaggle_test_rows_seen_in_training_part": int(near_dup.sum()),
        "kaggle_test_share_seen_in_training_part": round(float(near_dup.mean()), 4),
        "validation_rows_seen_in_training_part": va_near_dup,
        "training_positive_rate": round(float(tr[TARGET].mean()), 4),
        "validation_positive_rate": round(float(va[TARGET].mean()), 4),
        "threshold": "0.5 fixed for every model (not tuned on validation)",
        "kaggle_file_fit_on": "training part only (same model object as the validation numbers)",
        "kaggle_preprocessing": "same clean_text() as training and validation",
        "models": {},
    }
    dup_label = te_keys[near_dup].map(label_of_key).to_numpy()
    for name, r in runs.items():
        sub = pd.read_csv(OUT_DIR / "submissions" / f"{name}.csv")
        te_pred = sub["target"].to_numpy()
        out["models"][name] = {
            "val_f1": r["val_f1"],
            "public_f1": public_f1.get(name),
            "gap_public_minus_val": None if public_f1.get(name) is None else public_f1[name] - r["val_f1"],
            "val_predicted_positive_rate": round(float(np.mean(r["val_pred"])), 4),
            "kaggle_test_predicted_positive_rate": round(float(te_pred.mean()), 4),
            "agreement_with_training_label_on_seen_rows": round(float((te_pred[near_dup] == dup_label).mean()), 4),
            "val_accuracy": round(float((r["val_pred"] == va[TARGET].to_numpy()).mean()), 4),
        }
    print(f"[step6] Kaggle test rows already in the training part: {out['kaggle_test_rows_seen_in_training_part']} "
          f"({out['kaggle_test_share_seen_in_training_part']:.1%}); validation: {va_near_dup}")
    for name, m in out["models"].items():
        print(f"  {name}: agreement on seen rows {m['agreement_with_training_label_on_seen_rows']:.3f} "
              f"vs val accuracy {m['val_accuracy']:.3f}; predicted positive rate val {m['val_predicted_positive_rate']:.3f} "
              f"/ test {m['kaggle_test_predicted_positive_rate']:.3f}")
    return out
