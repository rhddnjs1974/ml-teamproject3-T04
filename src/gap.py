"""Step 6: 로컬 val F1 과 Kaggle public F1 의 차이를 설명할 근거를 계산한다 (재학습 없음).

PDF 가 준 원인 후보를 하나씩 숫자로 판정한다.
  1) 같거나 거의 같은 문장이 양쪽에 있다 → Kaggle test 중 train.csv / training 부분과 dedup_key 가 같은 행의 수,
                                           그 행들에서 예측이 training 쪽 라벨과 일치하는 비율 (val 정확도와 비교),
                                           중복 처리로 지운 validation 트윗의 사본을 되돌려 넣었을 때의 val F1
  2) test 의 양성 비율이 다르다          → 같은 TPR, FPR 의 모델이 train.csv 원래 양성 비율에서 낼 F1,
                                           모델별 예측 양성 비율 (val vs Kaggle test)
  3) 전처리가 다르다                      → 같은 clean_text(); 모델 A 어휘 기준 미등록 단어 비율 (val vs test)
  4) val 로 판정 기준값을 조정했다        → 아니오: 0.5 고정
  5) training 부분보다 많은 데이터로 학습 → 아니오: Kaggle 파일도 training 부분으로 fit 한 같은 모델에서 나옴
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from config import TARGET, TEXT_COL
from data import clean_text, dedup_key


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def _f1_at_rate(y, pred, rate):
    """F1 of a classifier with the same TPR and FPR on a set whose positive rate is `rate`."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    return _f1(rate * tpr, (1 - rate) * fpr, rate * (1 - tpr))


def analyse_gap(runs, public_f1, raw_train, clean_train, tr, va, test, va_x, te_x, va_y):
    raw_rate = float(raw_train[TARGET].mean())
    raw_keys = raw_train[TEXT_COL].map(dedup_key)
    tr_keys = set(tr[TEXT_COL].map(dedup_key))
    va_keys = set(va[TEXT_COL].map(dedup_key))
    te_keys = test[TEXT_COL].map(dedup_key)

    # copies of validation tweets that duplicate handling removed (each copy keeps its own label)
    dropped = raw_train[~raw_train["id"].isin(clean_train["id"])]
    back = dropped[raw_keys[dropped.index].isin(va_keys)]
    back_x = [clean_text(t) for t in back[TEXT_COL]]
    va_back_y = np.concatenate([va_y, back[TARGET].to_numpy()])

    # training-part label of each test tweet that has a copy in the training part
    tr_label = dict(zip(tr[TEXT_COL].map(dedup_key), tr[TARGET]))
    in_tr = te_keys.isin(tr_keys).to_numpy()

    out = {
        "positive_rate": {"train_csv_raw": round(raw_rate, 4),
                          "after_duplicate_handling": round(float(clean_train[TARGET].mean()), 4),
                          "validation": round(float(va_y.mean()), 4),
                          "removed_duplicate_rows": round(float(dropped[TARGET].mean()), 4)},
        "duplicates": {"test_rows": int(len(test)),
                       "test_rows_with_copy_in_train_csv": int(te_keys.isin(set(raw_keys)).sum()),
                       "test_rows_with_copy_in_training_part": int(in_tr.sum()),
                       "test_rows_with_copy_in_validation_part": int(te_keys.isin(va_keys).sum()),
                       "validation_rows_with_copy_in_training_part": len(va_keys & tr_keys),
                       "removed_copies_of_validation_tweets": int(len(back)),
                       "removed_copies_positive_rate": round(float(back[TARGET].mean()), 4)},
        "threshold": "0.5 for every model, not tuned on validation",
        "kaggle_fit": "every Kaggle file comes from the training-part fit (5,535 rows), not a refit on train.csv",
        "models": {},
    }
    for name in ("baseline", "model_a", "model_b"):
        if name not in runs:
            continue
        model, va_pred = runs[name]["model"], runs[name]["val_pred"]
        te_pred = np.asarray(model.predict(te_x)).astype(int)
        back_pred = np.asarray(model.predict(back_x)).astype(int) if back_x else np.array([], dtype=int)
        copy_lbl = np.array([tr_label[k] for k in te_keys[in_tr]])
        out["models"][name] = {
            "val_f1": round(runs[name]["val_f1"], 4),
            "public_f1": public_f1.get(name),
            "gap_public_minus_val": None if public_f1.get(name) is None else round(public_f1[name] - runs[name]["val_f1"], 4),
            "val_accuracy": round(float((va_pred == va_y).mean()), 4),
            "val_f1_at_raw_positive_rate": round(_f1_at_rate(va_y, va_pred, raw_rate), 4),
            "val_f1_with_removed_copies_back": round(float(f1_score(va_back_y, np.concatenate([va_pred, back_pred]))), 4),
            "val_predicted_positive_rate": round(float(va_pred.mean()), 4),
            "test_predicted_positive_rate": round(float(te_pred.mean()), 4),
            "test_copies_agree_with_training_label": round(float((te_pred[in_tr] == copy_lbl).mean()), 4),
        }
    if "model_a" in runs:
        vocab = runs["model_a"]["model"].encode

        def unk_share(texts):
            ids, mask = vocab(texts)
            return round(float((ids[mask.bool()] == 1).float().mean()), 4)

        out["preprocessing"] = {"unknown_word_share_validation": unk_share(va_x),
                                "unknown_word_share_test": unk_share(te_x)}
    for name, m in out["models"].items():
        print(f"[step6] {name}: val F1 {m['val_f1']}, at raw positive rate {m['val_f1_at_raw_positive_rate']}, "
              f"with removed copies back {m['val_f1_with_removed_copies_back']}; agreement on test copies "
              f"{m['test_copies_agree_with_training_label']} vs val accuracy {m['val_accuracy']}")
    return out
