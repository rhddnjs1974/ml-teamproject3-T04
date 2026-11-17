"""Step 4: 가장 큰 두 오류 카테고리에 대해 한 가지씩 바꿔서 다시 학습하고 validation 으로 비교한다.

각 수정은 최고 모델(Step 4 모델)의 학습 함수를 그대로 쓰고, 입력 데이터만 한 가지 바꾼다.
validation 라벨은 절대 바꾸지 않는다.

Fix 1 — 1위 "label appears wrong" (14개)
    가설: 학습 데이터에도 틀린 라벨이 섞여 있어 모델이 그것까지 외운다.
    변경: training 부분 안에서만 5-fold 교차예측(TF-IDF + 로지스틱 회귀)을 하고,
          자기 라벨에 준 확률이 NOISE_PROB 미만인 행을 training 에서 뺀다. (기준값은 실행 전에 고정)

Fix 2 — 2위 "disaster word, no disaster event" (3개, 동률 셋 중 선택)
    가설: 모델이 문맥보다 재난 단어(keyword) 자체에 반응한다.
    변경: 각 트윗의 keyword 단어를 텍스트에서 [MASK] 로 가리고 training/validation 모두 같은 규칙 적용.
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from baseline import build_baseline
from config import ROOT, SEED

# 모델 B 결과를 보기 전에, 빠지는 행 수만 보고 정함 (training 의 약 3%: 0.1 은 8행, 0.25 는 187행)
NOISE_PROB = 0.25
MASK = "[MASK]"


def suspected_noise(tr_x, tr_y):
    """training 부분 안에서만 교차예측. 자기 라벨에 준 확률이 NOISE_PROB 미만이면 의심 행."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    prob1 = cross_val_predict(build_baseline(), tr_x, tr_y, cv=cv, method="predict_proba")[:, 1]
    own = np.where(tr_y == 1, prob1, 1 - prob1)
    return own < NOISE_PROB


def mask_keyword(texts, keywords):
    """keyword(예: 'buildings%20burning')를 텍스트에서 [MASK] 로 바꾼다. keyword 가 없으면 그대로."""
    out = []
    for t, k in zip(texts, keywords):
        if isinstance(k, str) and k:
            phrase = k.replace("%20", " ")
            t = re.sub(r"\b" + re.escape(phrase) + r"\b", MASK, t, flags=re.IGNORECASE)
        out.append(t)
    return out


def run_fixes(train_fn, base_val_pred, tr, va, tr_x, tr_y, va_x, va_y):
    """두 수정을 각각 한 번씩 학습해 validation F1 과 30개 오류의 변화를 기록한다."""
    cats = _load_categories()
    base_f1 = float(f1_score(va_y, base_val_pred))
    results = {}

    noisy = suspected_noise(tr_x, tr_y)
    keep = ~noisy
    model = train_fn([x for x, k in zip(tr_x, keep) if k], tr_y[keep], va_x, va_y)
    results["fix1_drop_suspected_noisy_labels"] = _summarise(
        model.predict(va_x), base_f1, va, va_y, cats, "label appears wrong",
        {"rows_removed": int(noisy.sum()), "removed_positive": int((noisy & (tr_y == 1)).sum()),
         "rows_left": int(keep.sum()), "rule": f"5-fold OOF TF-IDF+LR on the training part, own-label prob < {NOISE_PROB}"})

    tr_m = mask_keyword(tr_x, tr["keyword"])
    va_m = mask_keyword(va_x, va["keyword"])
    changed = sum(a != b for a, b in zip(tr_m, tr_x))
    model = train_fn(tr_m, tr_y, va_m, va_y)
    results["fix2_mask_keyword"] = _summarise(
        model.predict(va_m), base_f1, va, va_y, cats, "disaster word, no disaster event",
        {"train_texts_changed": int(changed), "rule": "replace the row's keyword with [MASK] in training and validation text"})

    for name, r in results.items():
        print(f"[{name}] val F1 {r['val_f1']:.4f} (diff vs Step 4 model {r['diff_vs_step4_model']:+.4f}); "
              f"target category fixed {r['target_category_now_correct']}/{r['target_category_size']}")
    return results


def _load_categories():
    path = ROOT / "step4" / "categories.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


def _summarise(pred, base_f1, va, va_y, cats, target_cat, extra):
    pred = np.asarray(pred).astype(int)
    f1 = float(f1_score(va_y, pred))
    out = {"val_f1": f1, "diff_vs_step4_model": f1 - base_f1, **extra}
    if cats is not None:
        correct = dict(zip(va["id"], pred == va_y))
        ids = cats.loc[cats["category"] == target_cat, "id"]
        out["target_category"] = target_cat
        out["target_category_size"] = int(len(ids))
        out["target_category_now_correct"] = int(sum(correct[i] for i in ids))
        out["all_30_now_correct"] = int(sum(correct[i] for i in cats["id"]))
    return out
