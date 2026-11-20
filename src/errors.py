"""Step 4: 최고 모델이 validation 에서 틀린 트윗 30개를 뽑고, 그림 1·2 를 그린다.

선택 규칙 (코드가 적용): 최고 모델의 validation 예측 중
  거짓 양성(정답 0, 예측 1) 15개 + 거짓 음성(정답 1, 예측 0) 15개를 seed 42 로 무작위 추출.

사람이 하는 일: step4/categories.csv 의 category, reader 칸을 직접 채운다.
  - errors_30.csv 와 categories.csv 는 처음 한 번만 만들어지고, 그 뒤로는 코드가 절대 덮어쓰지 않는다.
  - 30개가 모두 채워지면 main.py 가 그림 2 (카테고리별 막대)를 그린다.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from config import ROOT, SEED, TARGET, TEXT_COL

N_PER_TYPE = 15
STEP4_DIR = ROOT / "step4"
FIG_DIR = ROOT / "figures"


def select_errors(va, va_pred):
    """틀린 validation 트윗에서 FP 15 + FN 15 를 seed 42 로 뽑는다.

    GPU 가 다르면 모델 B 의 예측이 조금 달라져 다른 30개가 뽑힐 수 있다. 사람이 읽고 분류한 30개가
    바뀌면 안 되므로, step4/errors_30.csv 가 이미 있으면 그 30개 id 를 그대로 쓰고(분할 파일과 같은 방식)
    이번 실행의 예측만 새로 붙인다. 이번 실행에서 더 이상 틀리지 않는 트윗은 개수를 기록한다.
    """
    df = va[["id", "keyword", TEXT_COL, TARGET]].copy()
    df["pred"] = va_pred
    fp = df[(df[TARGET] == 0) & (df["pred"] == 1)].assign(error_type="false positive")
    fn = df[(df[TARGET] == 1) & (df["pred"] == 0)].assign(error_type="false negative")
    counts = {"false_positives_total": int(len(fp)), "false_negatives_total": int(len(fn))}

    frozen = STEP4_DIR / "errors_30.csv"
    if frozen.exists():
        kept = pd.read_csv(frozen, encoding="utf-8-sig")[["no", "id", "error_type"]]
        picked = kept.merge(df, on="id", how="left")
        still = int(((picked[TARGET] == 0) & (picked["pred"] == 1) | (picked[TARGET] == 1) & (picked["pred"] == 0)).sum())
        counts["frozen_from"] = "step4/errors_30.csv"
        counts["still_misclassified_in_this_run"] = still
        if still < len(picked):
            print(f"[step4] warning: {len(picked) - still} of the frozen 30 are predicted correctly in this run")
        return picked[["no", "id", "keyword", TEXT_COL, TARGET, "pred", "error_type"]], counts

    picked = pd.concat([
        fp.sample(min(N_PER_TYPE, len(fp)), random_state=SEED),
        fn.sample(min(N_PER_TYPE, len(fn)), random_state=SEED),
    ]).reset_index(drop=True)
    picked.insert(0, "no", range(1, len(picked) + 1))
    return picked, counts


def save_errors(picked):
    """사람이 읽을 목록과 카테고리 입력 파일. 둘 다 처음 한 번만 만들고 그 뒤로는 덮어쓰지 않는다."""
    STEP4_DIR.mkdir(exist_ok=True)
    if not (STEP4_DIR / "errors_30.csv").exists():
        picked.to_csv(STEP4_DIR / "errors_30.csv", index=False, encoding="utf-8-sig")
    cat_file = STEP4_DIR / "categories.csv"
    if not cat_file.exists():
        sheet = picked[["no", "id", "error_type"]].copy()
        sheet["reader"] = ""          # 읽은 사람 이름
        sheet["category"] = ""        # 직접 정한 카테고리 이름
        sheet.to_csv(cat_file, index=False, encoding="utf-8-sig")


def plot_confusion(va_y, va_pred, model_name):
    """그림 1: 최고 모델의 validation 혼동행렬."""
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay.from_predictions(
        va_y, va_pred, display_labels=["not disaster (0)", "disaster (1)"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Figure 1. Confusion matrix\n{model_name}, validation part", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_confusion.png", dpi=150)
    plt.close(fig)


def plot_categories():
    """그림 2: 사람이 채운 카테고리의 개수. 30개가 다 채워졌을 때만 그린다."""
    cat_file = STEP4_DIR / "categories.csv"
    if not cat_file.exists():
        return None
    cats = pd.read_csv(cat_file, encoding="utf-8-sig")["category"].fillna("").astype(str).str.strip()
    if (cats == "").any():
        print(f"[step4] categories filled: {(cats != '').sum()}/{len(cats)} — figure 2 waits until all are filled")
        return None
    counts = cats.value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = counts.sort_values()
    ax.barh(bars.index, bars.values, color="#2a78b8")
    for y, v in enumerate(bars.values):
        ax.text(v + 0.2, y, str(v), va="center", fontsize=9)
    ax.set_xlim(0, bars.max() + 2)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel(f"number of tweets (total {int(counts.sum())})")
    ax.set_title("Figure 2. Error categories of the 30 misclassified tweets", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_categories.png", dpi=150)
    plt.close(fig)
    return counts.to_dict()


# ---------------------------------------------------------------- Step 5
LABEL_CAT = "label appears wrong"
# Hand judgments from the Step 5 reading (report, Section 5): borderline tweets about terrorism and the Iran deal
# with no real event. The strict reading does not count them; the lenient one also counts no. 28.
STRICT_EXCLUDE_IDS = (867, 886, 9484)
LENIENT_EXTRA_IDS = (7260,)


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def _wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _ceiling(fp_share, fn_share, fp_total, fn_total, n_pos):
    wrong_neg = fp_share * fp_total   # observed 0, should be 1
    wrong_pos = fn_share * fn_total   # observed 1, should be 0
    return {"label_problems_est": round(wrong_neg + wrong_pos, 1),
            "ceiling_f1": round(_f1(n_pos - wrong_pos, wrong_neg, wrong_pos), 4)}


def label_noise_estimate(error_counts, va_y):
    """Step 5: 30개 중 라벨 문제 개수, 전체 validation 으로 늘린 추정, 그리고 F1 상한.

    규칙: 읽은 사람이 'label appears wrong' 으로 분류한 트윗(라벨이 틀렸고 모델 예측이 맞다고 판단)을 센다.
    FP 15개, FN 15개를 따로 뽑았지만 모델은 FN 을 훨씬 많이 내므로, 오류 종류별 비율에 전체 FP, FN 개수를 곱한다.
    상한: 라벨이 틀린 행에서는 진짜 라벨을 맞히는 분류기도 관찰된 라벨과 어긋난다고 보고 F1 을 계산한다.
    strict / as_assigned / lenient 는 경계 사례를 세는 세 가지 읽기이고, FN 비율에는 Wilson 95% 구간을 붙인다.
    categories.csv 가 없으면 None.
    """
    if not (STEP4_DIR / "categories.csv").exists():
        return None
    cats = pd.read_csv(STEP4_DIR / "categories.csv", encoding="utf-8-sig")
    fp_total, fn_total = error_counts["false_positives_total"], error_counts["false_negatives_total"]
    n_pos = int(va_y.sum())
    is_fp = cats["error_type"] == "false positive"
    marked = cats["category"] == LABEL_CAT
    readings = {
        "strict": marked & ~cats["id"].isin(STRICT_EXCLUDE_IDS),
        "as_assigned": marked,
        "lenient": marked | cats["id"].isin(LENIENT_EXTRA_IDS),
    }
    out = {"rule": "a tweet counts when its reader judged the label wrong and the model's prediction right",
           "val_rows": int(len(va_y)), "val_positives": n_pos,
           "false_positives_total": fp_total, "false_negatives_total": fn_total, "readings": {}}
    for name, mask in readings.items():
        k_fp, n_fp = int((mask & is_fp).sum()), int(is_fp.sum())
        k_fn, n_fn = int((mask & ~is_fp).sum()), int((~is_fp).sum())
        entry = {"count_of_30": k_fp + k_fn, "fp_marked": f"{k_fp}/{n_fp}", "fn_marked": f"{k_fn}/{n_fn}",
                 **_ceiling(k_fp / n_fp, k_fn / n_fn, fp_total, fn_total, n_pos)}
        lo, hi = _wilson(k_fn, n_fn)
        entry["fn_share_95ci"] = [round(lo, 3), round(hi, 3)]
        entry["ceiling_f1_range_from_fn_ci"] = [_ceiling(k_fp / n_fp, s, fp_total, fn_total, n_pos)["ceiling_f1"]
                                                for s in (hi, lo)]
        out["readings"][name] = entry
    r = out["readings"]["as_assigned"]
    print(f"[step5] {r['count_of_30']}/30 label problems -> about {r['label_problems_est']} of "
          f"{fp_total + fn_total} validation errors, ceiling F1 {r['ceiling_f1']}")
    return out
