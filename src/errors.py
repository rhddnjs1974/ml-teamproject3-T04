"""Step 4: 최고 모델이 validation 에서 틀린 트윗 30개를 뽑고, 그림 1·2 를 그린다.

선택 규칙 (코드가 적용): 최고 모델의 validation 예측 중
  거짓 양성(정답 0, 예측 1) 15개 + 거짓 음성(정답 1, 예측 0) 15개를 seed 42 로 무작위 추출.

사람이 하는 일: step4/categories.csv 의 category, reader 칸을 직접 채운다.
  - 이 파일은 처음 한 번만 만들어지고, 그 뒤로는 코드가 절대 덮어쓰지 않는다.
  - 30개가 모두 채워지면 main.py 가 그림 2 (카테고리별 막대)를 그린다.
"""
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
    """틀린 validation 트윗에서 FP 15 + FN 15 를 seed 42 로 뽑는다."""
    df = va[["id", "keyword", TEXT_COL, TARGET]].copy()
    df["pred"] = va_pred
    fp = df[(df[TARGET] == 0) & (df["pred"] == 1)].assign(error_type="false positive")
    fn = df[(df[TARGET] == 1) & (df["pred"] == 0)].assign(error_type="false negative")
    picked = pd.concat([
        fp.sample(min(N_PER_TYPE, len(fp)), random_state=SEED),
        fn.sample(min(N_PER_TYPE, len(fn)), random_state=SEED),
    ]).reset_index(drop=True)
    picked.insert(0, "no", range(1, len(picked) + 1))
    counts = {"false_positives_total": int(len(fp)), "false_negatives_total": int(len(fn))}
    return picked, counts


def save_errors(picked):
    """사람이 읽을 목록(매번 새로 씀)과 카테고리 입력 파일(처음 한 번만 만듦)."""
    STEP4_DIR.mkdir(exist_ok=True)
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
    fig, ax = plt.subplots(figsize=(6, 3.4))
    counts.sort_values().plot.barh(ax=ax, color="#2a78b8")
    ax.set_xlabel("number of tweets (total 30)")
    ax.set_title("Figure 2. Error categories of the 30 misclassified tweets")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_categories.png", dpi=150)
    plt.close(fig)
    return counts.to_dict()
