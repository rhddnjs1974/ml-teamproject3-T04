"""데이터 로드, 텍스트 정제, 중복 처리, 80/20 분할.

분할은 make_or_load_split() 하나로만 한다. 처음 한 번 만들어 splits/split.json 에 저장하고,
그 뒤로는 모든 실험이 이 파일을 읽는다 (모델마다 다시 나누지 않는다).
"""
import html
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, SEED, SPLIT_FILE, TARGET, TEXT_COL, VAL_FRAC

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")


def load_raw():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def dedup_key(text):
    """중복 판정용 키: HTML 기호 복원, URL 삭제, 공백 정리, 소문자.

    같은 기사를 링크만 바꿔 다시 올린 트윗(t.co 주소만 다름)을 같은 문장으로 본다.
    """
    t = html.unescape(text)
    t = URL_RE.sub("", t)
    return SPACE_RE.sub(" ", t).strip().lower()


def clean_text(text):
    """모든 모델(TF-IDF, 모델 A, 모델 B)이 공통으로 쓰는 정제. 학습하는 값이 없으므로 누수 없음.

    - &amp; 같은 HTML 기호를 원래 문자로
    - URL 은 "http" 한 단어로 (링크가 있다는 사실은 남기고 주소는 버림)
    - @멘션은 "@user" 로 (사람 이름은 일반화가 안 됨)
    - 해시태그는 # 만 떼고 단어는 남김 (#earthquake → earthquake)
    """
    t = html.unescape(text)
    t = URL_RE.sub(" http ", t)
    t = MENTION_RE.sub(" @user ", t)
    t = t.replace("#", " ")
    return SPACE_RE.sub(" ", t).strip()


def deduplicate(train):
    """분할 전에 중복을 처리한다.

    규칙: dedup_key 가 같은 행들을 한 묶음으로 보고
      - 라벨 다수결로 묶음의 라벨을 정하고, 첫 번째 행 하나만 남긴다
      - 다수결이 동률(예: 1 하나, 0 하나)이면 그 묶음을 통째로 버린다
    → 같은 문장이 training 과 validation 양쪽에 들어갈 수 없다.
    """
    df = train.copy()
    df["_key"] = df[TEXT_COL].map(dedup_key)
    groups = df.groupby("_key")[TARGET].agg(["size", "mean"])
    tie = groups[groups["mean"] == 0.5].index
    majority = (groups["mean"] > 0.5).astype(int)

    kept = df[~df["_key"].isin(tie)].drop_duplicates("_key", keep="first").copy()
    kept[TARGET] = kept["_key"].map(majority)
    report = {
        "duplicate_groups": int((groups["size"] > 1).sum()),
        "rows_in_duplicate_groups": int(groups.loc[groups["size"] > 1, "size"].sum()),
        "conflicting_groups": int(((groups["mean"] > 0) & (groups["mean"] < 1)).sum()),
        "tied_groups_dropped": int(len(tie)),
        "rows_dropped": int(len(df) - len(kept)),
    }
    return kept.drop(columns="_key").reset_index(drop=True), report


def make_or_load_split(clean_train):
    """80/20 stratified 분할. 파일이 있으면 읽고, 없으면 만들어 저장한다 (id 로 저장)."""
    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    else:
        tr_ids, va_ids = train_test_split(
            clean_train["id"].to_numpy(), test_size=VAL_FRAC,
            stratify=clean_train[TARGET], random_state=SEED,
        )
        split = {"seed": SEED, "train_ids": sorted(map(int, tr_ids)), "val_ids": sorted(map(int, va_ids))}
        SPLIT_FILE.parent.mkdir(exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(split), encoding="utf-8")
    by_id = clean_train.set_index("id")
    tr = by_id.loc[split["train_ids"]].reset_index()
    va = by_id.loc[split["val_ids"]].reset_index()
    return tr, va


def load_split():
    """다른 파일(모델 A, B, 오류 분석)에서 쓰는 진입점: 정제된 train / val / Kaggle test."""
    raw_train, test = load_raw()
    clean_train, _ = deduplicate(raw_train)
    tr, va = make_or_load_split(clean_train)
    return tr, va, test


def describe(name, df):
    return {"set": name, "rows": int(len(df)), "positive_rate": round(float(df[TARGET].mean()), 4)}
