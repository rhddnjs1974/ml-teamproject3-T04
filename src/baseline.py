"""Step 2: TF-IDF + 로지스틱 회귀 베이스라인.

TF-IDF 단어 사전과 IDF 값은 training 부분에만 fit 한다 (Pipeline 안에 있으므로 자동).
판정 기준값은 0.5 그대로 (validation 으로 조정하지 않음).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import SEED


def build_baseline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
