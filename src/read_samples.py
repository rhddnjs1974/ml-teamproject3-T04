"""Step 1: 모델 코드 전에 사람이 읽을 트윗 40개를 뽑는다 (재난 20 + 재난 아님 20, seed 42).

사용법: python src/read_samples.py
결과: eda/samples_to_read.txt  ← 읽고 나서 맨 아래에 "내가 쓸 규칙 2개"를 직접 적는다
"""
import pandas as pd

from config import DATA_DIR, ROOT, SEED, TARGET, TEXT_COL


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    lines = []
    for label, name in [(1, "target = 1 (real disaster)"), (0, "target = 0 (not a disaster)")]:
        sample = train[train[TARGET] == label].sample(20, random_state=SEED)
        lines.append(f"===== {name}: 20 random tweets =====")
        for i, row in enumerate(sample.itertuples(), start=1):
            lines.append(f"{i:2d}. [id {row.id}] [keyword: {row.keyword}] {getattr(row, TEXT_COL)}")
        lines.append("")
    lines += [
        "===== 읽고 나서 직접 적기: 내가 판단할 때 쓸 규칙 2개 =====",
        "규칙 1: ",
        "규칙 2: ",
    ]
    out = ROOT / "eda" / "samples_to_read.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
