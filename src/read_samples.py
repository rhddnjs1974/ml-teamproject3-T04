"""Step 1: 모델 코드 전에 사람이 읽을 트윗 40개를 뽑는다 (재난 20 + 재난 아님 20, seed 42).

사용법: python src/read_samples.py
결과: eda/samples_to_read.txt  ← 읽고 나서 규칙 2개는 eda/step1_rules.md 에 직접 적는다
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
    lines.append("===== 읽고 나서 세운 규칙 2개는 eda/step1_rules.md 에 적는다 (이 파일은 다시 실행하면 덮어써짐) =====")
    out = ROOT / "eda" / "samples_to_read.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
