"""프로젝트 전체에서 공유하는 설정값 (경로, 시드, 분할 비율, 팀 정보)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"             # 모델별 제출 파일 등 (zip에는 넣지 않음)
SPLIT_FILE = ROOT / "splits" / "split.json"   # 한 번 만든 80/20 분할. 모든 실험이 이 파일을 읽는다

SEED = 42
VAL_FRAC = 0.2

ID_COL = "id"
TEXT_COL = "text"
TARGET = "target"

TEAM_ID = "T04"
MEMBERS = [
    {"student_id": "20212788", "name": "최재원", "role": "Steps 0-2 (data, duplicates, split, TF-IDF baseline), Step 6 (local vs leaderboard)"},
    {"student_id": "20225259", "name": "김민성", "role": "Step 5 (label noise), Step 7 (extension)"},
    {"student_id": "20223843", "name": "차승준", "role": "Step 3 (models A and B), report"},
]

# Kaggle에 제출한 뒤 받은 Public F1 (코드가 알 수 없는 값이라 여기에 기록)
PUBLIC_F1 = {
    "baseline": 0.79405,
    "model_a": 0.76647,
    "model_b": 0.83450,
}
