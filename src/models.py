"""Step 3 모델 A, 모델 B (담당: 차승준).

main.py 와의 약속 (이 형식만 지키면 내부 구현은 자유):

    train_model_a(tr_texts, tr_y, va_texts, va_y) -> model
    train_model_b(tr_texts, tr_y, va_texts, va_y) -> model
        tr_texts, va_texts : data.clean_text() 를 거친 문자열 list (정제는 이미 끝남, 따로 하지 말 것)
        tr_y, va_y         : 0/1 numpy 배열
        model.predict(texts) -> 0/1 numpy 배열
        (선택) model.info -> dict, 예: {"what_differs": "encoder: GRU -> DistilBERT", "epochs": 5}

규칙
  - 단어 사전, 토크나이저 어휘 등은 tr_texts 로만 만든다
  - va 는 early stopping / epoch 선택에만 쓴다
  - seed 42 (torch.manual_seed, 셔플 generator, random_state)
  - A 와 B 사이에는 한 요소만 다르게 하고, 그 요소와 고정한 것들을 보고서에 적는다
  - 분할은 data.load_split() 이 주는 것만 쓴다 (다시 나누지 않는다)
"""


def train_model_a(tr_texts, tr_y, va_texts, va_y):
    raise NotImplementedError("model A: 담당자가 구현")


def train_model_b(tr_texts, tr_y, va_texts, va_y):
    raise NotImplementedError("model B: 담당자가 구현")
