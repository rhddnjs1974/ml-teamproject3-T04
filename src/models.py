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
import random
import re
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from config import SEED

# Shared by A and B. Only the encoder (and the learning rate it needs) differs.
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 5
DROPOUT = 0.2
WEIGHT_DECAY = 0.01
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")
PAD, UNK = 0, 1
MIN_FREQ = 2
EMB_DIM = 100
GRU_HIDDEN = 128
GRU_LR = 1e-3

BERT_NAME = "distilbert-base-uncased"
BERT_LR = 2e-5
BERT_BASE_NAME = "bert-base-uncased"   # model C (Step 7): same recipe, 12 layers instead of DistilBERT's 6

HELD_FIXED = ("split, labels, clean_text(), max length 64 tokens, seed 42, batch 32, 5 epochs "
              "with the lowest-val-loss epoch kept, AdamW + linear decay, dropout 0.2 + linear head "
              "with one logit, BCE loss, threshold 0.5")


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WordVocab:
    def __init__(self, texts):
        counts = Counter(tok for t in texts for tok in TOKEN_RE.findall(t.lower()))
        words = sorted(w for w, c in counts.items() if c >= MIN_FREQ)
        self.index = {w: i for i, w in enumerate(words, start=2)}

    def __len__(self):
        return len(self.index) + 2

    def __call__(self, texts):
        ids = torch.full((len(texts), MAX_LEN), PAD, dtype=torch.long)
        for row, t in enumerate(texts):
            seq = [self.index.get(tok, UNK) for tok in TOKEN_RE.findall(t.lower())][:MAX_LEN] or [UNK]
            ids[row, :len(seq)] = torch.tensor(seq)
        return ids, (ids != PAD).long()


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD)
        self.gru = nn.GRU(EMB_DIM, GRU_HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(2 * GRU_HIDDEN, 1)

    def forward(self, ids, mask):
        packed = pack_padded_sequence(self.emb(ids), mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(self.drop(torch.cat([h[0], h[1]], dim=1))).squeeze(1)


class BertClassifier(nn.Module):
    def __init__(self, name=BERT_NAME):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        cls = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(self.drop(cls)).squeeze(1)


class TextClassifier:
    def __init__(self, net, encode, info):
        self.net, self.encode, self.info = net, encode, info

    @torch.no_grad()
    def predict_proba(self, texts):
        self.net.eval()
        ids, mask = self.encode(texts)
        out = [torch.sigmoid(self.net(ids[i:i + 256].to(DEVICE), mask[i:i + 256].to(DEVICE))).cpu()
               for i in range(0, len(texts), 256)]
        return torch.cat(out).numpy()

    def predict(self, texts):
        return (self.predict_proba(texts) >= THRESHOLD).astype(int)


def fit(net, encode, lr, tr_texts, tr_y, va_texts, va_y, info):
    net.to(DEVICE)
    data = TensorDataset(*encode(tr_texts), torch.tensor(tr_y, dtype=torch.float))
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1 - step / (EPOCHS * len(loader)))
    loss_fn = nn.BCEWithLogitsLoss()
    model = TextClassifier(net, encode, info)
    va_t = torch.tensor(va_y, dtype=torch.float)

    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for ids, mask, y in loader:
            opt.zero_grad()
            loss = loss_fn(net(ids.to(DEVICE), mask.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        prob = torch.tensor(model.predict_proba(va_texts))
        va_loss = float(nn.functional.binary_cross_entropy(prob.clamp(1e-7, 1 - 1e-7), va_t))
        va_f1 = float(f1_score(va_y, (prob.numpy() >= THRESHOLD).astype(int)))
        history.append({"epoch": epoch, "val_loss": round(va_loss, 4), "val_f1": round(va_f1, 4)})
        print(f"  epoch {epoch}: val loss {va_loss:.4f}  val F1 {va_f1:.4f}")
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            info["best_epoch"] = epoch

    net.load_state_dict(best_state)
    info["history"] = history
    return model


def train_model_a(tr_texts, tr_y, va_texts, va_y):
    seed_everything()
    vocab = WordVocab(tr_texts)
    info = {"encoder": "1-layer BiGRU, hidden 128, over 100-d word embeddings learned from scratch",
            "vocab_size": len(vocab), "lr": GRU_LR, "what_differs": "reference (encoder: BiGRU)",
            "held_fixed": HELD_FIXED}
    return fit(GRUClassifier(len(vocab)), vocab, GRU_LR, tr_texts, tr_y, va_texts, va_y, info)


def _train_bert(name, what_differs, tr_texts, tr_y, va_texts, va_y):
    from transformers import AutoTokenizer
    seed_everything()
    tok = AutoTokenizer.from_pretrained(name)

    def encode(texts):
        enc = tok(list(texts), max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    info = {"encoder": f"{name}, fine-tuned, [CLS] vector", "lr": BERT_LR,
            "what_differs": what_differs, "held_fixed": HELD_FIXED}
    return fit(BertClassifier(name), encode, BERT_LR, tr_texts, tr_y, va_texts, va_y, info)


def train_model_b(tr_texts, tr_y, va_texts, va_y):
    return _train_bert(BERT_NAME, "encoder: BiGRU from scratch -> pretrained DistilBERT",
                       tr_texts, tr_y, va_texts, va_y)


def train_model_c(tr_texts, tr_y, va_texts, va_y):
    """Step 7 third model: model B with a larger pretrained encoder (bert-base-uncased); nothing else changes."""
    return _train_bert(BERT_BASE_NAME, "encoder: pretrained DistilBERT (6 layers) -> bert-base-uncased (12 layers)",
                       tr_texts, tr_y, va_texts, va_y)
