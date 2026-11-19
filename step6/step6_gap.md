# Step 6 — Local against leaderboard (재료: 최재원)

숫자는 `results.json`의 `models`와 `step6`에서 가져온다. 승준님 최종 실행 뒤 val F1 이 바뀌었으면 표의 숫자만 그 값으로 바꾼다.
Public F1 과 날짜는 Kaggle Submissions 화면(그림 3)에서 옮긴 값이다 (코드가 알 수 없는 유일한 숫자).

## Table (for the report)

| Submission | Local val F1 | Public F1 | Gap (public − local) | Date |
|---|---|---|---|---|
| TF-IDF baseline | 0.7093 | 0.79405 | +0.085 | 2026-11-17 |
| Model A (BiGRU) | 0.6873 | 0.76647 | +0.079 | 2026-11-17 |
| Best = model B (DistilBERT), `submission.csv` | 0.7996 | 0.83450 | +0.035 | 2026-11-17 |

Every Kaggle file comes from the same training-part fit as its validation number, with the threshold fixed at 0.5.

## Gap paragraph (English draft)

All three public scores sit above the local ones, so the usual suspect, a leak that inflates the local number, does not
fit; the leaderboard is the easier test here. We ruled the five candidate causes in or out with numbers from our own run.
(i) **Near-identical texts in both parts — the main cause.** Our duplicate handling removed every validation tweet whose
text (URLs stripped) also appears in the training part, so the validation part has 0 such rows; the Kaggle test has
323 of 3,263 (9.9%). On those 323 rows the models agree with the training copy's label far more often than they are
right on validation: 94.1% for the baseline against 79.3% validation accuracy, 90.7% for model A and 95.0% for model B.
The leaderboard therefore contains a slice of near-free answers that our validation part, by design, does not.
(ii) **A different positive rate — a small contribution.** Duplicate handling lowered the positive rate from 0.430
(train.csv as downloaded) to 0.407, and the repeated texts are mostly news posts (52.6% positive among the 323 rows);
every model predicts more positives on the Kaggle test than on validation (baseline 0.320 vs 0.304), which helps F1 on
the positive class. (iii) **Threshold tuned on validation — ruled out:** 0.5 for every model. (iv) **Fitting on more
than the training part — ruled out:** the Kaggle files come from the training-part fit. (v) **Different preprocessing
— ruled out:** the Kaggle file goes through the same `clean_text()`.
The gap shrinks as the model gets stronger (+0.085, +0.079, +0.035): the duplicates help most the model that relies on
exact word overlap (TF-IDF n-grams) and least the pretrained encoder, which already gets most unseen tweets right.
So our local F1 is the conservative estimate of performance on genuinely new tweets, and the leaderboard number is
optimistic by roughly the share of tweets it has already seen.

## 한국어 요약 (팀 공유용)

- 세 모델 모두 리더보드가 로컬보다 높다 → 우리 코드의 누수가 아니라, 리더보드 쪽이 더 쉬운 시험이다.
- **주원인:** Kaggle test 의 9.9%(323행)는 URL만 다르고 training 과 같은 문장이다. 우리 val 에는 0행(중복 처리 때문).
  그 323행에서 모델 예측이 training 라벨과 90~95% 일치 → val 정확도(베이스라인 79%)보다 훨씬 높다. "이미 본 문제"가 섞인 셈.
- **작은 기여:** 중복 처리로 양성 비율이 0.430 → 0.407 로 낮아졌고, 겹치는 323행은 52.6%가 양성(뉴스 공유). test 에서 양성을 더 많이 예측.
- **해당 없음:** 기준값 조정(0.5 고정), 더 많은 데이터로 학습(training 부분만), 전처리 차이(같은 clean_text).
- 모델이 강할수록 차이가 작다(+0.085 → +0.035): 단어 겹침에 의존하는 TF-IDF 가 중복의 덕을 가장 많이 본다.
