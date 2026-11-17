# Step 4 카테고리 정의 (초안 — 팀 확인 필요)

세 명이 각자 10개씩 읽고 붙인 이름을 통일한 결과다. 정의 문장과 대표 트윗은 초안이니 읽은 사람이 확인하고 고친다.
모델: 모델 B (DistilBERT). 개수 합계 30. 그림: `figures/fig2_categories.png`.

| 카테고리 (영문) | 한 줄 정의 | 개수 | 번호 | 대표 트윗 (id) |
|---|---|---|---|---|
| label appears wrong | 사람이 읽기에 붙은 라벨이 틀려 보이고 모델의 예측이 오히려 맞아 보이는 경우 | 14 | 15, 16~27, 29 | 19 "@Truly_Stings Yo Dm me" (5213, 라벨 1) |
| sarcasm or joke | 재난 단어가 농담이나 비꼬는 말 안에서 쓰인 경우 | 3 | 5, 12, 14 | 12 "…never be bush fires in Scotland as the ground is always soaking wet" (1994) |
| disaster word, no disaster event | 재난 단어가 글자 그대로의 뜻으로 쓰였지만 재난 사건은 아닌 경우 (철거, 재판, 통계 논쟁) | 3 | 7, 8, 10 | 7 "Enugu Government to demolish illegal structures…" (3255) |
| news or headline quoting | 뉴스 제목이나 기사 목록을 옮긴 형식이라 재난 여부가 제목만으로 갈리는 경우 | 3 | 6, 13, 28 | 13 "Emergency services unsure how to cope with loss of paging network" (4611) |
| metaphorical use of a disaster word | 재난 단어를 비유로 쓴 경우 (계획이 "탈선", 방화범과 소방대) | 2 | 1, 9 | 9 "…like inviting an arsonist to join the fire brigade" (643) |
| past or hypothetical disaster | 지금이 아니라 과거의 사태나 앞으로 생길 수 있는 위기를 말하는 경우 | 2 | 2, 3 | 2 "A look at state actions a year after Ferguson's upheaval" (10113) |
| too little information (spam, cut-off text) | 스팸이거나 문장이 잘려서 트윗만으로는 판단할 정보가 없는 경우 | 2 | 11, 30 | 30 "Jacksonville family bands together as memorial is planned for …" (4233) |
| ambiguous keyword or place name | 키워드가 본문에 없거나 지명만으로 재난을 떠올리게 하는 경우 | 1 | 4 | 4 "Blending the old with the new in #Vanuatu to prepare for future emergencies" (2846) |

## 수정 실험 대상 (가장 큰 두 카테고리)

- 1위: label appears wrong (14) → Fix 1: training 안에서 라벨이 의심스러운 행을 빼고 다시 학습
- 2위: 3개짜리 셋이 동률. **disaster word, no disaster event** 를 골랐다. 셋 다 거짓 양성이라 실패 방식이 하나이고,
  "모델이 문맥보다 재난 단어에 반응한다"는 가설을 keyword 가리기([MASK]) 한 가지 변경으로 직접 시험할 수 있기 때문.
  → Fix 2: 트윗 속 keyword 를 [MASK] 로 가리고 다시 학습

두 실험의 코드는 `src/fixes.py`, 결과는 `results.json` 의 `step4.fixes`.
