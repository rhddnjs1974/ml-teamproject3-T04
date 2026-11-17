# Step 1 — 트윗 40개를 읽고 세운 규칙 (최재원)

읽은 트윗: `eda/samples_to_read.txt` (재난 20개 + 재난 아님 20개, seed 42). 모델 코드를 쓰기 전에 읽었다.

## 규칙 (원문)

1. 재난 단어가 장소나 조직, 사람들과 함께 쓰이면 real, 게임이나 노래 등에 들어간 이름이면 not real
2. 기관 이름 같은 게 들어가면 real, 개인적인 이야기면 not real

## Rules (English, for the report)

1. A disaster word used together with a place, an organisation, or a group of people means real; a disaster word that is part of a name (a game, a song, a venue) means not real.
2. A tweet that names an official body (agency, government, news outlet) is real; a tweet about the writer's personal life is not real.
