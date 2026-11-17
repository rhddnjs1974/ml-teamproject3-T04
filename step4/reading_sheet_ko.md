# Step 4 읽기 자료 — 모델 B(DistilBERT)가 틀린 트윗 30개

- 뽑은 규칙: 모델 B의 validation 예측 중 거짓 양성 15개 + 거짓 음성 15개, seed 42 (`src/errors.py`)
- **거짓 양성(FP)**: 정답은 0(재난 아님)인데 모델이 1(재난)이라고 함
- **거짓 음성(FN)**: 정답은 1(재난)인데 모델이 0(재난 아님)이라고 함
- 번역과 참고 설명은 Claude로 만든 읽기 보조 자료다 (보고서 AI 표에 기록). **카테고리 판단은 읽는 사람이 직접 한다.**
- 원문은 `step4/errors_30.csv`, 결과는 `step4/categories.csv`의 `category` 칸에 적는다.

## 할 일

1. 내 몫 10개를 원문과 번역으로 읽는다.
2. 각 트윗을 왜 모델이 틀렸는지 생각해서 카테고리를 하나 붙인다. 카테고리는 셋이 합의해서 4개 이상 정의한다.
   - PDF 예시(바꿔도 됨): 재난 단어의 비유적 사용 / 뉴스·헤드라인 인용 / 애매한 키워드나 지명 / 비꼬기·농담 / 사람이 봐도 애매함 / 라벨이 틀린 것 같음
3. "라벨이 틀린 것 같음"으로 본 트윗은 민성이의 Step 5 재료가 된다.

## 분담 (각자 FP 5개 + FN 5개)

| 읽는 사람 | 번호 |
|---|---|
| 최재원 | 1~5, 16~20 |
| 김민성 | 6~10, 21~25 |
| 차승준 | 11~15, 26~30 |

---

## 최재원 (1~5, 16~20)

**1. [id 3460] FP · 키워드 derailed**
> #Adani & #Modi plan for mining derailed! Australia court blocks huge India-backed coal mine

번역: 아다니와 모디의 광산 계획이 탈선(무산)됐다! 호주 법원이 인도가 지원하는 대형 석탄 광산을 막았다
참고: Adani는 인도 대기업, Modi는 인도 총리. "derailed"는 "계획이 무산됐다"는 뜻

**2. [id 10113] FP · 키워드 upheaval**
> A look at state actions a year after Ferguson's upheaval

번역: 퍼거슨 소요 사태 1년 뒤 주 정부들의 조치를 살펴본다
참고: 2014년 미국 퍼거슨에서 경찰 총격 이후 일어난 시위와 폭동

**3. [id 10111] FP · 키워드 upheaval**
> USW: 'The damage from abandoning the deal could well create a new level of uncertainty...economic upheaval & military unrest'

번역: USW: "협정을 파기하면 생길 피해는 새로운 수준의 불확실성… 경제적 격변과 군사적 불안을 낳을 수 있다"
참고: USW는 미국 철강노조. 이란 핵협정 관련 발언으로 보임

**4. [id 2846] FP · 키워드 cyclone**
> Blending the old with the new in #Vanuatu to prepare for future emergencies:

번역: 바누아투에서 미래의 비상사태에 대비하기 위해 옛것과 새것을 섞다
참고: 바누아투는 2015년 사이클론 팸 피해를 입은 남태평양 섬나라. 본문에 "cyclone"은 없음

**5. [id 8968] FP · 키워드 storm**
> this storm????

번역: 이 폭풍 뭐야????
참고: 맥락 없이 한 줄

**16. [id 4202] FN · 키워드 drowned**
> Sadly before she could save humanity Ursula drowned in the drool of a protoshoggoth but at least she sort of died doing what she loved.

번역: 안타깝게도 우르술라는 인류를 구하기 전에 프로토쇼고스의 침에 빠져 익사했다. 그래도 좋아하는 일을 하다 죽은 셈이다
참고: 쇼고스는 러브크래프트 소설 속 괴물. 소설이나 게임 이야기

**17. [id 10289] FN · 키워드 weapon**
> Please allow me to reiterate it's not the weapon it's the mindset of the individual! #professional #help! -LEGION!

번역: 다시 말하지만 문제는 무기가 아니라 개인의 사고방식이다! #전문가 #도움! -LEGION!
참고: 총기 규제 논쟁에서 자주 나오는 주장

**18. [id 5996] FN · 키워드 hazardous**
> #foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption

번역: #식품공포 네슬레 인도, 매기 라면 판매 금지 후 적자로 전환. 인간이 먹기에 안전하지 않고 위험함
참고: 2015년 인도에서 네슬레 매기 라면이 납 검출로 판매 금지된 사건. **Step 1 분석에서 이 문장은 라벨이 엇갈린 중복(1, 1, 0)이었음 → 다수결로 1**

**19. [id 5213] FN · 키워드 fatality**
> @Truly_Stings Yo Dm me

번역: (답장) 야 DM 보내
참고: 재난과 관련된 내용 없음

**20. [id 9557] FN · 키워드 threat**
> Meek Mill responds to Drake's OVO Fest set with wedgie threat

번역: 믹 밀이 드레이크의 OVO 페스티벌 공연에 "똥침(웨지) 협박"으로 응수했다
참고: 두 래퍼의 디스전 연예 기사. "wedgie"는 바지를 잡아당기는 장난

---

## 김민성 (6~10, 21~25)

**6. [id 1175] FP · 키워드 blight**
> Article by Michael Jackman at Metro Times Detroit: The group later downgraded the estimate to 37 square miles of...

번역: 메트로 타임스 디트로이트의 마이클 잭먼 기사: 그 단체는 나중에 추정치를 37제곱마일로 낮췄다…
참고: "blight"는 디트로이트의 버려진 빈집·폐허 지역(도시 황폐화)을 말함. 문장이 중간에 잘림

**7. [id 3255] FP · 키워드 demolish**
> Enugu Government to demolish illegal structures at International Conference Centre

번역: 에누구 주 정부가 국제회의장의 불법 건축물을 철거할 예정
참고: 에누구는 나이지리아의 주

**8. [id 10650] FP · 키워드 wounds**
> RT @DianneG: Gunshot wound #9 is in the bicep. only 1 of the 10 wounds that is not in the chest/torso area. #KerrickTrial #JonathanFerrell

번역: (리트윗) 9번째 총상은 이두근에 있다. 10개 상처 중 가슴·몸통이 아닌 유일한 곳이다
참고: 경찰관 Kerrick이 비무장 흑인 Jonathan Ferrell을 총으로 쏜 사건의 재판 중계

**9. [id 643] FP · 키워드 arsonist**
> Trusting Iran to stop terrorism is like inviting an arsonist to join the fire brigade - Telegraph

번역: 이란이 테러를 막으리라 믿는 것은 방화범을 소방대에 들이는 것과 같다 - 텔레그래프
참고: 영국 신문 칼럼 제목. 방화범·소방대는 비유

**10. [id 6505] FP · 키워드 injuries**
> @nalathekoala As a health care professional that deals all gun violence sequalae I consider suicides injuries accidents and homicides

번역: (답장) 총기 폭력의 후유증을 다루는 의료인으로서, 나는 자살·부상·사고·살인을 모두 (총기 폭력에) 포함해서 본다
참고: 총기 통계 논쟁 중 의견

**21. [id 9593] FN · 키워드 thunder**
> @AsimTanvir @NemesisK_PES @KevR_7 Snow? Thunder and lightning?

번역: (여러 명에게 답장) 눈? 천둥 번개?
참고: PES는 축구 게임(위닝일레븐). 앞 대화 맥락 없음

**22. [id 1789] FN · 키워드 buildings on fire**
> Such beautiful architecture in #NYC I love those fire escape routes on the buildings. #newyork

번역: 뉴욕 건축물 정말 아름답다. 건물에 붙은 비상 탈출용 계단(fire escape)이 너무 좋아
참고: "fire escape"는 건물 외벽의 철제 비상계단

**23. [id 867] FN · 키워드 bioterror**
> USATODAY: On today's #frontpage: #Bioterror lab faced secret sanctions. #RickPerry doesn't make the cut for FoxNew…

번역: USA투데이 오늘 1면: 생물테러 연구소가 비밀 제재를 받았다. 릭 페리는 폭스뉴스 (토론회) 출전 명단에 못 들었다…
참고: 신문 1면 기사 목록

**24. [id 915] FN · 키워드 bioterrorism**
> @HowardU If 90BLKs&8WHTs colluded 2 take WHT F @USAgov AUTH Hostage&2 make her look BLK w/Bioterrorism&use her lgl/org IDis ID still hers?

번역: (하워드 대학에) 흑인 90명과 백인 8명이 공모해서 미국 정부 권한으로 백인 여성을 인질로 잡고, 생물테러로 그녀를 흑인처럼 보이게 만들고 그녀의 법적 신분을 쓴다면, 그 신분은 여전히 그녀의 것인가?
참고: 약어투성이의 음모론성 트윗. 실제 사건 여부 불분명

**25. [id 886] FN · 키워드 bioterrorism**
> @DarrellIssa Does that 'great Iran deal' cover bioterrorism? You got cut off terrible of them. Keep up the good work.

번역: (대럴 아이사 의원에게) 그 "훌륭한 이란 협정"이 생물테러도 다루나요? 발언이 끊기셨네요, 그들 참 너무하네요. 계속 힘내세요
참고: 정치인에게 보내는 비꼬는 응원

---

## 차승준 (11~15, 26~30)

**11. [id 2866] FP · 키워드 damage**
> Reusing advanced in life equipments in transit to drumming champaign damage: FdbDP

번역: (뜻이 통하지 않는 문장) "오래된 장비를 재사용하는 중 드럼 샴페인 손상" 정도
참고: 자동 생성된 스팸 문장으로 보임

**12. [id 1994] FP · 키워드 bush fires**
> One thing you can be sure of. There will never be bush fires in Scotland as the ground is always soaking wet????

번역: 확실한 한 가지. 스코틀랜드에선 땅이 늘 흠뻑 젖어 있어서 산불이 날 일은 절대 없다????
참고: 날씨 농담

**13. [id 4611] FP · 키워드 emergency services**
> Emergency services unsure how to cope with loss of paging network

번역: 긴급 구조대가 무선호출(삐삐) 네트워크 중단에 어떻게 대처할지 몰라 난감해하고 있다
참고: 영국 뉴스. 통신망 중단 기사

**14. [id 1767] FP · 키워드 buildings burning**
> @PPFA At least they aren't burning buildings and looting stores.

번역: (미국 가족계획연맹에게) 적어도 그 사람들은 건물을 불태우거나 상점을 약탈하지는 않잖아
참고: 시위 비교 논쟁. Step 1에서 읽은 비슷한 트윗(id 1753)은 정답 1이었음

**15. [id 6821] FP · 키워드 loud bang**
> @SW_Trains strange loud impact bang noises under train to Epsom about to arrive #Wimbledon

번역: (사우스웨스트 철도에게) 곧 도착하는 엡섬행 열차 밑에서 이상하게 큰 충격음이 쾅쾅 납니다 #윔블던
참고: 승객이 철도 회사에 실시간 신고

**26. [id 9484] FN · 키워드 terrorism**
> @KurtSchlichter He's already done it by negotiating with the #1 state of terrorism in the World. What was his hurry in trying to get a deal

번역: (답장) 그는 세계 1위 테러 국가와 협상함으로써 이미 그렇게 했다. 왜 그렇게 서둘러 협정을 맺으려 했나
참고: 이란 핵협정 비판 정치 의견

**27. [id 9263] FN · 키워드 sunk**
> @SaintRobinho86 someone has to be at the bottom of every league. Tonight clearly demonstrated why the Lions are where they are - sunk!

번역: (답장) 어느 리그든 누군가는 꼴찌를 해야지. 오늘 밤 경기로 라이온스가 왜 그 순위인지 확실히 보여 줬다 - 침몰!
참고: 스포츠 팀 이야기. "sunk"는 비유

**28. [id 7260] FN · 키워드 nuclear disaster**
> #Obama signed up to a deal that far from making the world a safer place … via @upi #Iran #Nuclear #IranNuclearDeal

번역: 오바마는 세상을 더 안전하게 만들기는커녕 (위험하게 만드는) 협정에 서명했다 (UPI 경유) #이란 #핵 #이란핵협정
참고: 이란 핵협정 비판 기사 공유

**29. [id 1190] FN · 키워드 blizzard**
> @Ashayo @MsMiggi Hi Ashayo! I believe there will be VODs on YouTube after the presentation but there is nothing like seeing it live :)

번역: (답장) 안녕 아샤요! 발표 끝나면 유튜브에 다시보기 영상이 올라올 거야. 그래도 생방송으로 보는 게 최고지 :)
참고: "Blizzard"는 게임 회사 이름(블리자드). 본문에는 재난 내용 없음

**30. [id 4233] FN · 키워드 drowned**
> Jacksonville family bands together as memorial is planned for ...

번역: 잭슨빌의 한 가족이 (익사한 사람의) 추모식을 준비하며 서로 힘을 모으고 있다…
참고: 지역 뉴스 기사. 문장이 중간에 잘림
