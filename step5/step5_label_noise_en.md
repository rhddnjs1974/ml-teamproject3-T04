# Step 5. Label noise (draft)

## 5.1 Five tweets whose label looks wrong

Five tweets look mislabeled: three in the validation part (read in Step 4) and two in the training part (found with an LLM first pass, then read by hand). Each is judged with our two Step 1 reading rules (`eda/step1_rules.md`): a disaster word that comes with a real place, organisation or people means real; personal talk, jokes or names of games and songs mean not real.

**1. id 5213 (validation, keyword "fatality"): label 1, I would give 0**

> @Truly\_Stings Yo Dm me

The text names no event, place, agency or victim. The keyword is not even in the text, so the tweet was labeled 1 only because of how it was collected. Model B predicted 0, which is what a reader would say.

**2. id 1190 (validation, keyword "blizzard"): label 1, I would give 0**

> @Ashayo @MsMiggi Hi Ashayo! I believe there will be VODs on YouTube after the presentation but there is nothing like seeing it live :)

A chat about recordings of a presentation. There is no weather, no damage and no emergency, and the keyword does not appear in the text. This is personal talk under rule 2, so 0.

**3. id 4202 (validation, keyword "drowned"): label 1, I would give 0**

> Sadly before she could save humanity Ursula drowned in the drool of a protoshoggoth but at least she sort of died doing what she loved.

"Drowned" describes the death of a fictional character killed by an invented monster, told as a joke. No real person, place or event is involved, so 0.

**4. id 7903 (training, keyword "radiation emergency"): label 0, I would give 1**

> DOE's WIPP facility in NM investigating a site filter radiation reading; has activated its Emergency Ops Center; says no offsite release.

A named agency (DOE), a named facility and place (WIPP, New Mexico) and an activated emergency operations center describe a real incident. The severity is low, but rule 1 and rule 2 both point to real, so 1.

**5. id 613 (training, keyword "arsonist"): label 0, I would give 1**

> \#Vegetarian #Vegan Video shows arsonist torching popular BK restaurant Strictly Vegetarian... \[link\] #GoVegan #UniteBlue

The tweet reports a video of an arson at a named restaurant, which is a real fire set on purpose. The hashtags are promotional, but the event is real, so 1.

Extra cases for the appendix: ids 9557, 1789, 9593 (validation, label 1, I would give 0) and 4255, 4372 (training, label 0, I would give 1).

## 5.2 Label problems among the 30 Step 4 errors

14 of the 30 errors are label problems rather than model problems. The rule: a tweet counts when the reader judged that its label looks wrong and that the model's prediction looks right. Each of the three readers applied it to their own 10 tweets (5 false positives, 5 false negatives).

The 14 are 1 of the 15 false positives (no. 15) and 13 of the 15 false negatives (nos. 16 to 27 and 29). Four tweets are borderline: nos. 23, 25 and 26 are about terrorism, bioterror or the Iran deal without a real event, and no. 28 has the same topic but was filed as news quoting. The count depends on how the team treats them.

| Reading | Label problems out of 30 |
| --- | --- |
| Strict: borderline nos. 23, 25, 26 not counted | 11 |
| As assigned by the readers | 14 |
| Lenient: no. 28 counted too | 15 |

## 5.3 What label noise implies for the highest F1

Even a perfect classifier would score only about 0.86 F1 on our validation part, because about 10% of its rows carry a label that a reader judged wrong. Two pieces of evidence support this.

**1. Weighted estimate from the 30 errors.** Model B makes 206 validation errors (54 false positives, 152 false negatives). The 30 we read are 15 of each type, so the shares must be weighted: 1 of 15 false positives and 13 of 15 false negatives were label problems. That gives about 4 + 132 = 135 of the 206 errors (66%), or 9.8% of the 1,384 validation rows.

**2. Ceiling.** A classifier that matched the true label on every row would still disagree with the observed label on those 135 rows. Validation has 563 observed positives; about 132 of them should be 0. That leaves TP = 431, FN = 132 and FP = 4:

$$F1 = \frac{2 \, TP}{2 \, TP + FP + FN} = \frac{2 \cdot 431}{2 \cdot 431 + 4 + 132} \approx 0.86$$

The estimate is uncertain in two ways. The share 13/15 has a 95% interval of about 62% to 96%, which moves the ceiling between about 0.85 and 0.91. And it counts only the noise that Model B's errors exposed: mislabeled rows the model happens to agree with are invisible, so the real ceiling is lower. Model B's 0.80 is about 0.06 below this ceiling.

**3. Evidence that does not depend on the model.** train.csv has 300 groups of identical texts, and 66 of them (22%) carry both labels. "Hollywood movie about trapped miners released in Chile" appears 6 times and is labeled disaster once; "Reddit Will Now Quarantine Offensive Content" appears 7 times and is labeled disaster twice. When the same text gets both labels, no classifier can match all copies.

**Implication.** Differences of 0.01 to 0.03 F1 between our models are small next to this noise, and the room above Model B is limited. To check how far the LLM labels can be trusted, I read 30 training tweets blind (5 per direction and confidence cell, without seeing the LLM label) and compared my judgment with the LLM. We agreed on 9 of 10 at confidence 3 (90%, 95% interval 0.55 to 1.00), 3 of 10 at confidence 2 (30%, 0.07 to 0.65) and 6 of 10 at confidence 1 (60%, 0.26 to 0.88); overall 18 of 30 (60%, 0.41 to 0.77). This is agreement between my reading and the LLM, not accuracy against a true label, and the samples are small. Only confidence 3 looks reliable, so the Step 7 relabelling uses confidence 3 alone (604 rows: 17 changed from 0 to 1, 587 from 1 to 0).
