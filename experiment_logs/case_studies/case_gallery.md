# Native vs TSS Case Gallery

## 1. Vicuna-7B + EAGLE / Translation
- id: `sb_translation_192`
- skip layers: `[3, 25, 30]`
- BLEU: Native **0.142** → TSS **0.801**

**Problem**

Je dunkler das Fleisch , desto höher der ph-Wert .

**Native**

The darker the meat, the higher the pH value.

**TSS**

The darker the meat, the higher the pH.

**Reference**

The darker the meat , the higher the pH value .

---

## 2. Vicuna-7B + EAGLE / Translation
- id: `sb_translation_189`
- skip layers: `[3, 25, 30]`
- BLEU: Native **0.036** → TSS **0.506**

**Problem**

„ Das verändert die Zukunft meiner Familie “ , meinte der Mann .

**Native**

That changes the future of my family," said the man.

**TSS**

That changes the future of my family," the man said.

**Reference**

" This will change my future with my family , " the man said .

---

## 3. Vicuna-7B + EAGLE / Summarization
- id: `sum_289`
- skip layers: `[15, 22, 23]`
- ROUGE-L: Native **0.281** → TSS **0.430**

**Problem**

Arsenal must make a swift decision on Petr Cech this summer or risk losing out on the goalkeeper. The Chelsea shot-stopper looks certain to leave Stamford Bridge at the end of the season after losing the No 1 spot to Thibaut Courtois. Cech will hold talks with manager Jose Mourinho at the end of the season where the keeper will confirm his decision to leave west London. Arsenal must make a swift decision on Petr Cech this summer or risk losing out on the goalkeeper. The Chelsea stopper looks certain to leave Stamfo

**Native**

Arsenal must make a decision on signing Petr Cech from Chelsea this summer or risk losing out on the goalkeeper. Cech is likely to leave Chelsea at the end of the season after losing his No 1 spot to Thibaut Courtois. Arsenal is one of the clubs interested in signing Cech, but they will have to overcome several obstacl

**TSS**

Arsenal must make a decision on Petr Cech this summer or risk losing out to other clubs interested in the Chelsea goalkeeper. Cech is likely to leave Stamford Bridge at the end of the season after losing his No. 1 spot to Thibaut Courtois. Liverpool, Paris Saint-Germain, Roma, and Inter Milan are also interested in Cec

**Reference**

Arsenal risk missing out of Chelsea's Petr Cech is they hesitate on move. Cech looks set to leave Stamford Bridge after losing his No 1 spot. Liverpool, PSG, Roma and Inter Milan are all also interested in Cech. If the Gunners do not make their move early, they could be beaten to him. Chelsea are looking for a fee in excess of £10million for Czech keeper.

---

## 4. Vicuna-7B + EAGLE / Summarization
- id: `sum_263`
- skip layers: `[15, 22, 23]`
- ROUGE-L: Native **0.216** → TSS **0.354**

**Problem**

Atletico Madrid turned up the heat on Valencia in the battle for La Liga's third automatic Champions League berth with club top scorer Antoine Griezmann setting them on the way to a 2-0 stroll at bottom side Cordoba on Saturday. The victory for the champions at Cordoba's El Arcangel stadium in balmy Andalucia lifted Atletico above Valencia and onto 62 points from 29 matches. Valencia have 60 points in fourth and can reclaim third spot with a win at home to sixth-placed Villarreal on Sunday. France forward Griezmann

**Native**

Atletico Madrid beat Cordoba 2-0, with Antoine Griezmann scoring both goals, to move above Valencia in the battle for third place in La Liga. Saul Niguez also scored for Atletico, who now have 62 points from 29 matches. Valencia have 60 points and can reclaim third spot with a win over Villarreal on Sunday. Atletico al

**TSS**

Atletico Madrid defeated Cordoba 2-0 in a La Liga match, moving ahead of Valencia in the race for the third automatic Champions League spot. Antoine Griezmann scored both goals for Atletico, while Saul Niguez added a second. Cordoba suffered a 10th consecutive defeat and have no chance of making the playoffs. Sevilla, 

**Reference**

Atletico Madrid leapfrogged Valencia in race for third position. The La Liga duo are battling for automatic Champions League berth. Antoine Griezmann scored his 15th goal of the season in the fifth minute. Saul Niguez added a second before the interval to double his side's lead.

---

## 5. Vicuna-7B + EAGLE / QA
- id: `nq_35`
- skip layers: `[6, 8, 19, 21, 27]`
- F1: Native **0.000** → TSS **0.053**

**Problem**

what is the 180 degree line of longitude called

**Native**

The 180 degree line of longitude is called the International Date Line

**TSS**

The 180 degree line of longitude is called the "meridian". #

**Reference**

180th meridian or antimeridian

---

## 6. Vicuna-7B + EAGLE / QA
- id: `nq_49`
- skip layers: `[6, 8, 19, 21, 27]`
- F1: Native **0.000** → TSS **0.044**

**Problem**

georgia out of angus thongs and perfect snogging

**Native**

I'm sorry, but I'm not sure what you are asking. Could you please provide more context or clarify your question? The 2018-2023 World Outlook for 100% Vegetable and Fruit Juice Conc …

**TSS**

I'm not sure what the question is, but it seems to be about a movie titled "Angus Thong and Perfect Snogging" and if Georgia is in it. #

**Reference**

Georgia Groome as Georgia Nicolson

---

## 7. Vicuna-7B + EAGLE / MMLU
- id: `mmlu_prehistory_53`
- skip layers: `[3, 7, 9, 14, 20]`
- Acc: Native **0.000** → TSS **0.000**

**Problem**

The key factor in the survival and success of anatomically modern human beings was:
A. their ability to control fire and produce art.
B. their ability to interbreed with other hominids.
C. their ability to hunt large mammals.
D. their ability to adapt.

**Native**

A — A The 2018-2023 World Outlook for 100% Vegetable and Fruit Juice Concentrates …

**TSS**

A — A. their ability to control fire and produce art.

**Reference**

D

---

## 8. Vicuna-7B + EAGLE / MMLU
- id: `mmlu_machine_learning_35`
- skip layers: `[3, 7, 9, 14, 20]`
- Acc: Native **0.000** → TSS **0.000**

**Problem**

Suppose you are given an EM algorithm that finds maximum likelihood estimates for a model with latent variables. You are asked to modify the algorithm so that it finds MAP estimates instead. Which step or steps do you need to modify?
A. Expectation
B. Maximization
C. No modification necessary
D. Both

**Native**

A — A. Expectation The first time I saw the movie "The Secret Life of Walter Mitty," I was struck by the vivid imagery

**TSS**

A — A. Expectation 1. The term "cultural heritage" refers to which aspects of human societies and cultures?

**Reference**

B

---

## 9. Llama-2-13B + SAMD / Translation
- id: `sb_translation_240`
- skip layers: `[7, 11, 20, 31, 35, 38]`
- BLEU: Native **0.100** → TSS **0.328**

**Problem**

Ein Schuldspruch erfolgte außerdem in einer Reihe terrorismusbezogener Anklagepunkte , wozu eine Welle von Sprengstoffanschlägen auf Kirchen in Indonesien an Heiligabend im Jahr 2000 zählten .

**Native**

In addition to the sentence, a series of terrorism-related charges were brought, including a wave of bombings on churches in Indonesia on Christmas in the year 2000.

**TSS**

A judgment was also handed down in a series of terrorism-related charges, including a wave of bombings on churches in Indonesia on Christmas Eve in the year 2000.

**Reference**

He was also found guilty of a number of other terrorism-related charges , including a wave of bombings of churches across Indonesia on Christmas Eve in 2000 .

---

## 10. Llama-2-13B + SAMD / Translation
- id: `sb_translation_182`
- skip layers: `[7, 11, 20, 31, 35, 38]`
- BLEU: Native **0.065** → TSS **0.242**

**Problem**

Kerr und er selbst liebten sich noch immer , betonte der 36-Jährige .

**Native**

Kerr and he himself still loved each other, emphasized the 36-year-old.

**TSS**

He and himself loved each other still, emphasized the 36-year-old.

**Reference**

He and Kerr still love each other , emphasised the 36-year-old .

---

## 11. Llama-2-13B + SAMD / Summarization
- id: `sum_246`
- skip layers: `[12, 20, 27, 33, 34, 38]`
- ROUGE-L: Native **0.143** → TSS **0.280**

**Problem**

Campaign: Amanda Burleigh has won her battle. A midwife has won a decade-long battle to give mothers more time attached to their newborns before the umbilical cord is cut. Amanda Burleigh was convinced that clamping the cord within seconds of delivery – shutting off the blood supply from the placenta – was wrong. Since the 1950s, when doctors and midwives started giving women a hormonal injection to reduce the risk of haemorrhage, they have clamped the cord within seconds of birth. They feared the drug – since repl

**Native**

Midwife Amanda Burleigh has successfully campaigned for a change in medical guidelines regarding the timing of umbilical cord clamping after birth. For decades, the practice has been to clamp the cord within seconds of delivery, based on the belief that it reduces the risk of hemorrhage in mothers. However, Burleigh an

**TSS**

Midwife Amanda Burleigh has successfully campaigned for a change in medical practice by convincing the National Institute for Health and Care Excellence (NICE) to change its guidelines on cord clamping. The practice of clamping the umbilical cord within seconds of birth has been common practice since the 1950s, but Bur

**Reference**

Amanda Burleigh was convinced clamping cord immediately was wrong. Contacted other medics and they amassed evidence it could be harmful. National Institute for Health and Care Excellence guidelines have changed. Cord shouldn't be routinely clamped ‘earlier than one minute from the birth'

---

## 12. Llama-2-13B + SAMD / Summarization
- id: `sum_281`
- skip layers: `[12, 20, 27, 33, 34, 38]`
- ROUGE-L: Native **0.151** → TSS **0.288**

**Problem**

Chelsea have turned to the latest fitness technology to carry them over the line towards the Premier League title following a relentless campaign. Having led the chasing pack for most of the season, the Blues are currently  seven points clear with a game in hand over their closest rivals and seven games remaining ahead of the weekend's visit of Manchester United. Despite their commanding position, Jose Mourinho's side is leaving nothing to chance as players hit the gym following the announcement of a partnership wi

**Native**

Chelsea Football Club has partnered with Technogym, a leading wellness solutions provider, to enhance the team's fitness and performance. The partnership has seen Technogym kit out the club's Cobham training ground with state-of-the-art equipment. Chelsea's fitness coach Carlos Lalin has hailed the partnership, stating

**TSS**

Chelsea FC is using new fitness technology to help them in their quest for the Premier League title. The team has partnered with Technogym, a leading wellness solutions provider, to equip their training ground with state-of-the-art fitness equipment. Players such as Eden Hazard and Nemanja Matic have been pictured work

**Reference**

Chelsea are seven points clear at the top of the Premier League table. Blues have announced partnership with fitness leaders Technogym. Kurt Zouma, Eden Hazard and Nemanja Matic have been working out. Jose Mourinho's side host Man United at Stamford Bridge on Saturday.

---

## 13. Llama-2-13B + SAMD / QA
- id: `nq_30`
- skip layers: `[9, 10, 11, 19, 28, 38]`
- F1: Native **0.333** → TSS **0.750**

**Problem**

a request to the supreme court that it review a case that was already decided is made via a(n)

**Native**

Petition for Certiorari.

**TSS**

Petition for a Writ of Certiorari.

**Reference**

a writ of certiorari

---

## 14. Llama-2-13B + SAMD / QA
- id: `nq_48`
- skip layers: `[9, 10, 11, 19, 28, 38]`
- F1: Native **0.200** → TSS **0.500**

**Problem**

where is the new football stadium in atlanta

**Native**

The new football stadium in Atlanta is Mercedes-Benz Stadium. It is located in downtown Atlanta, Georgia, and is the home of the Atlanta Falcons of the National Football League (NFL).

**TSS**

The new football stadium in Atlanta is Mercedes-Benz Stadium.

**Reference**

Mercedes-Benz Stadium

---

## 15. Llama-2-13B + SAMD / MMLU
- id: `mmlu_professional_law_19`
- skip layers: `[7, 23, 24, 26, 28, 32, 33]`
- Acc: Native **0.000** → TSS **1.000**

**Problem**

An officer stopped a car for having a burned out headlight and license plate light. When the driver could not produce a driver's license, the officer asked him if he minded if they searched the vehicle. The officer did not advise the driver that he had a right to refuse consent or that if he refused his decision would be honored, but there was nothing duplicitous or coercive in the officer's conduct. The driver smiled and said "of course, no problem." The officer found stolen bank checks under the rear seat. In a later suppression motion, the driver claimed that his consent was not voluntary. What will the court most likely decide on that issue?
A. The consent was voluntarily given and was not the result of coercion or duress.
B. The consent was involuntary because it was the product of implied duress.
C. The consent was involuntary because the officer did not advise the driver of his right to refuse.
D. The consent became voluntary as a matter of law when the driver failed to produce a driver's license.

**Native**

C — Based on the information provided, the most likely decision of the court would be:

**TSS**

A — Based on the information provided, the correct answer is:

**Reference**

A

---

## 16. Llama-2-13B + SAMD / MMLU
- id: `mmlu_moral_scenarios_55`
- skip layers: `[7, 23, 24, 26, 28, 32, 33]`
- Acc: Native **0.000** → TSS **1.000**

**Problem**

For which of these two scenarios does the main character (who uses I/me/my) do something clearly morally wrong, according to ordinary moral standards in the US as of 2020? Scenario 1 | I picked the most rotten apples to donate to the food pantry. Scenario 2 | I sold exotic berries at the farmer's market.
A. Wrong, Wrong
B. Wrong, Not wrong
C. Not wrong, Wrong
D. Not wrong, Not wrong

**Native**

C — Based on ordinary moral standards in the US as of 2020, the main character does something clearly morally wrong in:

**TSS**

B — Based on ordinary moral standards in the US as of 2020, the main character does something clearly morally wrong in:

**Reference**

B

---
