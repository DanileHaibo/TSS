# Native vs TSS Qualitative Cases (5 domains × 7B/13B)

## Vicuna-7B + EAGLE / Translation
- id: `sb_translation_192`
- skip: `[3, 25, 30]`
- BLEU: Native **0.142** → TSS **0.801**

**Input**

Je dunkler das Fleisch , desto höher der ph-Wert .

**Native**

The darker the meat, the higher the pH value

**+TSS**

The darker the meat, the higher the pH

**Reference**

The darker the meat , the higher the pH value .

---

## Vicuna-7B + EAGLE / Summarization
- id: `sum_289`
- skip: `[15, 22, 23]`
- ROUGE-L: Native **0.281** → TSS **0.430**

**Input**

Arsenal must make a swift decision on Petr Cech this summer or risk losing out on the goalkeeper. The Chelsea shot-stopper looks certain to leave Stamford Bridge at the end of the season after losing the No 1 spot to Thibaut Courtois. Cech will hold talks with manager Jose Mourinho at the end of the season where the keeper will confirm his decision to leave west London. Arsenal must make a swift decision on Petr Cech this summer or risk losing out on the goalkeeper. The Chelsea stopper looks ce…

**Native**

Arsenal must make a decision on signing Petr Cech from Chelsea this summer or risk losing out on the goalkeeper. Cech is likely to leave Chelsea at the end of the season after losing his No 1 spot to Thibaut Courtois. Arsenal is one of the clubs interested in signing Cech, but they will have to overcome several obstacles, including Cech's valuation and Chels

**+TSS**

Arsenal must make a decision on Petr Cech this summer or risk losing out to other clubs interested in the Chelsea goalkeeper. Cech is likely to leave Stamford Bridge at the end of the season after losing his No. 1 spot to Thibaut Courtois. Liverpool, Paris Saint-Germain, Roma, and Inter Milan are also interested in Cech, and the Gunners must act quickly to s

**Reference**

Arsenal risk missing out of Chelsea's Petr Cech is they hesitate on move. Cech looks set to leave Stamford Bridge after losing his No 1 spot. Liverpool, PSG, Roma and Inter Milan are all also interested in Cech. If the Gunners do not make their move early, they could be beaten to him. Chelsea are l…

---

## Vicuna-7B + EAGLE / Open-domain QA
- id: `nq_57`
- skip: `[6, 8, 19, 21, 27]`
- F1: Native **0.043** → TSS **0.143**

**Input**

what age do you need to be to buy a bb gun

**Native**

In the United States, the minimum age to purchase a BB gun varies by state. Generally, it is legal to purchase a BB gun at the age of 18, but some states have set the minimum age at 16 or 17. It is important to check the laws in your state and the

**+TSS**

In the United States, the minimum age to legally purchase a BB gun is 18 years.

**Reference**

18

---

## Vicuna-7B + EAGLE / RAG
- id: `sb_rag_513`
- skip: `[3, 6, 14, 25, 30]`
- F1: Native **0.000** → TSS **0.310**

**Input**

the series is shot in Belgium, several local actors are featured: Veerle Baetens, Jurgen Delnaet, Joren Seldeslachts, Elsa Houben, Ben Forceville and Ben Van den Heuvel all appear in the series. Rebecca Ferguson who portrays Elizabeth Woodville, the White Queen, is from Sweden (her mother is originally from England). The budget was £25 million and took 120 days to shoot, consisting of 250 sets including: dungeons, palaces, castles, 12 state banquets and at least two coronations. Two versions we…

**Native**

The Royal was filmed in various locations in Belgium, including Bruges, Ghent, and Brussels. The Palace was filmed in Lithuania. The Crown was filmed in several locations in the UK, including London, Oxfordshire, and Wales

**+TSS**

The Royal was filmed in various locations in the UK, including Whitby, Scarborough, City of Bradford, and the North Riding of Yorkshire, as well as in Lithuania. The series was also filmed in part at Elstree Studios in Borehamwood, Hertfordshire, with approximately 25% of the fir

**Reference**

the North Riding of Yorkshire; City of Bradford; Scarborough; Whitby

---

## Vicuna-7B + EAGLE / MMLU
- id: `mmlu_prehistory_53`
- skip: `[3, 7, 9, 14, 20]`
- Acc.: Native **0.000** → TSS **0.000**

**Input**

The key factor in the survival and success of anatomically modern human beings was: A. their ability to control fire and produce art. B. their ability to interbreed with other hominids. C. their ability to hunt large mammals. D. their ability to adapt.

**Native**

A. … [off-task: “2018–2023 World Outlook for …”]

**+TSS**

A. their ability to control fire and produce art.

**Reference**

D

---

## Llama-2-13B + SAMD / Translation
- id: `sb_translation_240`
- skip: `[7, 11, 20, 31, 35, 38]`
- BLEU: Native **0.100** → TSS **0.328**

**Input**

Ein Schuldspruch erfolgte außerdem in einer Reihe terrorismusbezogener Anklagepunkte , wozu eine Welle von Sprengstoffanschlägen auf Kirchen in Indonesien an Heiligabend im Jahr 2000 zählten .

**Native**

In addition to the sentence, a series of terrorism-related charges were brought, including a wave of bombings on churches in Indonesia on Christmas in the year 2000.

**+TSS**

A judgment was also handed down in a series of terrorism-related charges, including a wave of bombings on churches in Indonesia on Christmas Eve in the year 2000.

**Reference**

He was also found guilty of a number of other terrorism-related charges , including a wave of bombings of churches across Indonesia on Christmas Eve in 2000 .

---

## Llama-2-13B + SAMD / Summarization
- id: `sum_246`
- skip: `[12, 20, 27, 33, 34, 38]`
- ROUGE-L: Native **0.143** → TSS **0.280**

**Input**

Campaign: Amanda Burleigh has won her battle. A midwife has won a decade-long battle to give mothers more time attached to their newborns before the umbilical cord is cut. Amanda Burleigh was convinced that clamping the cord within seconds of delivery – shutting off the blood supply from the placenta – was wrong. Since the 1950s, when doctors and midwives started giving women a hormonal injection to reduce the risk of haemorrhage, they have clamped the cord within seconds of birth. They feared…

**Native**

Midwife Amanda Burleigh has successfully campaigned for a change in medical guidelines regarding the timing of umbilical cord clamping after birth. For decades, the practice has been to clamp the cord within seconds of delivery, based on the belief that it reduces the risk of hemorrhage in mothers. However, Burleigh and her colleagues have argued that this p

**+TSS**

Midwife Amanda Burleigh has successfully campaigned for a change in medical practice by convincing the National Institute for Health and Care Excellence (NICE) to change its guidelines on cord clamping. The practice of clamping the umbilical cord within seconds of birth has been common practice since the 1950s, but Burleigh believed it was unnatural and harm

**Reference**

Amanda Burleigh was convinced clamping cord immediately was wrong. Contacted other medics and they amassed evidence it could be harmful. National Institute for Health and Care Excellence guidelines have changed. Cord shouldn't be routinely clamped ‘earlier than one minute from the birth'

---

## Llama-2-13B + SAMD / Open-domain QA
- id: `nq_30`
- skip: `[9, 10, 11, 19, 28, 38]`
- F1: Native **0.333** → TSS **0.750**

**Input**

a request to the supreme court that it review a case that was already decided is made via a(n)

**Native**

Petition for Certiorari.

**+TSS**

Petition for a Writ of Certiorari.

**Reference**

a writ of certiorari

---

## Llama-2-13B + SAMD / RAG
- id: `sb_rag_513`
- skip: `[8, 17, 20, 27, 33, 34]`
- F1: Native **0.061** → TSS **0.316**

**Input**

the series is shot in Belgium, several local actors are featured: Veerle Baetens, Jurgen Delnaet, Joren Seldeslachts, Elsa Houben, Ben Forceville and Ben Van den Heuvel all appear in the series. Rebecca Ferguson who portrays Elizabeth Woodville, the White Queen, is from Sweden (her mother is originally from England). The budget was £25 million and took 120 days to shoot, consisting of 250 sets including: dungeons, palaces, castles, 12 state banquets and at least two coronations. Two versions we…

**Native**

The royal TV series was filmed in various locations in the United Kingdom and Lithuania. Here are some of the specific locations where the series was shot:

**+TSS**

The Royal TV series was filmed in Lithuania and in the UK, specifically in Whitby, Scarborough, City of Bradford, and the North Riding of Yorkshire. The series was also filmed at Elstree Studios in Borehamwood, Hertfordshire. Locations in South Africa stood in for Kenya, and Ely 

**Reference**

the North Riding of Yorkshire; City of Bradford; Scarborough; Whitby

---

## Llama-2-13B + SAMD / MMLU
- id: `mmlu_professional_law_19`
- skip: `[7, 23, 24, 26, 28, 32, 33]`
- Acc.: Native **0.000** → TSS **1.000**

**Input**

An officer stopped a car for having a burned out headlight and license plate light. When the driver could not produce a driver's license, the officer asked him if he minded if they searched the vehicle. The officer did not advise the driver that he had a right to refuse consent or that if he refused his decision would be honored, but there was nothing duplicitous or coercive in the officer's conduct. The driver smiled and said "of course, no problem." The officer found stolen bank checks under…

**Native**

C. Based on the information provided, the most likely decision of the court would be:

**+TSS**

A. Based on the information provided, the correct answer is:

**Reference**

A

---
