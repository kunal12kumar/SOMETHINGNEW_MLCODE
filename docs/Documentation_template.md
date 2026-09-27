# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Somethingnew  
**Team Members:** Ritik Lodhi, Suyash Rawat, Kunal Kumar, Yuvraj Singh  
**Submission Date:** 27-09-2026

---

## 1. Executive Summary
We normalise names and addresses (including romanising eight Indian scripts) and generate candidates with **state-blocked TF-IDF search** over Source 2 and Source 3 (25.7 candidates per S1, 97.5% validation recall). Each candidate pair is scored by two complementary models:
- a **two-stage LightGBM** on string-similarity and within-entity competition features
- a fine-tuned **multilingual cross-encoder** (MiniLM, Apache-2.0, 118M parameters) that reads both raw records together

In **v8-lite**, the cross-encoder score, with its rank and gap within the entity, is fed into the LightGBM as a feature, together with **distinctive-word features** (rare name words with no counterpart on the other side). Final matches apply a **one-owner rule** (each S2/S3 record belongs to at most one S1 entity, which holds for all 7.6M labelled records) and a threshold chosen on both normal validation and a **"stress" validation** that mimics the test set's higher density of look-alike businesses.

The submitted version (**v10**) keeps only the matches that v8-lite and v7 agree on. It never leaves an entity without a match that v8-lite gave it, and it fills an entity that would otherwise have no match when v7 and v6 agree on it.

Results on 30,000 held-out entities, macro F0.5 (stress in brackets):
- **v10, v9 + filling empty entities: public leaderboard [V10 LB]**
- v9, agreement of v8-lite and v7: public leaderboard 0.97071
- v8-lite: 0.9791 (0.9774), public leaderboard 0.96987
- v7, the two models combined by a stacker: 0.9776 (0.9759), public leaderboard 0.96986
- LightGBM alone: 0.9669 (0.9634), public leaderboard 0.9575

Only the provided files are used, with permissively licensed libraries and models.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** 2.21M train S1 and ~10.3M S2+S3 records; 1.73M test S1 and ~10.0M S2+S3. All-pairs comparison is impossible.
- **Match structure:** 5.6% singletons, 3.46 matches per S1 on average (0–11). **No S2/S3 record matches two S1 entities.**
- **Test differs from train:** 5.8 S2/S3 records per test S1 vs 4.7 in train, with similar true matches per S1. So test has roughly **2× more non-matching look-alikes ("siblings")**. France (15% of test) has no training labels.
- **Name noise:**
  - Legal-form variants (Pvt/Private, LLC/L.L.C., SARL/SAS/EURL/SASU).
  - Digit-for-letter typos (H0rizon, 5umit); bracket tags ([LP], (Center)); leading junk ("-- "); honorifics (Mr, Smt, M/s).
  - Duplicated words; phone numbers; web domains; initials only ("RS" = "Roubaix Sport SAS").
  - **Invented trade names** with no word overlap ("Nylajax" = "Ember Properties Inc").
  - Aliases ("X dba / d/b/a / t/a / trading as / aka / fka / formerly (known as) / doing business as Y", 7–22k names each).
  - 23% of India S2 names in native script.
- **Address noise:**
  - Abbreviations (St, Rd, Ave, R = Rue, Crs = Cours); number labels (H.No, Plot No, N°, #).
  - Leading zeros; reformatted numbers (4-02 = 402, 823B); extra numbers; component reordering.
  - State as code or name, or in native script; Telangana ↔ Andhra Pradesh swaps (19%).
  - French regions (S1) vs departments (S2/S3); 3–6% empty addresses.
- **What separates a true match from a sibling:** comparing test outputs of successive models showed two distinct kinds of sibling:
  - A **nearby house number** on the same street: "Verrent Comstock" at 14102 vs 4115.
  - A **different distinctive word in the name**, often at the same address: "Grenadiers **Sportive**" vs "Grenadiers **Batiment**", "Supreme **Construction**" vs "Supreme **Builders**".
  - Meanwhile true matches often have noisy numbers ("5527-D" = 5527, "Fno 2-01" = 201).

  Pure string-similarity features handle the second kind poorly. A model that reads the words does not.

### 2.2 Solution Strategy
**Approach Type:** Blocking + transformer cross-encoder + feature-based LightGBM that uses the cross-encoder score as a feature + one-owner assignment  
**Core Innovation:**
- State-blocked retrieval that combines name and address in one TF-IDF vector.
- Data-driven state repair: a city → state map learned from S1, and "neighbour" states learned from labels.
- Two scorers with different failure modes, combined by a stacker that also sees each score's rank and gap within the entity.
- Model selection on a **stress validation** (hard negatives repeated to match the test distractor density) plus test-output comparisons, because plain validation gains did not always carry over to the test set.

Generalisation to the unseen country: all LightGBM features are similarities or agree/conflict flags (no country one-hot, no raw words). `country` is only an open-set grouping key, and the cross-encoder is multilingual (it covers English, French and Indian scripts).

---

## 3. Candidate Generation (Blocking)
- **Normalisation first** (`normalize.py`):
  - Raw text is kept, and we add normalised name, core name (legal forms and honorifics removed), alias, initials, address, state, city and the set of address numbers.
  - Indian scripts are romanised with `indic-transliteration` (MIT), plus silent-vowel and nasal rules.
  - Audit on 73,531 labelled pairs, mean similarity: name 78.2 → 87.5 (India S2: 65.8 → 86.0); address 85.2 → 92.9.
- **Search paths**, per country label and per source (S2 and S3 kept separately), using `sparse_dot_topn`:
  - `addr`: word TF-IDF on the normalised address.
  - `both`: character 3-gram TF-IDF on the name (weight 0.6) concatenated with word TF-IDF on the address (weight 0.4).
- **Blocking key = state:** a query searches its own state, learned neighbour states and records with no state. This cut search cost about 20× and *raised* recall.
- **State repair:** city → state map from S1 (≥3 occurrences, ≥90% purity). French S2/S3 records with no state: 68% → 5.9%.
- **Submitted shortlist:** `addr` top-5 + `both` top-10 per source → **44,496,050 test pairs (25.7 per S1)**, validation recall 97.5%.
  - The larger setting (top-10 + top-20, 54.4 per S1, recall 98.2%) did not improve the leaderboard with LightGBM (0.957 vs 0.958).
  - It was also twice as expensive for the cross-encoder.

| Setting (validation, 30k S1) | Recall | Candidates / S1 |
|---|---|---|
| name + addr TF-IDF, K=20, no blocking | 96.9% | 82 |
| + combined name+address path | 97.2% | 45 |
| + state blocking | 97.4% | 38 |
| + city→state fill + neighbour states (addr 10, both 20) | 98.2% | 55 |
| **submitted: addr 5, both 10** | **97.5%** | **25.7** |

---

## 4. Matching Model

**Scorer 1: two-stage LightGBM (MIT)**
- **Features:**
  - Name: Levenshtein ratio, token-sort, token-set, partial and Jaro-Winkler on core names; ratio and token-set on normalised names; no-space ratio (domains); best alias score; exact core match; initials vs glued name; legal-form agreement; lengths; transliteration flag; log frequency of each core name in the pool.
  - Address: token-set, token-sort and partial token-set ratios; house-number set overlap, near-equality (dropped or extra digit), exact match, largest-number equality and edit ratio; city equality and ratio; state agree/conflict/missing; missing-address flags.
  - Retrieval: scores, ranks, number of paths, candidate source.
- **Stage 2** adds within-entity features computed from the stage-1 probability: rank, gap to the best, gap to the best of the same source, number of candidates, number scoring above 0.5.
- **Training:** 1,000,000 training-fold S1 entities (26.9M shortlist pairs, 12.9% positive), with hard negatives from our own retrieval; 800 rounds, 63 leaves, learning rate 0.05.

**Scorer 2: cross-encoder**
- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters, ≤ 8B), fine-tuned as a binary pair classifier with a one-output head and BCE loss.
- **Input:** the raw text of both records, `"name | address"` for each, as a sentence pair (max 96 tokens).
- **Training:** 3,000,000 shortlist pairs from 150,000 training-fold entities, each shown in both orders (6M examples); 1 epoch, batch 256, learning rate 5e-5 with linear warm-up, bf16 on one A100 (35 min).
- Scoring the 44.5M test pairs took 110 minutes.

**Final scorer, v8-lite (submitted):** the two-stage LightGBM above, retrained with extra inputs:
- **Cross-encoder features:** the CE probability, its rank within the S1 entity, gap to the entity's best CE score, gap to the best of the same source, and the number of candidates the CE scores above 0.5. `ce0_gap` is the second most important feature after the stage-1 probability.
- **Distinctive-word features:**
  - For each pair, the core-name words with no fuzzy counterpart (ratio ≥ 80) on the other side.
  - The largest and summed rarity (IDF, computed over the split's own S2+S3 names) of those words.
  - How many there are on each side.

  This targets siblings such as "Great **Ventures**" vs "Great **Infra**". On LightGBM alone it added +0.005 (stress +0.0056).
- Support and address-frequency features are left out: they hurt the leaderboard in v4.
- **Training data:**
  - 99,062 training-fold entities never seen by v6 or the cross-encoder, drawn from a fixed random order with offset 1M.
  - We only used entities whose candidates all have cross-encoder scores: 3 chunks, the GPU budget we had.
  - The same candidate set as v7 (address top-5 + combined top-10), 3.37M pairs; 300 rounds, learning rate 0.1.
- The same 30,000 validation entities as all earlier versions, never used for training.

**Stacker (v7):**
- A small LightGBM (300 rounds, 31 leaves) on:
  - the two probabilities, their product and difference
  - each model's rank, gap to best and gap to best-of-source within the S1 entity
  - the number of candidates each model scores above 0.5
- Trained on the 30,000 validation entities with 5-fold cross-validation grouped by entity. That gives out-of-fold scores for model selection; the final stacker is then fit on all validation pairs.
- The cross-encoder and LightGBM never saw these entities.

**Threshold selection method:**
- One-owner rule first: each S2/S3 record is kept only for its highest-scoring S1.
- The threshold maximises **stress** macro F0.5 among thresholds within 0.0005 of the best **normal** macro F0.5.
- **Stress validation** repeats every hard negative (a non-match that either model scores above 0.3), so look-alikes are twice as common, as observed on test.
- Chosen: **v8-lite, threshold 0.75** (v7: stacker, threshold 0.80).

**Final decision, v9 and v10 (submitted):**
- Keep a v8-lite match only if v7 also made it.
- Exception: if that would leave the entity with no match, keep v8-lite's matches.

The reason is F0.5:
- For an entity with several matches, adding a pair helps only if it is right more than ~75% of the time.
- A pair that one strong model accepts and the other rejects is a borderline case, and those are right less often, especially on test with its denser look-alikes.
- For an entity whose only match would be removed, the entity scores 0 or 1, so a match right more than half of the time is worth keeping.

On test:
- Both models agree on 5,634,031 matches.
- 66,285 v8-lite-only matches are dropped.
- 3,482 entities keep v8-lite's matches to avoid an empty row.
- 5,637,800 matches remain.

v10 fills entities that would otherwise have no match:
- An entity with no match scores 1 if it truly has none and 0 otherwise, so a single match right more than half the time is worth adding.
- 1,063 of the 104,025 empty entities get v7's matches that v6 also made, when no other entity owns them.
- v9's leaderboard gain showed that disputed matches are right roughly 70% of the time on test.

**Validation design:** entity-grouped split (10% of S1 by a hash of the ID). Validation entities are never used to train LightGBM or the cross-encoder.

---

## 5. Results & Error Analysis

Validation on 30,000 held-out S1 entities (official macro F0.5, singletons included):

| Method | Normal | Stress (2× look-alikes) |
|---|---|---|
| LightGBM alone (v6 rule, threshold 0.75) | 0.9669 | 0.9634 |
| Cross-encoder alone (best threshold) | 0.9705 | 0.9680 |
| Fixed blend 0.7·CE + 0.3·LightGBM | 0.9766 | 0.9746 |
| Stacker, threshold 0.80 (v7) | 0.9776 | 0.9759 |
| LightGBM + CE features + word features, threshold 0.75 (v8-lite) | 0.9791 | 0.9774 |

History of leaderboard submissions:

| Version | Main change | Validation | Public LB |
|---|---|---|---|
| v2 | base LightGBM features | 0.9668 | 0.957 |
| v4 | + "support" features, address frequency, 500k training entities | 0.9711 | 0.955 |
| v6 | v2 features + normalisation fixes + 1M training entities + 25.7/S1 shortlist | 0.9669 | 0.9575 |
| v7 | v6 + cross-encoder + stacker | 0.9776 | 0.96986 |
| v8-lite | cross-encoder features + distinctive-word features inside LightGBM | 0.9791 | 0.96987 |
| v9 | matches both v8-lite and v7 agree on (never emptying an entity) | n/a | 0.97071 |
| **v10 (final)** | **v9 + empty entities filled where v7 and v6 agree** | n/a | **[V10 LB]** |

- **Common false positives (wrong merges):**
  - Siblings with the same name at a nearby house number.
  - Different businesses at the same address whose names differ by one distinctive word.
  - The cross-encoder removes many of the second kind: v7 dropped "Grenadiers Sportive SASU" (vs "Grenadiers Batiment SASU") and "Supreme Construction" (vs "Supreme Builders") that LightGBM had accepted.
- **Common false negatives (missed matches):**
  - Exact-name records with an **empty address** ("Olanium Hóspital" | '').
  - Abbreviated legal words ("Lille Cie" = "Lille Compagnie").
  - Pairs never retrieved (2.5% at 25.7/S1), mostly short native-script names with very short addresses.
- **Lessons learned:**
  - Features that compare a candidate with the entity's other candidates ("support") improved validation but hurt the leaderboard. On test they pulled in siblings.
  - Selecting models on a stress validation, and reading test-output differences, protected us from repeating that mistake.
  - v8-lite's +0.0015 on validation gave only +0.00002 on the leaderboard: the remaining test errors are mostly look-alike businesses that our validation split has fewer of.

---

## 6. Conclusion
Careful normalisation and state-blocked name+address retrieval give a compact candidate set (25.7 per entity). A feature-based LightGBM and a multilingual cross-encoder are complementary: the first is strong on structured signals such as numbers, city and state, the second on reading distinctive words, transliterations and trade names. Feeding the cross-encoder score into the LightGBM as a feature, together with distinctive-word features, reaches macro F0.5 0.979 (stress 0.977) on held-out data. The main lesson: in entity resolution with look-alike businesses, **validation must mimic the test distractor density**, and model changes should be checked on test outputs, not only on validation scores.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: all source is in `src/ber/`. `README.md` gives the exact command order:
1. preprocess → fill_state → splits → state_neighbors → name_freq
2. run_candidates (train, 1M) → matcher train (LightGBM, `models/v6`) → run_candidates (test) → predict (LightGBM scores + shortlist)
3. run_candidates (train, 150k) → export_ce → cross_encoder train / valid / test
4. stack: this gave v7.
5. v8-lite:
   - run_candidates (train, 300k, offset 1M) → word_idf
   - cross-encoder scores for the first 3 chunks of training pairs → ce_prep from-chunks / merge
   - matcher train (`--extra-scores --require-extra --word-idf`, `models/v8lite`)
   - predict
6. v9 / v10: combine (v7 AND v8-lite, `--fill` v6) → `output/matching_results.tsv`, `output/candidate_pairs.tsv`

Key modules:
- `normalize.py`, `lexicon.py`: normalisation.
- `candidates.py`: retrieval.
- `features.py`: pair and group features.
- `matcher.py`: LightGBM.
- `cross_encoder.py`: transformer pair model.
- `stack.py`: stacker, stress validation, final decision.
- `metric.py`: official macro F0.5.

Trained models ship in `models/`:
- `v8lite`: the final LightGBM
- `ce_model`: the cross-encoder
- `v6` (the step-2 LightGBM, used for the shortlist and the stacker inputs) is not shipped; the README's step 7 recreates it

**Compliance:**
- No external data, APIs, geocoding or registries. All rule lists are hand-written conventions, and all learned maps come from the provided files.
- Licences:
  - LightGBM (MIT)
  - PyTorch (BSD)
  - Hugging Face Transformers (Apache-2.0)
  - MiniLM cross-encoder base (Apache-2.0, 118M parameters)
  - indic-transliteration (MIT)
  - RapidFuzz (MIT)
  - sparse_dot_topn (Apache-2.0)
  - scikit-learn (BSD)

### B. Additional Results
- Cross-encoder training loss: 0.169 (step 500) → 0.015 (end); in-distribution validation F0.5 of the cross-encoder alone: 0.9705.
- Stacker stress F0.5 by threshold: 0.60 → 0.9742, 0.70 → 0.9757, 0.80 → 0.9759, 0.90 → 0.9749. Flat, so the result is robust to the exact threshold.
- Test output of v7: 5,680,877 matches; 93.9% of S1 entities with at least one match; 3.49 matches per matched entity.
- Runtime: preprocessing ~4 min and test candidates ~20 min on 44 CPU cores; LightGBM training ~25 min; cross-encoder training 35 min and test scoring 110 min on one A100.
