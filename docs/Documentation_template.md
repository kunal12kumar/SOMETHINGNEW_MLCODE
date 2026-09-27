# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Somethingnew  
**Team Members:** Ritik Lodhi, Suyash Rawat, Kunal Kumar, Yuvraj Singh  
**Submission Date:** 27-09-2026

---

## 1. Executive Summary
We normalise names and addresses (including romanising eight Indian scripts), generate candidates with **state-blocked TF-IDF search** over Source 2 and Source 3, and score each pair with a **two-stage LightGBM** on string-similarity and within-entity competition features. Final matches use a **one-owner rule** (each S2/S3 record goes to at most one S1 entity, which holds for all 7.6M labelled records) and a threshold chosen for precision under the test set's higher distractor density. The submitted candidate set has **25.7 candidates per S1** with 97.5% validation recall. The best public leaderboard score is **0.9575** (validation macro F0.5 0.967). Only the provided files are used, together with permissively licensed libraries and models.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** 2.21M train S1 and ~10.3M S2+S3 records; 1.73M test S1 and ~10.0M S2+S3. All-pairs comparison is impossible.
- **Match structure:** 5.6% singletons, 3.46 matches per S1 on average (0–11). **No S2/S3 record matches two S1 entities.**
- **Test differs from train:** 5.8 S2/S3 records per test S1 vs 4.7 in train, with similar true matches per S1. So test has roughly **2× more non-matching look-alikes ("siblings")**: the same or similar name at a nearby house number on the same street, or a different business at the same address. France (15% of test) has no training labels.
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

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier + one-owner assignment (optional cross-encoder rescoring)  
**Core Innovation:**
- State-blocked retrieval that combines name and address in one TF-IDF vector.
- Data-driven state repair: a city → state map learned from S1, and "neighbour" states learned from labels.
- A second LightGBM stage that sees how each candidate compares with the other candidates of the same entity.
- Decisions (threshold, feature set, shortlist size) checked against **test-set behaviour**, not only validation. Validation gains did not always carry over, so we compared test outputs before each upload.

Generalisation to the unseen country: every feature is a similarity or an agree/conflict flag, with no country one-hot and no raw words. `country` is only an open-set grouping key, and all learned maps come from the provided files of the same split.

---

## 3. Candidate Generation (Blocking)
- **Normalisation first** (`normalize.py`):
  - Raw text is kept, and we add normalised name, core name (legal forms and honorifics removed), alias, initials, address, state, city and the set of address numbers.
  - Indian scripts are romanised with `indic-transliteration` (MIT), plus silent-vowel and nasal rules.
  - "St"/"Street"/"Saint" share one token.
  - Audit on 73,531 labelled pairs, mean similarity: name 78.2 → 87.5 (India S2: 65.8 → 86.0); address 85.2 → 92.9.
- **Search paths**, per country label and per source (S2 and S3 kept separately), using `sparse_dot_topn`:
  - `addr`: word TF-IDF on the normalised address.
  - `both`: character 3-gram TF-IDF on the name (weight 0.6) concatenated with word TF-IDF on the address (weight 0.4).
  - A name-only path was dropped: 17× slower, with little extra recall.
- **Blocking key = state:** a query searches its own state, its learned neighbour states and records with no state. This cut search cost about 20× and *raised* recall, because out-of-state look-alikes no longer fill the top-K.
- **State repair:** city → state map from S1 (≥3 occurrences, ≥90% purity). French S2/S3 records with no state: 68% → 5.9%.
- **Final shortlist:** `addr` top-5 + `both` top-10 per source → **44,496,050 test pairs (25.7 per S1)**. The larger setting (top-10 + top-20, 54.4 per S1) scored lower on the leaderboard (0.957 vs 0.958).
- **How true matches were kept:** recall@K measured on 30,000 held-out S1 after every change.

| Setting (validation, 30k S1) | Recall | Candidates / S1 |
|---|---|---|
| name + addr TF-IDF, K=20, no blocking | 96.9% | 82 |
| + combined name+address path | 97.2% | 45 |
| + state blocking | 97.4% | 38 |
| + city→state fill + neighbour states (addr 10, both 20) | 98.2% | 55 |
| **submitted: addr 5, both 10** | **97.5%** | **25.7** |

---

## 4. Matching Model

**Features used** (RapidFuzz, vectorised in C++):
- **Name features:**
  - Levenshtein ratio, token-sort, token-set, partial ratio and Jaro-Winkler on core names.
  - Ratio and token-set on normalised names; no-space ratio (domains).
  - Best score against aliases; exact core match; initials vs glued name; legal-form agreement.
  - Name lengths; transliteration flag; log frequency of each core name in the pool.
- **Address features:**
  - Token-set, token-sort and partial token-set ratios.
  - House numbers: set overlap, near-equality (dropped or extra digit), exact match, largest-number equality and edit ratio.
  - City equality and ratio; state agree/conflict/missing; missing-address flags.
- **Other:**
  - Retrieval scores, ranks and number of paths; candidate source.
  - **Stage-2 within-entity features** from the stage-1 probability: rank, gap to the best, gap to the best of the same source, number of candidates, number scoring above 0.5.

**Model type:** two-stage LightGBM (MIT).
- Stage 1 uses 3-fold out-of-fold predictions grouped by S1. Stage 2 takes the pair features, the stage-1 probability and the group features.
- Settings: 800 rounds, 63 leaves, learning rate 0.05, bagging 0.8, minimum 200 rows per leaf.
- Trained on 1,000,000 training-fold S1 entities (26.9M shortlist pairs, 12.9% positive). The negatives are realistic hard negatives from our own retrieval.

**Threshold selection method:**
- One-owner rule first: each S2/S3 record is kept only for its highest-scoring S1.
- Then a fixed threshold. On validation, macro F0.5 is flat from 0.60 to 0.75 (best 0.65). We chose **0.75** because test has about 2× more distractors. A simulation that doubles validation distractors moves the optimum from 0.70 to 0.75.

**Validation design:** entity-grouped split (10% of S1 by a hash of the ID). An entity and all its matches fall entirely in train or entirely in validation.

**[Optional stage, if it improves results] Cross-encoder:**
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters), fine-tuned as a pair classifier on 3M shortlist pairs, shown in both orders.
- Input is the raw `name | address` of both records.
- It is combined with LightGBM by a small stacking model, chosen on normal validation and on a "stress" validation where hard negatives are repeated to match the test distractor density. Results: [CE RESULTS].

---

## 5. Results & Error Analysis

| Version | Main change | Validation macro F0.5 | Public LB |
|---|---|---|---|
| v2 | base features + near-equal numbers, name frequency | 0.9668 | 0.957 |
| v4 | + "support" features, address frequency, normalisation fixes, 500k training entities | 0.9711 | 0.955 |
| **v6** | v2 features + normalisation fixes + 1M training entities + 25.7/S1 shortlist + threshold 0.75 | 0.9669 | **0.958** |

- **F_0.5 Score (macro, validation):** 0.967 (US 0.974, India 0.956; singletons 0.964).
- **Pair precision / recall (validation):** 98.9% / 93.5%.
- **Where the validation score is lost** (oracle analysis):
  - removing all false positives would add 0.010
  - recovering true matches that were retrieved but scored too low would add 0.018
  - retrieving the missing 1.8% would add 0.006
- **Common false positives (wrong merges):**
  - **Sibling businesses:** the same or similar name at a nearby house number ("Coastal III LLC" at 519 vs 4510; "MZB Sportive SNC" at no. 10 vs 7).
  - Different businesses sharing an address.
  - These are much more frequent on test than in training.
- **Common false negatives (missed matches):**
  - Native-script names with short addresses, where retrieval misses them.
  - Invented trade names and initials at the same address.
  - Candidates with an empty address.
- **Lesson learned:**
  - "Support" features (similarity to the entity's strongest other candidate) improved validation by +0.004 but **lowered** the leaderboard.
  - On test they pulled in sibling look-alikes and pushed out initials and trade-name matches, as a pair-by-pair comparison of test outputs showed.
  - We removed them and checked every later change against test-output comparisons.

---

## 6. Conclusion
Careful normalisation, state-blocked name+address retrieval and a LightGBM pair model with within-entity competition features give a compact, fast and rule-compliant pipeline. It uses 25.7 candidates per entity and about 1 hour end-to-end on 44 CPU cores, and reaches 0.958 on the public leaderboard. The main remaining error is separating a business from its "siblings" (same name, nearby address), which appear about twice as often in the test data. Our key lesson: in entity resolution, validation must mimic the test distractor density, or feature changes that look good in validation can hurt on test.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (Python 3.12/3.13, dependencies pinned in `requirements.txt`). The full command order is in `README.md`:

preprocess → fill_state → splits → state_neighbors → name_freq → run_candidates (train) → matcher train → run_candidates (test) → predict

Key modules:
- `normalize.py`, `lexicon.py`: normalisation.
- `candidates.py`: retrieval.
- `features.py`: pair and group features.
- `matcher.py`: LightGBM training and threshold.
- `predict.py`: writes both TSVs.
- `metric.py`: official macro F0.5.
- `cross_encoder.py`, `stack.py`: optional cross-encoder stage.
- `models/v6/` holds the trained model.

**Compliance:**
- No external data, APIs, geocoding or registries. All rule lists are hand-written conventions, and all learned maps come from the provided files.
- Licences:
  - LightGBM (MIT)
  - indic-transliteration (MIT)
  - RapidFuzz (MIT)
  - sparse_dot_topn (Apache-2.0)
  - scikit-learn (BSD)
  - optional MiniLM cross-encoder (Apache-2.0, 118M parameters)

### B. Additional Results
- Threshold curve (v6, validation): 0.60 → 0.9669, 0.65 → 0.9676, 0.70 → 0.9671, 0.75 → 0.9669, 0.80 → 0.9657, 0.90 → 0.9598.
- Doubling validation distractors: 0.9667 → 0.9629, with the best threshold moving 0.70 → 0.75.
- Runtime on 44 cores: preprocessing ~4 min; test candidates ~20 min; training ~25 min; test scoring ~15 min.
