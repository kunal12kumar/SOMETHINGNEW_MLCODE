# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME]  
**Team Members:** [TEAM MEMBERS]  
**Submission Date:** 27-09-2026

---

## 1. Executive Summary
We normalise names and addresses (including romanising six Indian scripts), generate candidates with state-blocked TF-IDF search over Source 2 and Source 3, and score every candidate pair with a two-stage LightGBM model built on similarity features. Final matches come from a one-owner rule (each Source 2/3 record belongs to at most one Source 1 entity) and a threshold tuned directly for macro F0.5. On 30,000 held-out validation entities the pipeline scores **macro F0.5 = [VALID_F05]**, with candidate recall of 98.2%. It uses only the provided files, no external data or APIs, and only MIT-licensed libraries and models.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the training data:

- **Scale:** 2.21M Source 1 entities in train, about 5M records in each of Source 2 and 3; 1.73M Source 1 entities in test. All-pairs comparison is impossible, so candidate generation is essential.
- **Match structure:** 5.6% of Source 1 entities are singletons, the mean is 3.5 matches (range 0–11), and **no Source 2/3 record matches more than one Source 1 entity** (all 7.64M labelled IDs are unique). This gives the one-owner rule.
- **Name noise:** legal-form variants (Pvt/Private, Ltd/Limited, LLC/L.L.C.), digit-for-letter typos (H0rizon, 5umit), bracketed tags ([LP], (Center)), leading junk ("-- "), "dba / a/k/a / t/a" aliases, web domains (wilfordhancock.com), and **invented trade names** that share no words with the real name ("Ember Properties Inc" ↔ "Nylajax"). These can only be matched by address.
- **Scripts:** 23% of India Source 2 names are in Devanagari, Gujarati, Bengali, Odia, Tamil, Telugu, Kannada or Malayalam script; state names also appear in native script inside addresses.
- **Address noise:** abbreviations (St, Rd, Ave, R, Av), number labels (H.No, P.NO., N°, #), leading zeros, component reordering, duplicated tokens ("CITY CITY"), 3–6% empty addresses in Source 2/3.
- **State labels:** codes vs names (TX vs Texas, WB vs West Bengal). 99.4% of labelled pairs agree on state; the main exception is Telangana ↔ Andhra Pradesh.
- **France (test only):** Source 1 uses regions while 65–68% of Source 2/3 records use departments, so no state was detected for them before our city-based fill.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (with a one-owner assignment step)  
**Core Innovation:** State-blocked TF-IDF retrieval that combines name and address in one vector, a data-driven state repair (city → state map and learned "neighbour" states), and a two-stage matcher whose second stage sees how a candidate compares with the other candidates of the same entity. The threshold is tuned on the official macro F0.5 including singletons.

Generalisation to the unseen country: every feature is a similarity or an agree/conflict flag, never a country one-hot or a raw word. `country` is used only as an open-set grouping key, and all learned maps (city → state, neighbour states) come from the provided files of the same split.

---

## 3. Candidate Generation (Blocking)

- **Normalisation first:** see Appendix A (`normalize.py`). Raw text is kept alongside a normalised form and a "core" name (legal forms and honorifics removed), so distinct businesses are not collapsed.
- **Search paths** (per country label, per source, top-K kept separately for S2 and S3):
  - `addr`: word TF-IDF on the normalised address, K = 10. It finds matches whose names differ completely.
  - `both`: character 3-gram TF-IDF on the name (weight 0.6) concatenated with word TF-IDF on the address (weight 0.4), K = 20. A common name ranks high only when the address also agrees.
  - A name-only path was tested and dropped: 17× slower, with little extra recall.
- **Blocking key: state.** A query searches pool records in its own state, its learned neighbour states, and records with no detected state. This cut search time about 20× (0.8 ms/query instead of 15 ms) and *raised* recall because out-of-state look-alikes no longer fill the top-K.
- **State repair:**
  - A city → state map is learned from Source 1 of the same split (≥3 occurrences, ≥90% purity) and fills missing states in Source 2/3. For France this lowered no-state records from 68% to 7%.
  - Neighbour states come from training labels (state pairs with ≥0.5% of a state's matches), e.g. Telangana ↔ Andhra Pradesh.
- **Candidate pairs generated:** 94,296,471 for the test set (54.4 per Source 1 entity), from about 10M × 1.7M possible pairs.
- **How true matches were kept:** recall@K was measured on held-out validation entities after every change, with a target of ≥97%. Final candidate recall on validation: **98.2%**; for comparison, name + address union without blocking was 97.2%.

| Setting (validation, 30k S1) | Union recall | Candidates / S1 |
|---|---|---|
| name + addr TF-IDF, K=20 | 96.9% | 82 |
| + combined name+address path | 97.2% | 45 |
| + state blocking (name path dropped) | 97.4% | 38 |
| + city → state fill + neighbour states | **98.2%** | 55 |

---

## 4. Matching Model

**Features used** (about 40, all computed with RapidFuzz in C++):
- **Name features:** Levenshtein ratio, token-sort, token-set, partial ratio and Jaro-Winkler on core names; ratio and token-set on full normalised names; no-space ratio (for domains); best score against "dba" aliases; exact core match; initials vs glued name ("M R & X Sun" ↔ mr.com); legal-form agreement; name lengths; transliteration flag; how common each core name is in the pool (log frequency).
- **Address features:** token-set, token-sort and partial token-set ratios; house-number set overlap and near-equality (dropped/extra digit); largest-number equality and edit ratio; city equality and ratio; state agree/conflict/missing; missing-address flags.
- **Other:**
  - Retrieval scores and ranks from both paths, and the number of paths that found the pair; candidate source (S2/S3).
  - Stage-2 group features computed from the stage-1 score within each Source 1 entity: rank, gap to the best candidate, gap to the best candidate from the same source, number of candidates, and number scoring above 0.5.

**Model type:** two-stage LightGBM (MIT licence):
- **Stage 1** is trained with 3-fold out-of-fold predictions, grouped by Source 1 entity.
- **Stage 2** takes the pair features, the stage-1 score and the group features.
- **Settings:** 800 rounds, 63 leaves, learning rate 0.05, bagging and feature subsampling 0.8, minimum 200 rows per leaf.
- **Training data:** 150,000 training-fold Source 1 entities, 8.3M candidate pairs (6.2% positive). Negatives are the realistic hard negatives produced by our own candidate generation.

**Threshold selection method:**
1. Apply the one-owner rule: each Source 2/3 record is kept only for its highest-scoring Source 1 entity.
2. Grid-search the threshold from 0.20 to 0.95 on 30,000 validation entities, maximising the official macro F0.5 with singletons included.
3. The curve is flat between 0.60 and 0.75, so we chose [THRESHOLD] from that stable region.

**Validation design:** entity-grouped split (10% of Source 1 by a hash of the ID), so an entity and all its matches are entirely in train or entirely in validation.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation):** [VALID_F05] (US [VALID_US], India [VALID_IN]; singletons scored [VALID_SINGLE])
- **Pair precision / recall:** [PAIR_P] / [PAIR_R]
- **Common false positives (wrong merges):**
  - Near-identical businesses at the same address that differ only in legal form or one word ("Foundation Tech Private Limited" vs "Foundation Tech Limited").
  - Different businesses sharing a building ("Char Games Inc." vs "Games Patterson Inc.", same street number).
  - Adjacent house numbers on the same street.
- **Common false negatives (missed matches):**
  - Candidates with an empty address, where only the name is available.
  - House numbers with a dropped or extra digit (4120 vs 412), which we addressed with near-equal number features.
  - Invented trade names whose address is partial.
  - Transliterated names with very short addresses (1.8% of true pairs never reach the candidate set).

---

## 6. Conclusion
The problem is mostly about finding every true match: only 5.6% of Source 1 entities have no match, so recall at the candidate stage was the first bottleneck. Careful normalisation plus state-blocked name+address retrieval gave 98.2% candidate recall at about 55 candidates per entity. A similarity-feature LightGBM with within-entity competition features, the one-owner rule and an F0.5-tuned threshold then keeps pair precision near 99%. The main lesson: blocking on a repaired, data-driven key (state) improved both speed and recall. With more time, a fine-tuned multilingual bi-encoder (multilingual-e5, MIT) as an extra retrieval path would target the remaining transliterated-name misses.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`, with Python 3.13 and dependencies pinned in `requirements.txt`:

| File | Purpose |
|---|---|
| `src/ber/lexicon.py` | Hand-written formatting conventions: legal forms, address abbreviations, postal state codes |
| `src/ber/normalize.py` | Name/address normalisation, Indic transliteration (`indic-transliteration`, MIT) |
| `src/ber/preprocess.py` | Streams the six TSVs to normalised parquet |
| `src/ber/fill_state.py` | City → state map learned from Source 1; fills missing states |
| `src/ber/splits.py` | Entity-grouped validation split |
| `src/ber/state_neighbors.py` | States often swapped between sources, from training labels |
| `src/ber/name_freq.py` | Core-name frequency per split and country |
| `src/ber/candidates.py`, `run_candidates.py` | State-blocked TF-IDF candidate generation |
| `src/ber/features.py` | Pair and group features |
| `src/ber/matcher.py` | Two-stage LightGBM training, threshold tuning |
| `src/ber/predict.py` | Test scoring, writes `matching_results.tsv` and `candidate_pairs.tsv` |
| `src/ber/metric.py` | Official macro F0.5 |
| `src/ber/eval_candidates.py`, `analyze_errors.py`, `audit_normalize.py` | Recall, error and normalisation checks |
| `models/` | Trained LightGBM models and tuned threshold |
| `notebooks/colab_pipeline.ipynb` | Same pipeline on Google Colab |

Entry points and the exact command order are in `README.md`: preprocess → fill_state → splits → state_neighbors → name_freq → run_candidates (train) → matcher train → run_candidates (test) → predict.

**Compliance:**
- No external data, APIs, geocoding or lookups. All lists are written in our own code, and all learned maps come from the provided files.
- Models: LightGBM (MIT), with no pretrained neural model in the final pipeline.
- Libraries: pandas, scikit-learn, RapidFuzz, sparse_dot_topn, indic-transliteration and Unidecode are all under permissive licences.

### B. Additional Results
- Normalisation audit on 73,531 labelled pairs, mean similarity raw → normalised: name 78.2 → 87.5 (India S2: 65.8 → 86.0 from transliteration), address 85.2 → 92.9.
- Threshold curve (validation macro F0.5): 0.50 → 0.955, 0.60 → 0.958, 0.70 → 0.959, 0.80 → 0.957, 0.90 → 0.952.
- Runtime on 8 cores: preprocessing ~10 min; test candidate generation 62 min; training ~10 min; test scoring ~30–45 min.
