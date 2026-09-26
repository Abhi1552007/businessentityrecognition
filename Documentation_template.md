# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
The solution is a three-stage cascade. Stage 1 is a country-sharded, inverted-index **key blocking** step with stop-key pruning and sparse IDF scoring; it never compares all pairs. Stage 2 is a learned **LightGBM candidate pruner** that cuts the blocked set to **about 4.5 candidates per Source 1 entity**; the ground truth averages 3.5 true matches per entity. Stage 3 is a **LightGBM matcher** with "competition" context features, followed by a one-owner-per-record exclusivity rule and a per-entity **expected-F0.5 set selection**. Held-out validation macro F0.5 is **0.970**, using only the provided training data and no external lookups.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on 2.2M train Source 1 entities and 10.3M Source 2/3 records:
- Each Source 2/3 record matches **at most one** Source 1 entity. 26% of Source 2/3 records are decoys that match nothing, and 5.6% of Source 1 entities are singletons. The average is about 3.5 matches per entity (up to 9).
- Name noise includes typos, **leetspeak** (`Pub1ic`, `6urgaon`, `A1l`), accents (`Énterprises`), legal-suffix churn (`Pvt`/`Private`/`Ltd`/`[Limited]`), filler suffixes (`Center`, `Services`), and glued names or domains (`maurewilliamscolombier.com`). There are DBA/AKA/`|` patterns in which one part is an invented brand (`Drexveo dba Signature Valley …`). About 10% of Indian names in Source 2/3 are in **native scripts** (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Gurmukhi, Odia).
- Address noise includes reordered components, dropped components, abbreviations (`Rd`, `St`, `Saint` used for `St`), state names in native scripts or as abbreviations (`TN`, `UP`, `Texas` vs `TX`), padded house numbers (`0305`), unit suffixes (`1056c`), and null markers (`<NULL>`, `N/A`). About 3.4% of addresses are empty.
- The test set adds **France** (1.4M Source 2/3 records), which never appears in training. Therefore no feature uses the country value itself. Country is only a blocking-shard label, and every text rule is language-agnostic, plus a handful of French street abbreviations (`R.`→rue, `BD`→boulevard, `ALL.`→allée) and legal forms (SARL/SAS/EURL/SCI).

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned pruning + gradient-boosted classifier + constrained, F0.5-aware decoding.
**Core Innovation:**
1. Scalable key blocking with **name×address combination keys**. Common names such as "Global" or "Titan" are useless as blocks on their own, but `global × 4249` or `global × knox` stays selective.
2. A **learned pruner** whose output is the reported candidate set.
3. **Competition features plus exclusivity**, exploiting the fact that each Source 2/3 record belongs to at most one Source 1 entity.
4. **Expected-F0.5 subset selection** per entity, which handles singletons explicitly.

---

## 3. Candidate Generation (Blocking)

**Normalisation** (`normalize.py`, `translit.py`):
- Unicode accent folding.
- **Native-script transliteration** from a token dictionary *learned from train ground truth*: Latin Source 1 names are aligned positionally with native-script Source 2/3 names that have the same token count, e.g. `प्राइवेट→private`, `ராஜ→raj`. There are 1,343 learned tokens, with `unidecode` as a fallback.
- Leetspeak repair on mixed alphanumeric tokens and gluing of single-letter runs (`l l c`→`llc`).
- Splitting DBA/AKA/`|`/URL variants into alternative names.
- Abbreviation expansion, removal of legal and filler words to get a *core name*, address-token canonicalisation, US state name→USPS code, and learned state/abbreviation canonicalisation (`tn`→`tamil nadu`, `தமிழநாடு`→`tamil nadu`).

**Blocking keys** (`blocking.py`). All keys are prefixed by country and hashed to 64 bits:
- Core-name tokens, phonetic **skeletons** (typo/transliteration robust), the glued name and its 6-char prefix, **sorted token pairs** (word order), acronym, and a sorted-letter anagram key.
- 5-grams of long glued names and 5-char heads of tokens (`askykonnect` ↔ `sky konnect`).
- Address keys: house number + street skeleton, postal code, and street/locality bigrams.
- **Name×address combos**: first name token × each house number, and first name token × each locality skeleton.

**Scaling.**
- The index is sharded by country, and keys whose posting list is longer than **500** records are dropped (stop-key pruning). Work per Source 1 record is therefore bounded by `#keys × 500`, independent of corpus size, and the index shards further by key hash.
- Scoring is an IDF-weighted count of shared keys, computed as a chunked sparse product `L @ Rᵀ`. The top 40 per entity are kept.
- On train this gives 87.1M pairs (39.5 per entity) with **95.97% pair recall**, in about 12 minutes on 4 cores for 2.2M × 10.3M records.

**Stage-2 learned pruning:**
- A LightGBM model uses 20 cheap features: rapidfuzz ratio, token-set, partial and glued-name ratios on the best DBA-variant pair, address token-set, house-number Jaccard, lengths, the blocking score, and its rank/gap within the entity.
- It is trained on 250k entities, with out-of-fold predictions for those entities.
- Pairs with `p2 ≥ τ2` are kept, at most 12 per entity. `τ2` is the value that drops only 0.3% of the positives blocking found.

| stage | pairs / S1 entity | pair recall (train, full 2.2M) |
|---|---|---|
| blocking (top-40) | 39.5 | 95.97% |
| **stage-2 candidates (= candidate_pairs.tsv)** | **4.46** | **95.69%** |

- **Candidate pairs generated (train):** 9.85M for 2.2M entities.
- **How true matches were kept:** multiple redundant key families (exact, phonetic, prefix, glued, combination, address-only), transliteration before key extraction, and a pruning threshold chosen explicitly for 99.7% retention.

---

## 4. Matching Model

**Features used** (about 70, `features.py`, computed with multi-threaded rapidfuzz `cpdist`):
- **Name:** ratio, token-sort, token-set, partial ratio, Jaro-Winkler, Levenshtein distance, glued-name ratio and partial ratio, skeleton (phonetic) token-sort, full-name ratio and token-set (with legal words), best-variant token-set across DBA parts, token Jaccard, intersection, token counts, lengths, first-token equality, exact-core equality, and a multi-variant flag.
- **Address:** token-set, token-sort, partial token-set, alpha-token Jaccard and coverage, number Jaccard, intersection and conflict, first-number equality, address-component Jaccard and intersection, lengths, and a Source-3 flag.
- **Blocking:** key score, rank, gap, relative score, and candidate count.
- **Competition context:** the stage-2 probability, its rank, gap, sum and count within the Source 1 entity; its rank, gap and count within the Source 2/3 record, i.e. how many Source 1 entities compete for it; and the **margin to the best competing Source 1 entity** (`j_margin`, the single most important feature).

**Model type:** LightGBM binary classifier (127 leaves, learning rate 0.08, early stopping; 1,487 trees). It is trained on the candidates of 600k entities, with context computed over all 2.2M entities so that it matches the test-time distribution.

**Threshold selection method:**
1. **Exclusivity:** each Source 2/3 record keeps only its highest-probability Source 1 entity.
2. For each Source 1 entity, candidates are sorted by probability and the top-m set maximises the plug-in expected F0.5, `1.25·Σp_top / (0.25·Σp_all + m)`. That value is compared with `Π(1−p)`, the chance that the entity is a singleton, for which an empty prediction scores 1.0.
3. Candidates below a floor τ are never added. The mode ("threshold" vs "expected-F") and τ are tuned on 200k held-out entities that were never used for fitting.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), 200k held-out train entities:** **0.9702**. The ceiling with a perfect matcher on the same candidates is 0.9856.
- **Common false positives:** very generic short names at the same street, e.g. two "Global …" businesses, and chains or franchises with near-identical names in the same city.
- **Common false negatives:**
  - Invented DBA/brand names with only a partial address (`Zetairi` at `unit 215 DLF QE Gurgaon`).
  - Heavily corrupted names with an empty address (`global efstaet`).
  - Glued names that include a legal word (`privatecommotradenirvana`).

  The first two are the main source of the 4% blocking-recall gap. Adding them would cost precision, which F0.5 weights twice as heavily.

---

## 6. Conclusion
Careful, learned normalisation (transliteration dictionaries mined from the ground truth), redundant scalable blocking keys, and a two-model cascade give about 4.5 candidates per entity and a validation macro F0.5 of 0.970. The biggest lessons: modelling the one-owner-per-record constraint explicitly (competition features plus exclusive assignment) was the strongest single signal, and choosing each entity's match set by expected F0.5 handles singletons naturally.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/run.py` — entry point: `python run.py train …` then `python run.py predict …`
- `src/ber/normalize.py`, `translit.py`, `prep.py`, `cache.py` — normalisation
- `src/ber/blocking.py` — sharded key index
- `src/ber/features.py` — pair features
- `src/ber/pipeline.py` — stages, context, decision, metric

The exact commands are in `README.md`, with pinned versions in `requirements.txt`. The only models are two LightGBM ensembles (MIT licence, a few MB). No pretrained language models, external data or APIs are used.

### B. Additional Results
- Blocking recall vs top-k (40k-entity sample): k=5: 83.9%, k=10: 92.5%, k=20: 94.9%, k=40: 96.0%, k=100: 97.0%.
- Decision-rule sweep (validation): expected-F0.5 with τ=0.65 → 0.97021; plain threshold 0.70 → 0.97002.
