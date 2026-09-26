# Business Entity Resolution: reproducible pipeline

Blocking → learned candidate pruning → LightGBM matcher → exclusive, F0.5-aware decision.
Everything is learned from the provided training files only. There are no external
lookups, APIs or pretrained language models; the only models are two LightGBM
gradient-boosted tree ensembles (MIT licence, a few MB each).

## Environment

- Python 3.11, `pip install -r requirements.txt`
- Tested on 4 CPU cores and 15 GB RAM. Peak memory is about 11 GB during training.
- Wall-clock: training takes about 1.5 h and test inference about 1 h.

## Data layout

```
dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## Run end-to-end

From `code/business_entity_resolution/src/`:

```bash
# 1. learn transliteration maps, normalise, block, train stage-2 + stage-3, tune decision
python run.py train   --data <path>/dataset --work ../work

# 2. normalise + block test, prune candidates, match, write outputs
python run.py predict --data <path>/dataset --work ../work --out <path>/output

# 3. validate
python3 <path>/utils/validate_submission.py --matching <path>/output/matching_results.tsv \
        --candidate <path>/output/candidate_pairs.tsv --test-dir <path>/dataset/test
```

`predict` writes `output/candidate_pairs.tsv` (the pruned candidate set, the exact pairs
the matcher scores) and `output/matching_results.tsv`. All intermediate artefacts
(normalised parquet caches, `translit.json`, `stage2.txt`, `stage3.txt`, `config.json`)
go to `--work`.

## Source layout

| file | role |
|---|---|
| `src/run.py` | CLI: `train` / `predict` |
| `src/ber/normalize.py` | name/address normalisation: accent folding, leetspeak repair (`pub1ic` → `public`), DBA/AKA/URL splitting, legal-suffix handling, abbreviation expansion, US state canonicalisation, phonetic skeleton |
| `src/ber/translit.py` | learns native-script → Latin token dictionary (Devanagari, Tamil, Telugu …) and state/abbreviation canonicalisation from train ground truth |
| `src/ber/prep.py`, `src/ber/cache.py` | parallel normalisation into parquet tables |
| `src/ber/blocking.py` | country-sharded inverted-index key blocking with stop-key pruning; sparse IDF-weighted scoring |
| `src/ber/features.py` | vectorised pair features (rapidfuzz `cpdist`) |
| `src/ber/pipeline.py` | stage 1/2/3 orchestration, context features, exclusive assignment, expected-F0.5 set selection, metric |
