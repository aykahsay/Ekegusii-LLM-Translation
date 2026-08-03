# Datasets

## Master Sentence Corpus

`data/master_corpus/master_sentence_corpus.csv` -- 49,277 concepts.

| Column | Description |
|---|---|
| `concept_id` | Unique identifier, stable across all derived files |
| `English`, `Kiswahili`, `Ekegusii` | Parallel text (a row need not have all three -- see coverage below) |
| `source` | Provenance tag (e.g. `Trilingual-Bible`, `ENG-EKE`, `PSA`) |
| `dataset_origin` | Original collection batch |

Verified column-level coverage: `English` 100%, `Ekegusii` ~95.8% (47,191/49,277),
`Kiswahili` ~73.3% (36,110/49,277). Not every row is a complete triplet --
`TrilingualExperiment` (E4) filters to complete triplets only; `FullResourcesExperiment`
(E5) uses everything, including partial rows.

## Master Lexical Corpus

`data/master_corpus/master_lexical_corpus.csv` -- 268 dictionary entries, same
language columns as the sentence corpus plus `lexicon_id`.

**Known limitation (verified, not assumed):** the `English` column is empty for
**all 268 rows** as of this writing -- the lexical corpus is currently
Kiswahili<->Ekegusii only despite the schema having an English column.
`LexicalTaskGenerator` will silently produce zero English-direction lexical
tasks until this is populated. See `configs/datasets/lexical.yaml`.

## Fixed splits

`data/master_corpus/splits/`: `master_train.csv` (39,421 / 80%), `master_val.csv`
(4,928 / 10%), `master_test.csv` (4,928 / 10%). **These files must never be
regenerated** -- every experiment (E0-E8) evaluates against the exact same
`master_test.csv` for results to be comparable. `src.master_corpus.splitter.CorpusSplitter.assert_splits_not_overwritten`
guards against accidental regeneration.

## Derived training subsets

Built from `master_train.csv`, NOT independently re-split (see
`CorpusSplitter.project_to_existing_split` for how new derived subsets should
inherit split membership from the frozen assignment):

| File | Rows | Used by |
|---|---|---|
| `derived_train_eng_eke.csv` | 37,721 | E1 |
| `derived_train_swa_eke.csv` | 27,092 | E2 (also E3, combined with the above) |
| `derived_train_trilingual.csv` | 27,092 | E4 (source data before triplet filtering) |

## Config-driven access

`configs/datasets/{master,bilingual,trilingual,monolingual,lexical}.yaml` describe
each dataset's paths/columns/expected row counts. Load via
`src.utils.config.load_dataset_config(name)`, or resolve straight to a DataFrame
via `src.master_corpus.loader.UnifiedDatasetLoader`.
