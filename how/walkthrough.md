# Pipeline Fix & Master Corpus Rebuild: Walkthrough

This document summarizes the changes applied to the data pipeline to correct the 1-to-N duplicate misalignments and fix the Mojibake encoding artifacts.

## Changes Made

### 1. Enforced 1-to-1 Source Deduplication
In `src/create_clean_final_master_corpus.py`, the deduplication logic was updated to strictly evaluate the source sentence (`English`, or `Kiswahili` if `English` is empty). Previously, it used `dedup_key_swa = Kiswahili + "|||" + Ekegusii`. By dropping the `Ekegusii` text from the deduplication key, the script no longer treats N different target translations of the same source sentence as unique records. This prevents the previous issue where 406 target variations mapped to a single Kiswahili entry.

### 2. Applied Global Mojibake Cleaning
We modified the dataframe processing logic to explicitly route all text through the `clean_str()` function instead of using raw `str.strip()`. This ensures that all CP1252 parsing artifacts (`Ã©`, `â€œ`, `?`) natively introduced by the `latin-1` pandas fallback (especially from `Bible_engswguz_cleaned_prof.csv`) are thoroughly sanitized across the entire dataset.

## Execution and Validation

The updated script was successfully executed, generating a new `clean_final_dataset_corpus.csv`. The rebuild processed 257,340 initial records and consolidated them down to **72,155 completely unique, 1-to-1 aligned concepts**.

### Validation Results
A secondary script was executed to validate the integrity of the output file:
- **English Source Duplicates:** 0
- **Kiswahili Source Duplicates (where English is null):** 0
- **Mojibake Instances:** 0

The dataset is now strictly aligned, uniquely mapped, and free of encoding corruption, fully ready for downstream tasks.
