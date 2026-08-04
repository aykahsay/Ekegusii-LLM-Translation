"""
Project-Wide Constants
-----------------------
Single source of truth for filesystem paths, language codes, dataset column
schemas, and split names used throughout the Ekegusii-LLM-Translation package.
Centralizing these avoids each module hardcoding its own copy of paths/columns
that then silently drift out of sync (e.g. MasterCorpusManager's required
columns vs. a preprocessing script's assumed columns).
"""

from pathlib import Path

# --- Filesystem roots -------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
MASTER_CORPUS_DIR: Path = DATA_DIR / "master_corpus"
SPLITS_DIR: Path = MASTER_CORPUS_DIR / "splits"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"

SENTENCE_CORPUS_PATH: Path = MASTER_CORPUS_DIR / "master_sentence_corpus.csv"
LEXICAL_CORPUS_PATH: Path = MASTER_CORPUS_DIR / "master_lexical_corpus.csv"

# --- Languages ---------------------------------------------------------------
LANG_ENGLISH = "English"
LANG_KISWAHILI = "Kiswahili"
LANG_EKEGUSII = "Ekegusii"

SUPPORTED_LANGUAGES = (LANG_ENGLISH, LANG_KISWAHILI, LANG_EKEGUSII)

# ISO-ish short codes used in NLLB-style tags and file suffixes.
LANGUAGE_CODES = {
    LANG_ENGLISH: "eng",
    LANG_KISWAHILI: "swa",
    LANG_EKEGUSII: "eke",
}

# The six bidirectional translation directions the project evaluates.
TRANSLATION_DIRECTIONS = (
    (LANG_ENGLISH, LANG_EKEGUSII),
    (LANG_EKEGUSII, LANG_ENGLISH),
    (LANG_KISWAHILI, LANG_EKEGUSII),
    (LANG_EKEGUSII, LANG_KISWAHILI),
    (LANG_ENGLISH, LANG_KISWAHILI),
    (LANG_KISWAHILI, LANG_ENGLISH),
)

# --- Dataset schemas -----------------------------------------------------------
SENTENCE_CORPUS_COLUMNS = [
    "concept_id",
    LANG_ENGLISH,
    LANG_KISWAHILI,
    LANG_EKEGUSII,
    "source",
    "dataset_origin",
]

LEXICAL_CORPUS_COLUMNS = [
    "lexicon_id",
    LANG_ENGLISH,
    LANG_KISWAHILI,
    LANG_EKEGUSII,
    "source",
]

# --- Splits ---------------------------------------------------------------
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_NAMES = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

MASTER_SPLIT_FILES = {
    SPLIT_TRAIN: "master_train.csv",
    SPLIT_VAL: "master_val.csv",
    SPLIT_TEST: "master_test.csv",
}

# Fixed proportions for the master split. These MUST NEVER change once a
# split has been generated -- every experiment (E0-E8) depends on evaluating
# against the exact same held-out test set for results to be comparable.
MASTER_SPLIT_RATIOS = {
    SPLIT_TRAIN: 0.80,
    SPLIT_VAL: 0.10,
    SPLIT_TEST: 0.10,
}

# --- Experiment identifiers --------------------------------------------------
EXPERIMENT_IDS = (
    "E0_Baseline",
    "E1_English_Ekegusii",
    "E2_Swahili_Ekegusii",
    "E3_Bilingual",
    "E4_Trilingual",
    "E5_Full_Resources",
    "E6_Lexical_Augmentation",
    "E7_Curriculum_Learning",
    "E8_Final_Model",
)

# --- Models -------------------------------------------------------------------
MODEL_QWEN = "qwen"
MODEL_MISTRAL = "mistral"
SUPPORTED_MODELS = (MODEL_QWEN, MODEL_MISTRAL)

MODEL_HF_PATHS = {
    MODEL_QWEN: "Qwen/Qwen2.5-7B-Instruct",
    MODEL_MISTRAL: "mistralai/Mistral-7B-Instruct-v0.3",
}

# --- Reproducibility ----------------------------------------------------------
DEFAULT_SEED = 42

# --- Evaluation -----------------------------------------------------------
EVALUATION_METRICS = ("sacrebleu", "chrf", "comet", "lexical_accuracy", "rare_word_accuracy")
