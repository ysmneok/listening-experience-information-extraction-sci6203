from pathlib import Path

# Project roots

# listening-experience-text-mining/
PROJECT_ROOT = Path(__file__).resolve().parent

# listening-experience-text-mining/pipeline_statistical/
PIPELINE_STATISTICAL_DIR = PROJECT_ROOT / "pipeline_statistical"

# Data & lexicons (statistical pipeline)

DATA_DIR = PIPELINE_STATISTICAL_DIR / "data"
LEXICON_DIR = PIPELINE_STATISTICAL_DIR / "lexicons"

# Files

CORPUS_PATH = DATA_DIR / "all_reviews_merged_final.json"

PERCEPTUAL_LEXICON_PATH = LEXICON_DIR / "perceptual_adjectives_complete.json"
MUSICO_TECH_LEXICON_PATH = LEXICON_DIR / "musico_technical_complete.json"

# Results

RESULTS_DIR = PIPELINE_STATISTICAL_DIR / "outputs"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

