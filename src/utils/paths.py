"""Module defining absolute paths to various data directories and resource files."""
import os

BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
DOWNLOADED_DIR: str = os.path.join(DATA_DIR, "downloaded")
SCRAPED_DIR: str = os.path.join(DATA_DIR, "scraped")
GENERATED_DIR: str = os.path.join(DATA_DIR, "generated")
SYNTHESIZED_DIR: str = os.path.join(DATA_DIR, "synthesized")
EVAL_DIR: str = os.path.join(DATA_DIR, "eval")

CURATED_JSON_PATH: str = os.path.join(SCRAPED_DIR, "raw_curated_false_friends.json")
POLIMORF_TAB_GZ_PATH: str = os.path.join(DOWNLOADED_DIR, "PoliMorf-0.6.7.tab.gz")
LLM_DUMP_PATH: str = os.path.join(GENERATED_DIR, "llm_raw_output_dump.txt")

POLIMORF_PARQUET_PATH: str = os.path.join(GENERATED_DIR, "polimorf.parquet")
UNIFIED_FF_PATH: str = os.path.join(SCRAPED_DIR, "unified_false_friends.json")
GENDER_MISMATCH_PATH: str = os.path.join(GENERATED_DIR, "gender_mismatches.json")
PREP_MISMATCH_PATH: str = os.path.join(GENERATED_DIR, "preposition_mismatches_llm.json")
HUMAN_EVAL_PATH: str = os.path.join(EVAL_DIR, "authentic_human_eval.json")

SYNTHETIC_TRAIN_PATH: str = os.path.join(SYNTHESIZED_DIR, "synthetic_train.parquet")
SYNTHETIC_VAL_PATH: str = os.path.join(SYNTHESIZED_DIR, "synthetic_val.parquet")
SYNTHETIC_TEST_PATH: str = os.path.join(SYNTHESIZED_DIR, "synthetic_test.parquet")

WIKIPEDIA_TRAIN_PATH: str = os.path.join(DOWNLOADED_DIR, "Wikipedia-pl-train.parquet")
WIKIPEDIA_VAL_PATH: str = os.path.join(DOWNLOADED_DIR, "Wikipedia-pl-val.parquet")
WIKIPEDIA_TEST_PATH: str = os.path.join(DOWNLOADED_DIR, "Wikipedia-pl-test.parquet")
