"""Orchestrator script to fetch, generate, and synthesize all required data for the pipeline."""
import os
import re
import sys
import logging
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.append(os.path.abspath("."))
from src.utils.paths import (
    DATA_DIR,
    DOWNLOADED_DIR,
    SCRAPED_DIR,
    GENERATED_DIR,
    SYNTHESIZED_DIR,
    EVAL_DIR,
    POLIMORF_TAB_GZ_PATH,
    POLIMORF_PARQUET_PATH,
    UNIFIED_FF_PATH,
    GENDER_MISMATCH_PATH,
    HUMAN_EVAL_PATH,
    PREP_MISMATCH_PATH,
    SYNTHETIC_TRAIN_PATH,
    SYNTHETIC_VAL_PATH,
    SYNTHETIC_TEST_PATH,
    WIKIPEDIA_TRAIN_PATH,
)

from src.data.synthesis.process_polimorf import main as process_polimorf_main
from src.data.acquisition.wiktionary_scrapper import main as wiktionary_scrapper_main
from src.data.synthesis.synthesize_data import main as synthesize_data_main
from src.data.generation.generate_gender_mismatches import (
    main as generate_gender_mismatches_main,
)
from src.data.generation.generate_human_eval_llm import (
    main as generate_human_eval_main,
)
from src.data.generation.generate_preposition_mismatches_llm import (
    main as generate_preposition_mismatches_main,
)
from src.config import config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_sentences(texts: list[str]) -> list[str]:
    """Cleans and segments raw article texts into sentences within a valid length.

    Args:
        texts: List of article strings.

    Returns:
        List of cleaned, parsed sentences.
    """
    sents: list[str] = []
    for t in texts:
        # split by punctuation and newlines to avoid joining text paragraphs
        for s in re.split(r"[.!?\n]", t):
            s_clean = s.strip()
            words = s_clean.split()
            # a sentence should have between 5 and 30 words, and no formula junk
            if 5 <= len(words) <= 30:
                if any(
                    char in s_clean
                    for char in [
                        "=",
                        "{",
                        "}",
                        "\\",
                        "*",
                        "[",
                        "]",
                        "|",
                        "<",
                        ">",
                        "+",
                        "/",
                    ]
                ):
                    continue
                sents.append(s_clean + ".")
    return sents


def main() -> None:
    """Main orchestrator function that runs data downloading, processing, and synthesis."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOWNLOADED_DIR, exist_ok=True)
    os.makedirs(SCRAPED_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(SYNTHESIZED_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    polimorf_url = "https://zil.ipipan.waw.pl/PoliMorf?action=AttachFile&do=get&target=PoliMorf-0.6.7.tab.gz"

    if not os.path.exists(POLIMORF_PARQUET_PATH) and not os.path.exists(
        POLIMORF_TAB_GZ_PATH
    ):
        logging.info(f"Downloading PoliMorf from {polimorf_url}...")
        urllib.request.urlretrieve(polimorf_url, POLIMORF_TAB_GZ_PATH)

    if not os.path.exists(POLIMORF_PARQUET_PATH):
        logging.info("Processing PoliMorf into a Parquet file...")
        process_polimorf_main()
    else:
        logging.info("PoliMorf Parquet file already exists.")

    if not os.path.exists(WIKIPEDIA_TRAIN_PATH):
        logging.info("Streaming Native Polish Wikipedia from HuggingFace...")
        wiki_ds = load_dataset(
            "wikimedia/wikipedia", "20231101.pl", split="train", streaming=True
        )
        articles: list[str] = []
        for i, row in enumerate(wiki_ds):
            if i >= 4000:
                break
            articles.append(row["text"])

        pl_full = extract_sentences(articles)
        pd.DataFrame({"text": pl_full}).to_parquet(WIKIPEDIA_TRAIN_PATH)
        logging.info(f"Saved {len(pl_full)} base sentences.")
    else:
        logging.info("Wikipedia Native Corpus already exists.")

    if not os.path.exists(UNIFIED_FF_PATH):
        logging.info("Downloading false friends data...")
        wiktionary_scrapper_main()
    else:
        logging.info("False friends data is already available.")

    if not os.path.exists(GENDER_MISMATCH_PATH):
        logging.info("Generating gender mismatches...")
        generate_gender_mismatches_main()
        if not os.path.exists(GENDER_MISMATCH_PATH):
            raise RuntimeError(f"Pipeline failure: Failed to generate {GENDER_MISMATCH_PATH}")
    else:
        logging.info("Gender mismatches already generated.")

    if not os.path.exists(PREP_MISMATCH_PATH):
        logging.info("Generating preposition mismatches...")
        generate_preposition_mismatches_main()
        if not os.path.exists(PREP_MISMATCH_PATH):
            raise RuntimeError(f"Pipeline failure: Failed to generate {PREP_MISMATCH_PATH}")
    else:
        logging.info("Preposition mismatches already generated.")

    if not os.path.exists(HUMAN_EVAL_PATH):
        logging.info("Generating external validation dataset...")
        generate_human_eval_main()
        if not os.path.exists(HUMAN_EVAL_PATH):
            raise RuntimeError(f"Pipeline failure: Failed to generate {HUMAN_EVAL_PATH}")
    else:
        logging.info("External validation dataset already exists.")

    synthetic_full_path = os.path.join(SYNTHESIZED_DIR, "synthetic_full.parquet")
    if not os.path.exists(synthetic_full_path):
        logging.info("Generating full synthetic dataset...")
        synthesize_data_main(
            WIKIPEDIA_TRAIN_PATH,
            synthetic_full_path,
            max_pairs=config.max_pairs,
            max_injections_per_word=config.max_injections_per_word,
        )

    if (
        not os.path.exists(SYNTHETIC_TRAIN_PATH)
        or not os.path.exists(SYNTHETIC_VAL_PATH)
        or not os.path.exists(SYNTHETIC_TEST_PATH)
    ):
        logging.info(
            "Performing Lemma-Level Split on synthetic data to prevent data leakage..."
        )
        df_synth = pd.read_parquet(synthetic_full_path)
        unique_lemmas = df_synth["error_lemma"].dropna().unique()

        train_lemmas, temp_lemmas = train_test_split(
            unique_lemmas, test_size=0.2, random_state=42
        )
        val_lemmas, test_lemmas = train_test_split(
            temp_lemmas, test_size=0.5, random_state=42
        )

        train_df = df_synth[df_synth["error_lemma"].isin(train_lemmas)].reset_index(
            drop=True
        )
        val_df = df_synth[df_synth["error_lemma"].isin(val_lemmas)].reset_index(
            drop=True
        )
        test_df = df_synth[df_synth["error_lemma"].isin(test_lemmas)].reset_index(
            drop=True
        )

        train_df.to_parquet(SYNTHETIC_TRAIN_PATH)
        val_df.to_parquet(SYNTHETIC_VAL_PATH)
        test_df.to_parquet(SYNTHETIC_TEST_PATH)
        logging.info(
            f"Saved Lemma-Split sets: Train ({len(train_df)} rows), Val ({len(val_df)} rows), Test ({len(test_df)} rows)."
        )

    logging.info("Data preparation complete! You can now run main.ipynb.")


if __name__ == "__main__":
    main()
