"""Script to parse and convert the PoliMorf dictionary into a fast Parquet cache."""
import os
import gzip
import logging
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utils.paths import POLIMORF_TAB_GZ_PATH, POLIMORF_PARQUET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_polimorf(filepath: str, output_file: str) -> None:
    """Parses the PoliMorf tab file into a compressed Parquet file.

    Args:
        filepath: Path to the `.tab.gz` PoliMorf dictionary.
        output_file: Output path for the `.parquet` file.
    """
    if not os.path.exists(filepath) or not filepath.endswith(".gz"):
        logging.error(f"Could not find '{filepath}' or it is not a .gz file.")
        raise FileNotFoundError(f"Input file not found at '{filepath}'.")

    logging.info(f"Counting lines in {filepath} for progress bar...")
    with gzip.open(filepath, "rb") as f:
        total_lines = sum(1 for _ in f)

    batch = []
    logging.info(f"Parsing PoliMorf from {filepath}...")
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Parsing PoliMorf"):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            form, lemma, tag = parts[0], parts[1], parts[2]
            tag_parts = tag.split(":")
            pos = tag_parts[0]

            if pos not in ("subst", "adj", "fin", "num", "praet"):
                continue

            feature_key = ":".join(tag_parts[1:])
            batch.append((lemma, pos, feature_key, form))

    logging.info("Creating dataframe and dropping duplicates...")
    df = pd.DataFrame(batch, columns=["lemma", "pos", "features", "form"])
    df.drop_duplicates(subset=["lemma", "pos", "features"], inplace=True)
    df.to_parquet(output_file)
    logging.info("Successfully created Parquet file.")


def main() -> None:
    """Main execution block of the PoliMorf parser."""
    if not os.path.exists(POLIMORF_TAB_GZ_PATH):
        logging.error(f"Input file not found at '{POLIMORF_TAB_GZ_PATH}'.")
        raise FileNotFoundError(f"Input file not found at '{POLIMORF_TAB_GZ_PATH}'.")

    parse_polimorf(POLIMORF_TAB_GZ_PATH, POLIMORF_PARQUET_PATH)


if __name__ == "__main__":
    main()
