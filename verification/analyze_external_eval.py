"""Script to analyze the generated authentic_human_eval.json dataset."""
import os
import sys
import json
import logging
from collections import Counter

sys.path.append(os.path.abspath('.'))
from src.utils.paths import HUMAN_EVAL_PATH

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main() -> None:
    """Analyzes the generated external validation dataset for duplicates and statistics."""
    if not os.path.exists(HUMAN_EVAL_PATH):
        logging.error(f"Dataset not found at {HUMAN_EVAL_PATH}. Run prepare_data.py first.")
        return

    with open(HUMAN_EVAL_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    if total == 0:
        logging.warning("The dataset is empty.")
        return

    identities = 0
    errors = 0

    pair_counter = Counter()

    for item in data:
        src = item['source'].strip()
        tgt = item['expected'].strip()

        pair_counter.update([(src, tgt)])

        if src == tgt:
            identities += 1
        else:
            errors += 1

    duplicates = sum(count - 1 for count in pair_counter.values() if count > 1)

    logging.info("=== External Validation Dataset Analysis ===")
    logging.info(f"Total examples: {total}")
    logging.info(f"Identity translations (source == target): {identities} ({identities/total:.2%})")
    logging.info(f"Error corrections (source != target): {errors} ({errors/total:.2%})")
    logging.info(f"Exact duplicates found: {duplicates} ({(duplicates/total):.2%})\n")

    if duplicates > 0:
        logging.info("--- top 5 most duplicated sentences ---")
        for (src, tgt), count in pair_counter.most_common(5):
            if count > 1:
                logging.info(f"[{count}x] {src}")


if __name__ == "__main__":
    main()
