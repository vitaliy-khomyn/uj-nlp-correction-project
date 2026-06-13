"""Generates static rules for misgendering Polish nouns by RU/UA L1 speakers."""
import json
import os
import logging
import pandas as pd
import sys
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utils.paths import POLIMORF_PARQUET_PATH, GENDER_MISMATCH_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    """Generates a database of Polish nouns misgendered by RU/UA L1 speakers."""
    if not os.path.exists(POLIMORF_PARQUET_PATH):
        logging.error(f"PoliMorf Parquet not found at {POLIMORF_PARQUET_PATH}")
        return

    df = pd.read_parquet(POLIMORF_PARQUET_PATH)
    df_subst = df[df["pos"] == "subst"]
    rows = df_subst[["lemma", "features", "form"]].values.tolist()

    # build sets for O(1) collision detection
    valid_forms = {row[2].lower() for row in rows}

    gender_mismatches: Dict[str, Dict[str, str]] = {}

    for lemma, features, _ in rows:
        lemma = lemma.lower()
        if not lemma.isalpha() or len(lemma) < 5:
            continue

        wrong_form = None
        wrong_gender = None

        # pattern 1: Greek/Latin origin
        if lemma.endswith("um") and ":n" in features:
            wrong_gender, wrong_form = "m3", lemma

        # pattern 2a: orthographic calque
        elif (
            lemma.endswith("mat")
            and not lemma.endswith("omat")
            and ":m3" in features
        ):
            wrong_gender, wrong_form = "f", lemma[:-1]

        # pattern 2b: orthographic calque
        elif (lemma.endswith("em") or lemma.endswith("am")) and ":m3" in features:
            wrong_gender, wrong_form = "f", lemma + "a"

        # pattern 3: orthographic calque
        elif lemma.endswith(("oza", "yza", "eza")) and ":f" in features:
            wrong_gender, wrong_form = "m3", lemma[:-1]

        if wrong_form and wrong_gender:
            if wrong_form != lemma and wrong_form in valid_forms:
                continue

            gender_mismatches[lemma] = {
                "wrong_gender": wrong_gender,
                "wrong_form": wrong_form,
            }

    manual_overrides: Dict[str, Dict[str, str]] = {
        "cytat": {"wrong_gender": "f", "wrong_form": "cytata"},
        "metoda": {"wrong_gender": "m3", "wrong_form": "metod"},
        "ból": {"wrong_gender": "f", "wrong_form": "ból"},
        "model": {"wrong_gender": "f", "wrong_form": "model"},
        "cel": {"wrong_gender": "f", "wrong_form": "cel"},
        "detal": {"wrong_gender": "f", "wrong_form": "detal"},
        "medal": {"wrong_gender": "f", "wrong_form": "medal"},
        # the "-pis" group
        "podpis": {"wrong_gender": "f", "wrong_form": "podpis"},  # ru: подпись
        "napis": {"wrong_gender": "f", "wrong_form": "napis"},
        "zapis": {"wrong_gender": "f", "wrong_form": "zapis"},
        "opis": {"wrong_gender": "f", "wrong_form": "opis"},
        # the "-eń" group
        "stopień": {"wrong_gender": "f", "wrong_form": "stopień"},  # ru: степень
        "cień": {"wrong_gender": "f", "wrong_form": "cień"},
        # short cognate exceptions
        "klasa": {"wrong_gender": "m3", "wrong_form": "klas"},  # ru: класс
        "szansa": {"wrong_gender": "m3", "wrong_form": "szans"},
        "plaża": {"wrong_gender": "m3", "wrong_form": "plaż"},
        "kontrola": {"wrong_gender": "m3", "wrong_form": "kontrol"},
        "telegram": {"wrong_gender": "f", "wrong_form": "telegrama"},
    }
    gender_mismatches.update(manual_overrides)

    os.makedirs(os.path.dirname(GENDER_MISMATCH_PATH), exist_ok=True)
    with open(GENDER_MISMATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(gender_mismatches, f, ensure_ascii=False, indent=4)
    logging.info(
        f"Dynamically generated {len(gender_mismatches)} generalized gender mismatch"
        f" rules to {GENDER_MISMATCH_PATH}"
    )


if __name__ == "__main__":
    main()
