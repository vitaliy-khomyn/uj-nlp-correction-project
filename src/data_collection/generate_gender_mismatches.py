import json
import os
import logging
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.paths import POLIMORF_PARQUET_PATH, GENDER_MISMATCH_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    Generates a massive, automatically filtered database of Polish nouns misgendered
    by RU/UA L1 speakers using generalized morphological rules and collision detection.
    """
    if not os.path.exists(POLIMORF_PARQUET_PATH):
        logging.error(f"PoliMorf Parquet not found at {POLIMORF_PARQUET_PATH}")
        return

    df = pd.read_parquet(POLIMORF_PARQUET_PATH)
    df_subst = df[df['pos'] == 'subst']
    rows = df_subst[['lemma', 'features', 'form']].values.tolist()

    # 1. Build sets for O(1) collision detection
    # valid_lemmas = {row[0].lower() for row in rows}
    valid_forms = {row[2].lower() for row in rows}

    gender_mismatches = {}

    for lemma, features, _ in rows:
        lemma = lemma.lower()
        # 2. Length Filter: Ignore short words to avoid native roots (e.g., krem, dżem)
        if not lemma.isalpha() or len(lemma) < 5:
            continue

        wrong_form = None
        wrong_gender = None

        # Pattern 1: Greek/Latin Origin (PL -um Neut -> RU -ум Masc m3)
        if lemma.endswith('um') and ':n' in features:
            wrong_gender, wrong_form = "m3", lemma

        # Pattern 2a: Orthographic Calque (PL -mat Masc m3 -> RU -ma Fem f)
        # e.g., temat -> tema, schemat -> schema (Drop the 't')
        elif lemma.endswith('mat') and not lemma.endswith('omat') and ':m3' in features:
            wrong_gender, wrong_form = "f", lemma[:-1]

        # Pattern 2b: Orthographic Calque (PL -em / -am Masc m3 -> RU -ema / -ama Fem f)
        # e.g., problem -> problema, program -> programa (Append 'a')
        elif (lemma.endswith('em') or lemma.endswith('am')) and ':m3' in features:
            wrong_gender, wrong_form = "f", lemma + "a"

        # Pattern 3: Orthographic Calque (PL -a Fem f -> RU Consonant Masc m3)
        # Tightened to strictly medical/scientific/international suffixes
        # e.g., diagnoza -> diagnoz, analiza -> analiz
        elif lemma.endswith(('oza', 'yza', 'eza')) and ':f' in features:
            wrong_gender, wrong_form = "m3", lemma[:-1]

        # Pattern 4 REMOVED: -el/-al -> f generates too many false positives
        # (hotel, portfel, festiwal, profil are all masculine in RU).

        if wrong_form and wrong_gender:
            # 3. LEMMA COLLISION DETECTION
            # If the generated fake word is actually a real Polish word form (e.g., format -> forma, kilogram -> kilograma), DISCARD IT.
            if wrong_form != lemma and wrong_form in valid_forms:
                continue

            gender_mismatches[lemma] = {"wrong_gender": wrong_gender, "wrong_form": wrong_form}

    # Add isolated exceptions that don't fit broad regex rules
    manual_overrides = {
        "cytat": {"wrong_gender": "f", "wrong_form": "cytata"}, # PL m3 -> RU f (цитата)
        "metoda": {"wrong_gender": "m3", "wrong_form": "metod"}, # Add back method specifically
        "ból": {"wrong_gender": "f", "wrong_form": "ból"},       # PL m3 -> RU f (боль)
        "model": {"wrong_gender": "f", "wrong_form": "model"},   # Soft-sign exceptions
        "cel": {"wrong_gender": "f", "wrong_form": "cel"},
        "detal": {"wrong_gender": "f", "wrong_form": "detal"},
        "medal": {"wrong_gender": "f", "wrong_form": "medal"},   # PL m3 -> RU f (медаль)

        # --- The "-pis" group (PL m3 -> RU f) ---
        "podpis": {"wrong_gender": "f", "wrong_form": "podpis"}, # RU: подпись
        "napis": {"wrong_gender": "f", "wrong_form": "napis"},   # RU: надпись
        "zapis": {"wrong_gender": "f", "wrong_form": "zapis"},   # RU: запись
        "opis": {"wrong_gender": "f", "wrong_form": "opis"},     # RU: опись

        # --- The "-eń" group (PL m3 -> RU f) ---
        "stopień": {"wrong_gender": "f", "wrong_form": "stopień"}, # RU: степень
        "cień": {"wrong_gender": "f", "wrong_form": "cień"},       # RU: тень

        # --- Short Cognate Exceptions ---
        "klasa": {"wrong_gender": "m3", "wrong_form": "klas"},     # RU: класс
        "szansa": {"wrong_gender": "m3", "wrong_form": "szans"},   # RU: шанс
        "plaża": {"wrong_gender": "m3", "wrong_form": "plaż"},     # RU: пляж
        "kontrola": {"wrong_gender": "m3", "wrong_form": "kontrol"}, # RU: контроль
        "telegram": {"wrong_gender": "f", "wrong_form": "telegrama"} # RU: телеграмма
    }
    gender_mismatches.update(manual_overrides)

    os.makedirs(os.path.dirname(GENDER_MISMATCH_PATH), exist_ok=True)
    with open(GENDER_MISMATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(gender_mismatches, f, ensure_ascii=False, indent=4)
    logging.info(f"Dynamically generated {len(gender_mismatches)} generalized gender mismatch rules to {GENDER_MISMATCH_PATH}")


if __name__ == "__main__":
    main()
