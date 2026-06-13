"""Orchestrates algorithmic data corruption simulating L1->L2 grammatical interference."""
import json
import logging
import os
import pandas as pd
import random
import spacy
from typing import Dict, List, Optional, Tuple, Any, Set
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utils.paths import UNIFIED_FF_PATH, GENDER_MISMATCH_PATH, PREP_MISMATCH_PATH, POLIMORF_PARQUET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SPACY_BATCH_SIZE: int = 50
RANDOM_SEED: int = 42
MAX_PAIRS_MULTIPLIER: int = 4
TARGET_PAIRS_MULTIPLIER: int = 2
PRINT_EVERY_N: int = 1000

POS_MAP: Dict[str, str] = {"NOUN": "subst", "ADJ": "adj", "VERB": "fin"}
CASE_MAP: Dict[str, str] = {
    "Nom": "nom",
    "Gen": "gen",
    "Dat": "dat",
    "Acc": "acc",
    "Inst": "inst",
    "Loc": "loc",
    "Voc": "voc",
}
NUMBER_MAP: Dict[str, str] = {"Sing": "sg", "Plur": "pl"}
PERSON_MAP: Dict[str, str] = {"1": "pri", "2": "sec", "3": "ter"}

PREP_CONFUSIONS: Dict[str, List[str]] = {
    "w": ["na", "do", "z"],
    "na": ["w", "dla", "o"],
    "z": ["od", "w", "przez"],
    "do": ["dla", "w", "na"],
    "dla": ["do", "za", "na"],
    "od": ["z", "przez", "na"],
    "o": ["na", "w", "z"],
}
PREP_DEFAULT_CASES: Dict[str, str] = {
    "w": "loc",
    "na": "loc",
    "z": "inst",
    "do": "gen",
    "dla": "gen",
    "od": "gen",
    "o": "loc",
}

VERB_PREP_ERRORS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "czekać": {"na": ("dla", "gen")},
    "śmiać": {"z": ("nad", "inst")},
    "tęsknić": {"za": ("po", "loc")},
    "zależeć": {"od": ("na", "loc")},
    "jechać": {"do": ("w", "acc")},
    "iść": {"do": ("w", "acc")},
    "przyzwyczaić": {"do": ("k", "dat")},
    "ożenić": {"z": ("na", "loc")},
    "znać": {"na": ("w", "loc")},
    "przyznać": {"do": ("w", "loc")},
    "dbać": {"o": ("za", "inst")},
}

NOUN_PREP_ERRORS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "uniwersytet": {"na": ("w", "loc")},
    "firma": {"w": ("na", "loc")},
    "kuchnia": {"w": ("na", "loc")},
    "początek": {"na": ("w", "loc")},
    "koniec": {"na": ("w", "loc")},
    "urlop": {"na": ("w", "loc")},
    "wakacje": {"na": ("w", "loc")},
    "praca": {"w": ("na", "loc")},
    "zakupy": {"na": ("za", "inst")},
    "krym": {"na": ("w", "loc")},
}

TYPO_MAP: Dict[str, List[str]] = {
    "u": ["ó", "y", "i"],
    "ó": ["u"],
    "ż": ["rz", "z"],
    "h": ["ch", "g"],
    "g": ["h"],
    "a": ["s", "z", "q", "w"],
    "s": ["a", "d", "w", "x", "ś", "sz"],
    "e": ["w", "r", "d", "ę"],
    "i": ["y", "u", "o", "j", "k"],
    "o": ["i", "p", "l", "ó"],
    "ą": ["a", "om", "on"],
    "ę": ["e", "em", "en"],
    "y": ["i", "u", "t", "h"],
    "w": ["e", "q", "s", "a", "v", "f"],
    "r": ["e", "t", "f", "d", "rz"],
    "t": ["r", "y", "g", "f"],
    "z": ["x", "s", "a", "ż", "ź"],
    "c": ["x", "v", "d", "f", "ć", "cz"],
    "n": ["b", "m", "h", "j", "ń"],
}


def load_resources(
    unified_ff_path: str,
    polimorf_path: str,
    gender_mismatch_path: str,
    prep_mismatch_path: str,
) -> Tuple[
    Optional[List[Dict[str, Any]]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    """Loads the false friends map and PoliMorf memory dictionary.

    Args:
        unified_ff_path: Unified false friends database path.
        polimorf_path: Parsed PoliMorf dictionary path.
        gender_mismatch_path: Path to gender mismatch rules.
        prep_mismatch_path: Path to preposition mismatch rules.

    Returns:
        A tuple containing loaded datasets for false friends, PoliMorf, gender errors, and preposition errors.
    """
    if not all(
        os.path.exists(p)
        for p in [unified_ff_path, polimorf_path, gender_mismatch_path]
    ):
        logging.error("Missing required primary resource files.")
        raise RuntimeError("Missing required primary resource files.")

    prep_data = {}
    if os.path.exists(prep_mismatch_path):
        try:
            with open(prep_mismatch_path, "r", encoding="utf-8") as f:
                prep_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.warning(f"Failed to load preposition mismatches: {e}")

    try:
        with open(unified_ff_path, "r", encoding="utf-8") as f:
            ff_data = json.load(f)
        with open(gender_mismatch_path, "r", encoding="utf-8") as f:
            gender_data = json.load(f)

        logging.info("Loading PoliMorf Parquet into memory dictionary...")
        df = pd.read_parquet(polimorf_path)
        polimorf_dict: Dict[str, Dict[str, Dict[str, str]]] = {}
        for row in df.itertuples(index=False):
            if row.lemma not in polimorf_dict:
                polimorf_dict[row.lemma] = {}
            if row.pos not in polimorf_dict[row.lemma]:
                polimorf_dict[row.lemma][row.pos] = {}
            polimorf_dict[row.lemma][row.pos][row.features] = row.form
        return ff_data, polimorf_dict, gender_data, prep_data
    except Exception as e:
        raise RuntimeError(f"Failed to load resource files: {e}")


def build_target_map(ff_data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Builds a map from a correct Polish lemma to its false friend lemma.

    Args:
        ff_data: List of false friends dictionary records.

    Returns:
        A target translation to wrong lemma dictionary mapping.
    """
    target_map: Dict[str, str] = {}
    for item in ff_data:
        error_lemma = item.get("pl_word", "").lower()
        if not error_lemma:
            continue
        for lang, ff_info in item.get("false_friends", {}).items():
            meanings = ff_info.get("meaning", [])
            if isinstance(meanings, str):
                meanings = [meanings]
            for m in meanings:
                m_clean = m.strip().lower()
                if len(m_clean) > 2:
                    target_map[m_clean] = error_lemma
    logging.info(f"Built {len(target_map)} target -> error mappings.")
    return target_map


def get_first_morph(morph: Any, key: str) -> Optional[str]:
    """Extracts the primary morphological feature from a SpaCy morph object.

    Args:
        morph: SpaCy token morphology dictionary or object.
        key: The morphological feature to look up.

    Returns:
        The extracted feature string.
    """
    val = morph.get(key)
    return val[0] if val else None


def get_spacy_gender(morph: Any) -> Optional[str]:
    """Derives PoliMorf-compatible gender representation from SpaCy's morph output.

    Args:
        morph: SpaCy morphology object.

    Returns:
        Short gender identifier.
    """
    gender = get_first_morph(morph, "Gender")
    animacy = get_first_morph(morph, "Animacy")
    if gender == "Masc":
        if animacy == "Hum":
            return "m1"
        if animacy == "Inan":
            return "m3"
        return "m2"
    if gender == "Fem":
        return "f"
    if gender == "Neut":
        return "n"
    return None


class PolimorfCache:
    """Cache wrapper for fast PoliMorf dictionary lookups."""

    def __init__(self, polimorf_dict: Dict[str, Dict[str, Dict[str, str]]]):
        """Initializes the cache.

        Args:
            polimorf_dict: The inner nested PoliMorf lookup table.
        """
        self.cache = polimorf_dict

    def find_inflected_form(
        self, lemma: str, pos: str, morph_key: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Looks up the inflected form of a lemma.

        Args:
            lemma: The base word.
            pos: Part of speech.
            morph_key: The morphological string constraint.

        Returns:
            Form and alternative gender.
        """
        if lemma not in self.cache:
            return None, None
        lemma_pos_data = self.cache[lemma].get(pos, {})

        if morph_key in lemma_pos_data:
            return lemma_pos_data[morph_key], morph_key.split(":")[-1]

        if pos == "subst" and morph_key.endswith(":m"):
            base_key = morph_key[:-2]
            for m_gen in [":m1", ":m2", ":m3"]:
                alt_key = base_key + m_gen
                if alt_key in lemma_pos_data:
                    return lemma_pos_data[alt_key], m_gen[1:]
        return None, None

    def find_inflected_adj_or_det(
        self, lemma: str, num: str, case: str, gender: str
    ) -> Optional[str]:
        """Finds the inflected form of an adjective or determiner.

        Args:
            lemma: Adjective/determiner lemma.
            num: Number.
            case: Grammatical case.
            gender: Grammatical gender.

        Returns:
            The inflected word form.
        """
        if lemma not in self.cache:
            return None
        lemma_adj_data = self.cache[lemma].get("adj", {})

        exact_key = f"{num}:{case}:{gender}"
        if exact_key in lemma_adj_data:
            return lemma_adj_data[exact_key]

        if gender and gender.startswith("m"):
            for m_gen in ["m1", "m2", "m3"]:
                alt_key = f"{num}:{case}:{m_gen}"
                if alt_key in lemma_adj_data:
                    return lemma_adj_data[alt_key]
        return None

    def find_inflected_verb(self, lemma: str, num: str, person: str) -> Optional[str]:
        """Finds the inflected form of a finite verb.

        Args:
            lemma: Verb lemma.
            num: Number.
            person: Person (e.g. pri, sec, ter).

        Returns:
            The inflected verb form.
        """
        if lemma not in self.cache:
            return None
        lemma_fin_data = self.cache[lemma].get("fin", {})

        for aspect in ["imperf", "perf"]:
            key = f"{num}:{person}:{aspect}"
            if key in lemma_fin_data:
                return lemma_fin_data[key]
        return None

    def find_inflected_numeral(
        self, lemma: str, case: str, gender: str
    ) -> Optional[str]:
        """Finds the inflected form of a numeral.

        Args:
            lemma: Numeral lemma.
            case: Grammatical case.
            gender: Grammatical gender.

        Returns:
            The inflected numeral form.
        """
        if lemma not in self.cache:
            return None
        lemma_num_data = self.cache[lemma].get("num", {})

        genders_to_try = [gender]
        if gender in ("m1", "m2", "m3"):
            genders_to_try.extend([g for g in ("m1", "m2", "m3") if g != gender])

        for g in genders_to_try:
            for ac in ["rec", "congr"]:
                key = f"pl:{case}:{g}:{ac}"
                if key in lemma_num_data:
                    return lemma_num_data[key]
        return None

    def find_inflected_past_verb(
        self, lemma: str, num: str, gender: str
    ) -> Optional[str]:
        """Finds the inflected form of a past tense verb.

        Args:
            lemma: Verb lemma.
            num: Number.
            gender: Grammatical gender.

        Returns:
            The inflected past verb form.
        """
        if lemma not in self.cache:
            return None
        lemma_praet_data = self.cache[lemma].get("praet", {})

        genders_to_try = [gender]
        if gender in ("m1", "m2", "m3"):
            genders_to_try.extend([g for g in ("m1", "m2", "m3") if g != gender])
        if num == "pl" and gender != "m1":
            genders_to_try.extend(["f", "n", "m2", "m3", "m2.m3.f.n"])

        for g in genders_to_try:
            for aspect in ["imperf", "perf"]:
                key = f"{num}:{g}:{aspect}"
                if key in lemma_praet_data:
                    return lemma_praet_data[key]
        return None


def _match_case(original: str, new_form: str) -> str:
    """Preserves capitalization of original tokens on new_form."""
    if not original or not new_form:
        return new_form
    if original.istitle():
        return new_form.capitalize()
    if original.isupper():
        return new_form.upper()
    return new_form


def _get_modifiers_to_update(noun_token: Any, tokens: List[Any]) -> List[Any]:
    """Finds all adjectives/determiners modifying the noun.

    Uses dependencies and linear window lookups.
    """
    modifiers = set()

    # dependency-based children
    for child in noun_token.children:
        if child.pos_ in ("ADJ", "DET") and child.dep_ in ("amod", "det"):
            modifiers.add(child)

    # linear window heuristics (check 2 words before and 1 word after the noun)
    noun_idx = noun_token.i
    for offset in (-1, -2):
        idx = noun_idx + offset
        if 0 <= idx < len(tokens):
            tok = tokens[idx]
            if tok.pos_ in ("ADJ", "DET") and (
                tok.head == noun_token or tok.head == noun_token.head
            ):
                modifiers.add(tok)
    idx = noun_idx + 1
    if 0 <= idx < len(tokens):
        tok = tokens[idx]
        if tok.pos_ in ("ADJ", "DET") and (
            tok.head == noun_token or tok.head == noun_token.head
        ):
            modifiers.add(tok)

    return list(modifiers)


def _cascade_case_to_children(
    token_to_replace: Any,
    new_noun_gender: str,
    polimorf_cache: PolimorfCache,
    final_tokens: List[str],
    tokens: List[Any],
) -> None:
    """Cascades gender changes to adjectives, determiners, numerals, and past-tense verbs."""
    modifiers = _get_modifiers_to_update(token_to_replace, tokens)
    for child in modifiers:
        adj_num = NUMBER_MAP.get(get_first_morph(child.morph, "Number"))
        adj_case = CASE_MAP.get(get_first_morph(child.morph, "Case"))
        if adj_num and adj_case:
            new_adj_form = polimorf_cache.find_inflected_adj_or_det(
                child.lemma_.lower(), adj_num, adj_case, new_noun_gender
            )
            if new_adj_form:
                final_tokens[child.i] = _match_case(child.text, new_adj_form)

    # also handle numerals
    for child in token_to_replace.children:
        if child.pos_ == "NUM" and child.dep_ == "nummod":
            num_case = CASE_MAP.get(get_first_morph(child.morph, "Case"))
            if num_case:
                new_num_form = polimorf_cache.find_inflected_numeral(
                    child.lemma_.lower(), num_case, new_noun_gender
                )
                if new_num_form:
                    final_tokens[child.i] = _match_case(child.text, new_num_form)

    if token_to_replace.dep_ == "nsubj" and token_to_replace.head.pos_ == "VERB":
        head_verb = token_to_replace.head
        verb_tense = get_first_morph(head_verb.morph, "Tense")
        verb_num = NUMBER_MAP.get(get_first_morph(head_verb.morph, "Number"))
        if verb_tense == "Past" and verb_num:
            new_verb_form = polimorf_cache.find_inflected_past_verb(
                head_verb.lemma_.lower(), verb_num, new_noun_gender
            )
            if new_verb_form:
                final_tokens[head_verb.i] = _match_case(
                    head_verb.text, new_verb_form
                )


def _is_noun_governed_preposition(token: Any) -> bool:
    """Checks if the preposition directly modifies a noun."""
    return token.dep_ == "case" and token.head.pos_ == "NOUN"


def _is_verb_governed_preposition_via_noun(token: Any) -> bool:
    """Checks if the preposition modifies a noun that is governed by a verb."""
    return (
        token.dep_ == "case"
        and token.head.pos_ == "NOUN"
        and token.head.head.pos_ == "VERB"
    )


def _is_verb_direct_preposition(token: Any) -> bool:
    """Checks if the preposition directly modifies a verb."""
    return token.head.pos_ == "VERB"


def _inject_preposition(
    tokens: List[Any], polimorf_cache: PolimorfCache
) -> Tuple[
    Optional[int],
    Optional[str],
    Optional[str],
    List[Tuple[int, str]],
    Optional[str],
]:
    """Injects interference errors related to prepositional government."""
    possible_preps = [i for i, t in enumerate(tokens) if t.pos_ == "ADP"]
    random.shuffle(possible_preps)

    for target_idx in possible_preps:
        token = tokens[target_idx]
        prep_lemma = token.lemma_.lower()
        head = token.head

        new_prep, new_case = None, None
        error_key = f"prep_{prep_lemma}"

        if _is_noun_governed_preposition(token):
            noun_lemma = head.lemma_.lower()
            if noun_lemma in NOUN_PREP_ERRORS and prep_lemma in NOUN_PREP_ERRORS[noun_lemma]:
                new_prep, new_case = NOUN_PREP_ERRORS[noun_lemma][prep_lemma]
                error_key = f"prep_{noun_lemma}"

        if not new_prep and _is_verb_governed_preposition_via_noun(token):
            verb_lemma = head.head.lemma_.lower()
            if verb_lemma in VERB_PREP_ERRORS and prep_lemma in VERB_PREP_ERRORS[verb_lemma]:
                new_prep, new_case = VERB_PREP_ERRORS[verb_lemma][prep_lemma]
                error_key = f"prep_{verb_lemma}"

        if not new_prep and _is_verb_direct_preposition(token):
            verb_lemma = head.lemma_.lower()
            if verb_lemma in VERB_PREP_ERRORS and prep_lemma in VERB_PREP_ERRORS[verb_lemma]:
                new_prep, new_case = VERB_PREP_ERRORS[verb_lemma][prep_lemma]
                error_key = f"prep_{verb_lemma}"

        if not new_prep and prep_lemma in PREP_CONFUSIONS:
            new_prep = random.choice(PREP_CONFUSIONS[prep_lemma])
            new_case = PREP_DEFAULT_CASES.get(new_prep)

        if not new_prep:
            continue

        changes = []
        if _is_noun_governed_preposition(token):
            noun = token.head
            if new_case:
                num = NUMBER_MAP.get(get_first_morph(noun.morph, "Number"))
                gender = get_spacy_gender(noun.morph)
                if num and gender:
                    morph_key = f"{num}:{new_case}:{gender}"
                    new_noun_form, _ = polimorf_cache.find_inflected_form(
                        noun.lemma_.lower(), "subst", morph_key
                    )
                    if new_noun_form:
                        changes.append(
                            (noun.i, _match_case(noun.text, new_noun_form))
                        )

                        modifiers = _get_modifiers_to_update(noun, tokens)
                        for child in modifiers:
                            child_num = NUMBER_MAP.get(
                                get_first_morph(child.morph, "Number")
                            )
                            if child_num:
                                new_adj_form = (
                                    polimorf_cache.find_inflected_adj_or_det(
                                        child.lemma_.lower(),
                                        child_num,
                                        new_case,
                                        gender,
                                    )
                                )
                                if new_adj_form:
                                    changes.append(
                                        (
                                            child.i,
                                            _match_case(child.text, new_adj_form),
                                        )
                                    )

        return target_idx, error_key, new_prep, changes, None

    return None, None, None, [], None


def _inject_case(
    tokens: List[Any], polimorf_cache: PolimorfCache
) -> Tuple[
    Optional[int],
    Optional[str],
    Optional[str],
    List[Tuple[int, str]],
    Optional[str],
]:
    """Corrupts the grammatical case of nouns to emulate structural L1 translation errors."""
    possible_nouns = [
        i for i, t in enumerate(tokens) if t.pos_ == "NOUN" and t.morph.get("Case")
    ]
    if possible_nouns:
        target_idx = random.choice(possible_nouns)
        token = tokens[target_idx]
        num = NUMBER_MAP.get(get_first_morph(token.morph, "Number"))
        orig_case = CASE_MAP.get(get_first_morph(token.morph, "Case"))
        orig_gender = get_spacy_gender(token.morph)

        if num and orig_case and orig_gender:
            new_case = "acc" if orig_case in ("gen", "inst", "loc") else "nom"
            morph_key = f"{num}:{new_case}:{orig_gender}"
            new_form, _ = polimorf_cache.find_inflected_form(
                token.lemma_.lower(), "subst", morph_key
            )
            if new_form:
                changes = []
                modifiers = _get_modifiers_to_update(token, tokens)
                for child in modifiers:
                    child_num = NUMBER_MAP.get(
                        get_first_morph(child.morph, "Number")
                    )
                    if child_num:
                        new_adj_form = polimorf_cache.find_inflected_adj_or_det(
                            child.lemma_.lower(), child_num, new_case, orig_gender
                        )
                        if new_adj_form:
                            changes.append(
                                (child.i, _match_case(child.text, new_adj_form))
                            )

                # update numerals if present
                for child in token.children:
                    if child.pos_ == "NUM" and child.dep_ == "nummod":
                        new_num_form = polimorf_cache.find_inflected_numeral(
                            child.lemma_.lower(), new_case, orig_gender
                        )
                        if new_num_form:
                            changes.append(
                                (child.i, _match_case(child.text, new_num_form))
                            )

                return (
                    target_idx,
                    f"case_{token.lemma_.lower()}",
                    new_form,
                    changes,
                    None,
                )
    return None, None, None, [], None


def _inject_misgendering(
    tokens: List[Any], gender_map: Dict[str, Any], polimorf_cache: PolimorfCache
) -> Tuple[
    Optional[int],
    Optional[str],
    Optional[str],
    List[Tuple[int, str]],
    Optional[str],
]:
    """Misgenders nouns and cascades the error to their modifiers."""
    possible_nouns = [
        i
        for i, t in enumerate(tokens)
        if t.pos_ == "NOUN" and t.lemma_.lower() in gender_map
    ]
    if not possible_nouns:
        return None, None, None, [], None

    target_idx = random.choice(possible_nouns)
    token = tokens[target_idx]

    gender_info = gender_map[token.lemma_.lower()]
    wrong_gender = gender_info["wrong_gender"]
    base_wrong_form = gender_info.get("wrong_form", token.text)

    changes = []
    modifiers = _get_modifiers_to_update(token, tokens)
    for child in modifiers:
        num = NUMBER_MAP.get(get_first_morph(child.morph, "Number"))
        case = CASE_MAP.get(get_first_morph(child.morph, "Case"))
        if num and case:
            new_form = polimorf_cache.find_inflected_adj_or_det(
                child.lemma_.lower(), num, case, wrong_gender
            )
            if new_form:
                changes.append((child.i, _match_case(child.text, new_form)))

    # handle past tense verb head agreement
    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
        head_verb = token.head
        verb_tense = get_first_morph(head_verb.morph, "Tense")
        verb_num = NUMBER_MAP.get(get_first_morph(head_verb.morph, "Number"))

        if verb_tense == "Past" and verb_num:
            new_verb_form = polimorf_cache.find_inflected_past_verb(
                head_verb.lemma_.lower(), verb_num, wrong_gender
            )
            if new_verb_form:
                changes.append(
                    (head_verb.i, _match_case(head_verb.text, new_verb_form))
                )

    if not changes and base_wrong_form.lower() == token.text.lower():
        return None, None, None, [], None

    return target_idx, token.lemma_.lower(), base_wrong_form, changes, None


def _inject_false_friend(
    tokens: List[Any],
    target_map: Dict[str, str],
    polimorf_cache: PolimorfCache,
    injection_counts: Dict[str, int],
    max_injections: int,
) -> Tuple[
    Optional[int],
    Optional[str],
    Optional[str],
    List[Tuple[int, str]],
    Optional[str],
]:
    """Swaps a correct word with a confusing cognate false friend."""
    possible_targets = [
        i
        for i, t in enumerate(tokens)
        if t.lemma_.lower() in target_map
        and t.pos_ in POS_MAP
        and not t.ent_type_
        and t.dep_ != "fixed"
    ]
    if not possible_targets:
        return None, None, None, [], None

    target_idx = random.choice(possible_targets)
    token = tokens[target_idx]
    error_lemma = target_map[token.lemma_.lower()]

    if injection_counts.get(error_lemma, 0) >= max_injections:
        return None, None, None, [], None

    polimorf_pos = POS_MAP[token.pos_]
    num = NUMBER_MAP.get(get_first_morph(token.morph, "Number"))
    case = CASE_MAP.get(get_first_morph(token.morph, "Case"))
    person = PERSON_MAP.get(get_first_morph(token.morph, "Person"))
    orig_gender = get_spacy_gender(token.morph)

    new_form, new_noun_gender = None, None
    if polimorf_pos == "subst" and num and case:
        # search for a matching number and case, ignoring original gender
        lemma_subst_data = polimorf_cache.cache.get(error_lemma, {}).get("subst", {})
        prefix = f"{num}:{case}:"
        for key, form in lemma_subst_data.items():
            if key.startswith(prefix):
                new_form = form
                new_noun_gender = key.split(":")[-1]
                break
        if not new_form:
            generic_key = f"{num}:{case}"
            if generic_key in lemma_subst_data:
                new_form = lemma_subst_data[generic_key]
                new_noun_gender = None
    elif polimorf_pos == "adj" and num and case and orig_gender:
        new_form = polimorf_cache.find_inflected_adj_or_det(
            error_lemma, num, case, orig_gender
        )
    elif polimorf_pos == "fin" and num and person:
        new_form = polimorf_cache.find_inflected_verb(error_lemma, num, person)

    if not new_form:
        return None, None, None, [], None

    if (
        token.pos_ == "ADJ"
        and token.text.lower().startswith("nie")
        and not new_form.lower().startswith("nie")
    ):
        new_form = "nie" + new_form

    return target_idx, error_lemma, new_form, [], new_noun_gender


def _fix_phonotactics(tokens: List[str], target_idx: int) -> None:
    """Fixes orphaned phonotactic prepositions (w/we, z/ze)."""
    if target_idx > 0:
        prev_word = tokens[target_idx - 1].lower()
        next_word = tokens[target_idx].lower()
        vowels = "aeiouyąęó"
        if prev_word in ("w", "we"):
            needs_we = (
                next_word.startswith(("w", "f"))
                and len(next_word) > 1
                and next_word[1] not in vowels
            )
            correct_prep = "we" if needs_we else "w"
            tokens[target_idx - 1] = (
                correct_prep.capitalize()
                if tokens[target_idx - 1].istitle()
                else correct_prep
            )
        elif prev_word in ("z", "ze"):
            needs_ze = (
                next_word.startswith(("z", "s", "ź", "ś", "ż", "sz", "rz", "cz"))
                and len(next_word) > 1
                and next_word[1] not in vowels
            )
            correct_prep = "ze" if needs_ze else "z"
            tokens[target_idx - 1] = (
                correct_prep.capitalize()
                if tokens[target_idx - 1].istitle()
                else correct_prep
            )


def process_doc(
    doc: Any,
    target_map: Dict[str, str],
    gender_map: Dict[str, Any],
    polimorf_cache: PolimorfCache,
    injection_counts: Dict[str, int],
    max_injections: int,
) -> Optional[Tuple[str, str]]:
    """The core logic to inject a false friend into a single SpaCy Doc object.

    Args:
        doc: The input SpaCy document.
        target_map: Polish lemma to false friend map.
        gender_map: Noun misgendering rules dictionary.
        polimorf_cache: PoliMorf lookup database cache.
        injection_counts: Injected counts tracker dictionary.
        max_injections: Maximum allowed mutations.

    Returns:
        The mutated sentence and the associated error lemma, or None.
    """
    tokens = list(doc)
    final_tokens = [t.text for t in tokens]

    injected_any = False
    primary_error_lemma = None

    # enforce at most one mutation strategy per sentence to keep syntax clean
    mutation_chance = 0.70
    if random.random() < mutation_chance:
        strategies = [
            (
                "false_friend",
                0.30,
                lambda: _inject_false_friend(
                    tokens,
                    target_map,
                    polimorf_cache,
                    injection_counts,
                    max_injections,
                ),
            ),
            (
                "preposition",
                0.25,
                lambda: _inject_preposition(tokens, polimorf_cache),
            ),
            ("case", 0.25, lambda: _inject_case(tokens, polimorf_cache)),
            (
                "misgendering",
                0.20,
                lambda: _inject_misgendering(tokens, gender_map, polimorf_cache),
            ),
        ]

        # weighted choice of strategy
        selected_strategy = random.choices(
            strategies, weights=[s[1] for s in strategies], k=1
        )[0]

        name, _, strategy_func = selected_strategy
        target_idx, error_lemma, new_form, extra_changes, new_noun_gender = (
            strategy_func()
        )

        if target_idx is not None and new_form is not None:
            token_to_replace = tokens[target_idx]
            final_tokens[target_idx] = _match_case(token_to_replace.text, new_form)

            for idx, form in extra_changes:
                final_tokens[idx] = form

            if new_noun_gender:
                orig_gender = get_spacy_gender(token_to_replace.morph)
                if orig_gender and new_noun_gender != orig_gender:
                    _cascade_case_to_children(
                        token_to_replace,
                        new_noun_gender,
                        polimorf_cache,
                        final_tokens,
                        tokens,
                    )

            _fix_phonotactics(final_tokens, target_idx)

            if name == "false_friend":
                injection_counts[error_lemma] = (
                    injection_counts.get(error_lemma, 0) + 1
                )

            primary_error_lemma = error_lemma
            injected_any = True

    if injected_any and random.random() < 0.05:
        eligible_indices = [
            i for i, t in enumerate(final_tokens) if len(t) > 3 and t.isalpha()
        ]
        if eligible_indices:
            idx = random.choice(eligible_indices)
            word = final_tokens[idx]
            char_idx = random.randint(1, len(word) - 2)
            c = word[char_idx].lower()
            if c in TYPO_MAP:
                replacement = random.choice(TYPO_MAP[c])
                if word[char_idx].isupper():
                    replacement = replacement.capitalize()
                final_tokens[idx] = (
                    word[:char_idx] + replacement + word[char_idx + 1 :]
                )
            else:
                final_tokens[idx] = word[:char_idx] + word[char_idx + 1 :]

    if not injected_any:
        return None

    joined_text = "".join(
        [text + t.whitespace_ for text, t in zip(final_tokens, tokens)]
    )
    return joined_text.strip(), primary_error_lemma


def main(
    corpus_path: str,
    output_path: str,
    max_pairs: int = 5000,
    max_injections_per_word: int = 50,
    min_sentence_length: int = 6,
) -> None:
    """Executes the sentence synthesis pipeline.

    Args:
        corpus_path: Parquet path to input text corpus.
        output_path: Parquet path to write synthetic sentences.
        max_pairs: Upper bound limit of generated output sentence pairs.
        max_injections_per_word: Injections cap count per target word.
        min_sentence_length: Word count minimum bounds for sentence selection.
    """
    ff_data, polimorf_dict, gender_map, prep_data = load_resources(
        UNIFIED_FF_PATH,
        POLIMORF_PARQUET_PATH,
        GENDER_MISMATCH_PATH,
        PREP_MISMATCH_PATH,
    )
    if not (ff_data and polimorf_dict and gender_map):
        raise RuntimeError("Failed to load required resources.")

    if prep_data:
        VERB_PREP_ERRORS.update(prep_data.get("VERB_PREP_ERRORS", {}))
        NOUN_PREP_ERRORS.update(prep_data.get("NOUN_PREP_ERRORS", {}))
        logging.info(
            f"Loaded additional LLM preposition errors. Total verb rules:"
            f" {len(VERB_PREP_ERRORS)}, noun rules: {len(NOUN_PREP_ERRORS)}"
        )

    target_map = build_target_map(ff_data)
    injection_counts = {ff: 0 for ff in set(target_map.values())}
    hard_negative_counts = {ff: 0 for ff in set(target_map.values())}

    polimorf_cache = PolimorfCache(polimorf_dict)

    try:
        nlp = spacy.load("pl_core_news_lg", disable=["attribute_ruler"])
    except OSError:
        logging.error(
            "SpaCy model not found. Run: python -m spacy download pl_core_news_lg"
        )
        raise RuntimeError(
            "SpaCy model not found. Run: python -m spacy download pl_core_news_lg"
        )

    logging.info(f"Loading and deduplicating corpus from {corpus_path} in batches...")

    df_corpus = pd.read_parquet(corpus_path, columns=["text"])
    df_corpus["norm_key"] = (
        df_corpus["text"]
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
    )
    df_corpus = df_corpus.dropna(subset=["norm_key"])
    df_corpus = df_corpus[df_corpus["norm_key"] != ""]
    df_corpus = df_corpus.drop_duplicates(subset=["norm_key"])
    real_corpus = df_corpus["text"].tolist()
    logging.info(f"Deduplicated corpus to {len(real_corpus)} unique sentences.")

    random.seed(RANDOM_SEED)
    random.shuffle(real_corpus)

    dataset_rows = []
    pair_id = 0

    if min_sentence_length > 0:
        filtered_corpus = [
            s for s in real_corpus if len(s.split()) >= min_sentence_length
        ]
    else:
        filtered_corpus = real_corpus
    corpus_to_process = filtered_corpus[: max_pairs * MAX_PAIRS_MULTIPLIER]
    logging.info(
        f"Processing up to {len(corpus_to_process)} sentences (min length:"
        f" {min_sentence_length}) for sentence-level synthesis..."
    )

    n_process = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
    docs = nlp.pipe(
        corpus_to_process, batch_size=SPACY_BATCH_SIZE, n_process=n_process
    )

    for i, doc in enumerate(docs):
        if len(dataset_rows) >= max_pairs * TARGET_PAIRS_MULTIPLIER:
            logging.info("Target dataset size reached. Stopping synthesis.")
            break

        result = process_doc(
            doc,
            target_map,
            gender_map,
            polimorf_cache,
            injection_counts,
            max_injections_per_word,
        )

        native_ffs = {
            t.lemma_.lower()
            for t in doc
            if t.lemma_.lower() in injection_counts.keys() and t.pos_ in POS_MAP
        }
        for ff in native_ffs:
            if hard_negative_counts[ff] < max_injections_per_word:
                dataset_rows.append(
                    {
                        "sentence": doc.text,
                        "label": 0,
                        "pair_id": pair_id,
                        "error_lemma": ff,
                    }
                )
                pair_id += 1
                hard_negative_counts[ff] += 1

        if result:
            new_text_unit, error_lemma = result
            if new_text_unit.strip() != doc.text.strip():
                dataset_rows.append(
                    {
                        "sentence": doc.text,
                        "label": 0,
                        "pair_id": pair_id,
                        "error_lemma": error_lemma,
                    }
                )
                dataset_rows.append(
                    {
                        "sentence": new_text_unit,
                        "label": 1,
                        "pair_id": pair_id,
                        "error_lemma": error_lemma,
                    }
                )
                pair_id += 1

        if (i + 1) % PRINT_EVERY_N == 0:
            logging.info(
                f"Processed {i + 1} sentences. Current pairs:"
                f" {len(dataset_rows)//2}/{max_pairs}"
            )

    df_synth = pd.DataFrame(dataset_rows)
    logging.info(f"Generated {len(df_synth)} rows ({len(df_synth)//2} injections).")

    used_ffs = {k: v for k, v in injection_counts.items() if v > 0}
    logging.info("Used false friends statistics:")
    for ff, count in sorted(used_ffs.items(), key=lambda item: item[1], reverse=True)[
        :5
    ]:
        logging.info(f" - {ff}: {count} times")

    if not df_synth.empty:
        df_synth.to_parquet(output_path)
        logging.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
