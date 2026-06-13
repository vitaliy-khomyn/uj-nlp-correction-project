"""Unified False Friend Database Builder.

Scrapes Wiktionary indexes and merges them with curated local false friend lists.
"""

import re
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utils.paths import CURATED_JSON_PATH, UNIFIED_FF_PATH


def load_curated_db() -> List[Dict[str, str]]:
    """Loads the hand-curated raw false friends JSON database.

    Returns:
        A list of curated false friend records, or an empty list if not found.
    """
    if os.path.exists(CURATED_JSON_PATH):
        with open(CURATED_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Warning: Could not find {CURATED_JSON_PATH}")
    return []


WIKTIONARY_URLS: Dict[str, str] = {
    "en": r"https://en.wiktionary.org/wiki/Appendix:False_friends_between_English_and_Polish",
    "ru": r"https://pl.wiktionary.org/wiki/Indeks:Rosyjski_-_Fa%C5%82szywi_przyjaciele",
    "uk": r"https://pl.wiktionary.org/wiki/Indeks:Ukrai%C5%84ski_-_Fa%C5%82szywi_przyjaciele",
}


def fetch_soup(url: str) -> Optional[BeautifulSoup]:
    """Fetches a URL and returns a BeautifulSoup object.

    Args:
        url: The endpoint to request.

    Returns:
        A parsed HTML soup object, or None if request failed.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like"
            " Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None


def split_meanings(text: str) -> List[str]:
    """Cleans and splits a raw meaning string into a list of individual meanings.

    Args:
        text: The raw string extracted from Wiktionary containing multiple meanings.

    Returns:
        A list of cleaned, individual meaning strings.
    """
    if not text or text == "-":
        return []

    text = re.sub(r"\(.*?\)", "", text)
    parts = re.split(r"[,;/]|\d+\.\s*", text)

    cleaned_meanings: List[str] = []
    for p in parts:
        p = re.sub(r"^[\w\s]{1,35}:\s*", "", p.strip())
        if p.strip():
            cleaned_meanings.append(p.strip())

    return cleaned_meanings


def scrape_wiktionary_table(url: str, lang_code: str) -> List[Dict[str, Any]]:
    """Scrapes false friend pairings and meanings from a structured Wiktionary table.

    Args:
        url: The URL of the Wiktionary page to scrape.
        lang_code: The L2 language code (e.g., 'ru', 'uk').

    Returns:
        A list of dictionaries containing parsed row mappings.
    """
    print(f"Fetching {lang_code.upper()} data from: {url}")
    soup = fetch_soup(url)
    if not soup:
        return []

    tables = soup.find_all("table")
    scraped_items: List[Dict[str, Any]] = []

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        headers = [th.text.strip().lower() for th in rows[0].find_all(["th", "td"])]

        col_map: Dict[str, int] = {}
        for i, header in enumerate(headers):
            if "polski" in header or "polish word" in header:
                col_map["pl_word"] = i
            elif header in [
                "rosyjski",
                "ukraiński",
                "czeski",
                "słowacki",
                "słoweński",
                "chorwacki",
                "english word",
            ]:
                col_map["l2_word"] = i
            elif (
                "znaczenie wyrazu" in header
                or "właściwe znaczenie" in header
                or "polish translation" in header
            ):
                col_map["l2_meaning"] = i
            elif "poprawne" in header or "english translation" in header:
                col_map["pl_meaning"] = i

        # check if this is a navigation table or a valid wiki table
        if len(col_map) < 2 and "wikitable" not in table.get("class", []):
            continue

        # fallback to expected maps if automatic detection is incomplete
        if len(col_map) < 4:
            if lang_code in ["ru", "sl", "hr"]:
                col_map = {"pl_word": 0, "l2_word": 1, "l2_meaning": 2, "pl_meaning": 3}
            else:
                col_map = {"pl_word": 0, "pl_meaning": 1, "l2_word": 2, "l2_meaning": 3}

        for row in rows[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) < 2:
                continue

            try:

                def clean_text(index: int) -> str:
                    if index >= len(cols):
                        return ""
                    text = cols[index].text.split("[")[0].strip()
                    # remove Ukrainian/Russian stress accent marks
                    return text.replace("\u0301", "")

                pl_word = (
                    clean_text(col_map["pl_word"])
                    .split("(")[0]
                    .split(",")[0]
                    .strip()
                    .lower()
                )
                pl_meaning_raw = clean_text(col_map["pl_meaning"])
                l2_word = (
                    clean_text(col_map["l2_word"])
                    .split("(")[0]
                    .split(",")[0]
                    .strip()
                    .lower()
                )
                l2_meaning_raw = clean_text(col_map["l2_meaning"])

                if pl_word and l2_word:
                    scraped_items.append(
                        {
                            "pl_word": pl_word,
                            "pl_meaning": split_meanings(pl_meaning_raw),
                            "l2_word": l2_word,
                            "l2_meaning": split_meanings(l2_meaning_raw),
                        }
                    )
            except (IndexError, AttributeError, KeyError):
                continue

    return scraped_items


def build_unified_database() -> List[Dict[str, Any]]:
    """Aggregates false friends data from Wiktionary and local curated sets.

    Returns:
        A sorted list of aggregated false friend entries.
    """
    unified_db: Dict[str, Dict[str, Any]] = {}

    for lang, url in WIKTIONARY_URLS.items():
        data = scrape_wiktionary_table(url, lang)
        for item in data:
            pl_word = item["pl_word"]
            if pl_word not in unified_db:
                unified_db[pl_word] = {
                    "pl_word": pl_word,
                    "pl_meaning": {},
                    "false_friends": {},
                }

            pl_meaning = item["pl_meaning"]
            if pl_meaning:
                unified_db[pl_word]["pl_meaning"][lang] = pl_meaning

            unified_db[pl_word]["false_friends"][lang] = {
                "word": item["l2_word"],
                "meaning": item["l2_meaning"],
            }

    curated_db = load_curated_db()
    for item in curated_db:
        pl_word = item["pl"].lower().replace("\u0301", "")
        if pl_word not in unified_db:
            unified_db[pl_word] = {
                "pl_word": pl_word,
                "pl_meaning": {},
                "false_friends": {},
            }

        if item.get("pl_meaning"):
            unified_db[pl_word]["pl_meaning"]["en"] = split_meanings(item["pl_meaning"])

        ua_word = item["ua"].replace("\u0301", "")
        unified_db[pl_word]["false_friends"]["uk"] = {
            "word": ua_word,
            "meaning": split_meanings(item["ua_meaning"]),
        }
        unified_db[pl_word]["false_friends"]["ru"] = {
            "word": ua_word,
            "meaning": split_meanings(item["ua_meaning"]),
        }

    return sorted(list(unified_db.values()), key=lambda x: x["pl_word"])


def main() -> None:
    """Builds and saves the unified false friends database JSON file."""
    final_output = build_unified_database()

    os.makedirs(os.path.dirname(UNIFIED_FF_PATH), exist_ok=True)

    with open(UNIFIED_FF_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(f"Generated unified JSON with {len(final_output)} Polish words to {UNIFIED_FF_PATH}")


if __name__ == "__main__":
    main()
