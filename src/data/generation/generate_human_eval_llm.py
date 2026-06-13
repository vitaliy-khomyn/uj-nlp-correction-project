"""Script to orchestrate LLM-based generation of the human evaluation dataset."""
import os
import json
import logging
from dotenv import load_dotenv
import sys
from typing import List, Dict, TypedDict, Set
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utils.paths import HUMAN_EVAL_PATH
from src.utils.llm_client import ask_llm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SYSTEM_PROMPT: str = """
You are an expert Polish, Russian, and Ukrainian linguist specializing in L2 language acquisition and cross-lingual interference.
Your task is to generate a comprehensive JSON dataset of authentic, messy Polish sentences written by native Russian/Ukrainian speakers, exhibiting grammatical errors, false friends, wrong prepositions, and misgendering.

Focus ONLY on structural, grammatical, and lexical interferences. Do NOT add any extra commentary or explanations.
"""

USER_PROMPT_TEMPLATE: str = """
Generate a highly diverse list of exactly {error_count} authentic L2 Polish sentences containing interference errors,
and exactly {identity_count} perfectly correct Polish sentences (where source == expected).
The sentences MUST be about this topic: "{topic}".

Mix preposition errors, false friends, case errors, and typical spelling mistakes.
Ensure every sentence is unique and distinct from the examples.
"""


class SentencePair(TypedDict):
    source: str
    expected: str


def fetch_eval_sentences(
    target_errors: int = 400, target_identities: int = 100
) -> List[Dict[str, str]]:
    """Fetches evaluation sentences using the LLM client.

    Args:
        target_errors: Number of target erroneous sentences.
        target_identities: Number of target identity sentences.

    Returns:
        The generated sentence pairs.
    """
    models_to_try: List[str] = ["gemini-3.5-flash"]

    topics: List[str] = [
        "traveling and holidays",
        "job interviews and office work",
        "shopping and groceries",
        "family and relationships",
        "hobbies and sports",
        "technology and computers",
        "health and going to the doctor",
        "school and university life",
        "weather and nature",
        "renting an apartment and home life",
        "cooking and restaurants",
        "movies and entertainment",
        "public transport and commuting",
        "learning foreign languages",
    ]

    unique_sources: Set[str] = set()
    final_pairs: List[Dict[str, str]] = []
    error_count: int = 0
    identity_count: int = 0
    attempts: int = 0
    max_attempts: int = len(topics) * 3

    while (
        error_count < target_errors or identity_count < target_identities
    ) and attempts < max_attempts:
        topic = random.choice(topics)
        req_errors = min(20, target_errors - error_count)
        req_identities = min(5, target_identities - identity_count)

        if req_errors == 0 and req_identities == 0:
            break

        user_prompt = USER_PROMPT_TEMPLATE.format(
            error_count=req_errors, identity_count=req_identities, topic=topic
        )

        logging.info(f"Requesting {req_errors} err / {req_identities} id pairs about '{topic}'...")
        attempts += 1

        try:
            result = ask_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=8192,
                models_to_try=models_to_try,
                response_schema=list[SentencePair],
            )

            if result:
                for item in result:
                    src = item.get("source", "").strip()
                    tgt = item.get("expected", "").strip()

                    if not src or not tgt or src in unique_sources:
                        continue

                    is_identity = src == tgt
                    if is_identity and identity_count < target_identities:
                        unique_sources.add(src)
                        final_pairs.append({"source": src, "expected": tgt})
                        identity_count += 1
                    elif not is_identity and error_count < target_errors:
                        unique_sources.add(src)
                        final_pairs.append({"source": src, "expected": tgt})
                        error_count += 1

            logging.info(
                f"Progress: {error_count}/{target_errors} errors, {identity_count}/{target_identities} identities."
            )
            time.sleep(2)
        except Exception as e:
            logging.error(f"Attempt failed: {e}")
            time.sleep(5)
            continue

    return final_pairs


def main() -> None:
    """Generates the external validation dataset and saves it to a file."""
    if os.path.exists(HUMAN_EVAL_PATH):
        logging.info(
            f"External validation dataset already exists at {HUMAN_EVAL_PATH}. Skipping generation."
        )
        return
    generated_data = fetch_eval_sentences()
    if generated_data:
        os.makedirs(os.path.dirname(HUMAN_EVAL_PATH), exist_ok=True)
        with open(HUMAN_EVAL_PATH, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=4)
        logging.info(
            f"Successfully generated {len(generated_data)} external validation sentences to {HUMAN_EVAL_PATH}."
        )
    else:
        raise RuntimeError(
            "LLM failed to generate external validation sentences. Halting pipeline."
        )


if __name__ == "__main__":
    main()
