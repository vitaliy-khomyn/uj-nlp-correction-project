import os
import json
import logging
import requests
from dotenv import load_dotenv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.paths import PREP_MISMATCH_PATH, LLM_DUMP_PATH

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# We use Groq's OpenAI-compatible endpoint.
API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.environ.get("GROQ_API_KEY")

SYSTEM_PROMPT = """
You are an expert Polish, Russian, and Ukrainian linguist specializing in L2 language acquisition and cross-lingual interference.
Your task is to generate a comprehensive JSON database of prepositional government (rekcja przyimkowa) mistakes
and spatial preposition calques made by native Russian/Ukrainian speakers when speaking Polish.

You must output ONLY valid JSON.

The JSON must have two root keys: "VERB_PREP_ERRORS" and "NOUN_PREP_ERRORS".
Inside each, map the Polish lemma to its correct Polish preposition, and then to a list containing the INCORRECT preposition used by RU/UA speakers and the resulting grammatical case abbreviation ('nom', 'gen', 'dat', 'acc', 'inst', 'loc', 'voc').

Example Format:
{
  "VERB_PREP_ERRORS": {
    "czekać": {"na": ["dla", "gen"]},
    "śmiać": {"z": ["nad", "inst"]},
    "tęsknić": {"za": ["po", "loc"]},
    "ożenić": {"z": ["na", "loc"]},
    "znać": {"na": ["w", "loc"]}
  },
  "NOUN_PREP_ERRORS": {
    "uniwersytet": {"na": ["w", "loc"]},
    "firma": {"w": ["na", "loc"]},
    "krym": {"na": ["w", "loc"]},
    "weekend": {"w": ["na", "loc"]}
  }
}
"""

USER_PROMPT = """
Generate a highly exhaustive list of at least 60 VERB_PREP_ERRORS and 40 NOUN_PREP_ERRORS.
Focus on the most common traps: 'o' vs 'za', 'na' vs 'w', 'na' vs 'za', and zero-preposition drops.
Make sure the case abbreviation correctly matches the case that the *wrong* preposition would force.
Do not include any markdown formatting, preambles, or postambles. Output raw JSON only.
"""


def fetch_preposition_mismatches() -> dict:
    if not API_KEY:
        logging.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        return {}

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # We loop through highly reliable Groq models until one succeeds.
    models_to_try = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]

    for model_id in models_to_try:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            "temperature": 0.3,
            "max_tokens": 8000
        }

        logging.info(f"Attempting generation with model: {model_id}...")
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]

            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            parsed_json = json.loads(content.strip())
            return parsed_json

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status in [400, 403, 404, 503, 429]:
                logging.warning(f"Model {model_id} failed ({status}). Trying next fallback...")
                continue
            logging.error(f"API Request failed with status {status}: {e.response.text}")
            break
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}")
            break
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response as JSON from {model_id}. The model may have hallucinated formatting.")
            os.makedirs(os.path.dirname(LLM_DUMP_PATH), exist_ok=True)
            with open(LLM_DUMP_PATH, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Raw output dumped to {LLM_DUMP_PATH} for inspection.")
            break

    return {}


def main():
    generated_data = fetch_preposition_mismatches()

    if not generated_data:
        logging.warning("No data was generated. Exiting.")
        return

    verb_errors = generated_data.get("VERB_PREP_ERRORS", {})
    noun_errors = generated_data.get("NOUN_PREP_ERRORS", {})

    logging.info(f"Successfully generated {len(verb_errors)} verb mismatches and {len(noun_errors)} noun mismatches.")

    os.makedirs(os.path.dirname(PREP_MISMATCH_PATH), exist_ok=True)
    with open(PREP_MISMATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved to {PREP_MISMATCH_PATH}")
    logging.info("Review the JSON file to ensure the linguistic rules are accurate before integrating.")


if __name__ == "__main__":
    main()
