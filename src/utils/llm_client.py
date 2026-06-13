"""Utility client for sending generation requests to the Gemini API."""
import os
import json
import logging
import time
from typing import List, Optional, Any


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    models_to_try: Optional[List[str]] = None,
    dump_path: Optional[str] = None,
    response_schema: Any = None,
) -> Any:
    """Sends a request to the Gemini API with structured outputs and error dumping.

    Args:
        system_prompt: Core instructions for the LLM.
        user_prompt: Generation trigger prompt.
        temperature: Sampler temperature setting.
        max_tokens: Output length limit.
        models_to_try: List of fallback models.
        dump_path: File path to dump outputs on JSON parsing failure.
        response_schema: Optional TypedDict or Pydantic model for JSON structured output.

    Returns:
        Parsed JSON dictionary or list, or None on failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error(
            "GEMINI_API_KEY not found in environment variables. Please add it to"
            " your .env file."
        )
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        logging.error(
            "google-generativeai is not installed. Please run `pip install"
            " google-generativeai`."
        )
        return None

    genai.configure(api_key=api_key)

    if not models_to_try:
        models_to_try = ["gemini-3.5-flash", "gemini-2.5-flash"]

    for model_id in models_to_try:
        logging.info(f"Attempting generation with model: {model_id}...")
        try:
            model = genai.GenerativeModel(
                model_name=model_id, system_instruction=system_prompt
            )

            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json" if response_schema else "text/plain",
                response_schema=response_schema if response_schema else None,
            )

            response = model.generate_content(
                user_prompt, generation_config=generation_config
            )
            content = response.text

            if response_schema:
                return json.loads(content)
            else:
                return content

        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to parse LLM response as JSON from {model_id}. The model may"
                " have hallucinated formatting."
            )
            if not dump_path:
                try:
                    from src.utils.paths import LLM_DUMP_PATH

                    dump_path = LLM_DUMP_PATH
                except ImportError:
                    dump_path = os.path.join(
                        os.getcwd(), "data", "generated", "llm_raw_output_dump.txt"
                    )
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "a", encoding="utf-8") as f:
                f.write(
                    f"\n\n=== FAILED JSON PARSE [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===\n"
                )
                f.write(content if "content" in locals() else str(e))
            logging.info(f"Raw output saved to {dump_path} so you don't lose tokens!")
        except Exception as e:
            logging.error(f"Failed with {model_id}: {e}")
            continue

    return None
