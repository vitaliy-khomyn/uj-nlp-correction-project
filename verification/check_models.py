import os
import google.generativeai as genai
from dotenv import load_dotenv


def main():
    """Checks and lists available Gemini models using the API key."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env")
        return

    genai.configure(api_key=api_key)

    print("Available Gemini Models for Text Generation:\n" + "-"*40)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")


if __name__ == "__main__":
    main()
