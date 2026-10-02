import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "")

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    print("Listing models with google-genai:")
    for model in client.models.list():
        print(f" - {model.name}")
except Exception as e:
    print("google-genai list error:", e)

try:
    import google.generativeai as genai_legacy
    genai_legacy.configure(api_key=api_key)
    print("\nListing models with legacy SDK:")
    for m in genai_legacy.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")
except Exception as e:
    print("legacy list error:", e)
