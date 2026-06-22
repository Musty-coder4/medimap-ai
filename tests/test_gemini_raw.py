import os

from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say hello",
    )
    print("SUCCESS:", response.text)
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))
