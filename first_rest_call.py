
import os
import requests
from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

url = f"{endpoint}/chat/completions"

headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
}

payload = {
    "model": deployment,
    "messages": [
        {
            "role": "system",
            "content": (
                "Du bist ein Dokumentklassifikator. "
                "Antworte ausschließlich mit einer dieser Kategorien: "
                "Rechnung, Bewerbung, Sonstiges."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rechnungsnummer 2026-001. "
                "Gesamtbetrag: 149,00 EUR."
            ),
        },
    ],
    "temperature": 0,
    "max_tokens": 30,
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=30,
)

response.raise_for_status()

result = response.json()
classification = result["choices"][0]["message"]["content"]

print("Dokumentklasse:", classification)
