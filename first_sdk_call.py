
import os

from dotenv import load_dotenv
from openai import OpenAI

# Lokale Konfiguration laden
load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Azure-v1-Endpoint verwenden
if not endpoint.endswith("/openai/v1"):
    endpoint += "/openai/v1"

# SDK-Client erstellen
client = OpenAI(
    base_url=endpoint + "/",
    api_key=api_key,
    timeout=30.0,
    max_retries=0,
)

# Dieselbe Anfrage wie beim REST-Aufruf
response = client.chat.completions.create(
    model=deployment,
    messages=[
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
    temperature=0,
    max_tokens=30,
)

classification = response.choices[0].message.content

print("Dokumentklasse:", classification)
print("Verwendetes Modell:", response.model)

if response.usage:
    print("Input-Tokens:", response.usage.prompt_tokens)
    print("Output-Tokens:", response.usage.completion_tokens)
