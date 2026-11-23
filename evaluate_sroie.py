import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
if not endpoint.endswith("/openai/v1"):
    endpoint += "/openai/v1"

client = OpenAI(
    base_url=endpoint + "/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    timeout=45.0,
    max_retries=1,
)

deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Zum Einstieg nur 10 Beispiele verwenden.
# Später erhöhen wir diese Zahl.
NUM_EXAMPLES = 10

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


def normalize(value):
    """Einfacher Vergleich ohne Unterschiede bei Leerzeichen
    und Groß-/Kleinschreibung."""
    value = str(value or "").upper().strip()
    return re.sub(r"\s+", " ", value)


def extract_fields(ocr_text):
    """Extrahiert strukturierte Felder aus OCR-Text."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract receipt information from OCR text. "
                    "Return valid JSON with exactly these keys: "
                    "company, date, address, total. "
                    "Use strings for every value. "
                    "Use an empty string if a value is missing. "
                    "Do not invent information."
                ),
            },
            {
                "role": "user",
                "content": ocr_text,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=350,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned an empty response")

    return json.loads(content), response.usage


print("Lade SROIE-Testbeispiele ...")

dataset = load_dataset(
    "jsdnrs/ICDAR2019-SROIE",
    split="test",
    streaming=True,
)

fields = ["company", "date", "address", "total"]
field_correct = {field: 0 for field in fields}
results = []
total_tokens = 0

for index, example in enumerate(dataset.take(NUM_EXAMPLES), start=1):
    # Vorhandene OCR-Transkripte als Texteingabe
    ocr_text = "\n".join(example["words"])

    # Annotierte Referenzwerte
    expected = example["entities"]

    start = time.perf_counter()

    try:
        predicted, usage = extract_fields(ocr_text)
        duration = time.perf_counter() - start

        for field in fields:
            if normalize(predicted.get(field)) == normalize(expected.get(field)):
                field_correct[field] += 1

        if usage:
            total_tokens += usage.total_tokens

        results.append({
            "document_id": index,
            "expected": expected,
            "predicted": predicted,
            "duration_seconds": round(duration, 2),
        })

        print(f"\nDokument {index}")
        print("Ground Truth:", expected)
        print("Vorhersage:  ", predicted)

    except Exception as error:
        print(f"\nDokument {index}: Fehler: {error}")
        results.append({
            "document_id": index,
            "expected": expected,
            "predicted": None,
            "error": str(error),
        })

with open(
    output_dir / "sroie_predictions.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(results, file, ensure_ascii=False, indent=2)

print("\n--- SROIE EVALUATION ---")

for field in fields:
    score = field_correct[field] / NUM_EXAMPLES
    print(f"{field}: {score:.1%}")

print(f"Gesamt-Tokens: {total_tokens}")
print("Ergebnisse: results/sroie_predictions.json")
