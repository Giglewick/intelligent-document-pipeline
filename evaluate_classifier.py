import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

if not endpoint.endswith("/openai/v1"):
    endpoint += "/openai/v1"

client = OpenAI(
    base_url=endpoint + "/",
    api_key=api_key,
    timeout=30.0,
    max_retries=0,
)

# Synthetische Dokumente mit bekannten Labels
test_documents = [
    {
        "text": "Rechnung Nr. R-104, Gesamtbetrag 249,00 EUR.",
        "expected": "Rechnung",
    },
    {
        "text": "Hiermit bewerbe ich mich als Softwareentwicklerin.",
        "expected": "Bewerbung",
    },
    {
        "text": "Einladung zum Sommerfest am 15. Juli.",
        "expected": "Sonstiges",
    },
    {
        "text": "Bitte begleichen Sie den Rechnungsbetrag von 89 EUR.",
        "expected": "Rechnung",
    },
    {
        "text": "Sehr geehrte Damen und Herren, ich bewerbe mich um ein Praktikum.",
        "expected": "Bewerbung",
    },
    {
        "text": "Die Wartung der Heizungsanlage findet am Montag statt.",
        "expected": "Sonstiges",
    },
]

system_prompt = (
    "Du bist ein Dokumentklassifikator. "
    "Klassifiziere den Text als Rechnung, Bewerbung oder Sonstiges. "
    "Antworte ausschließlich mit dem Kategorienamen."
)

correct = 0
total_input_tokens = 0
total_output_tokens = 0
total_duration = 0.0

for index, document in enumerate(test_documents, start=1):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": document["text"]},
        ],
        temperature=0,
        max_tokens=30,
    )

    duration = time.perf_counter() - start
    total_duration += duration

    prediction = (response.choices[0].message.content or "").strip()
    expected = document["expected"]

    is_correct = prediction == expected
    correct += int(is_correct)

    if response.usage:
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens

    print(
        f"{index}. Erwartet: {expected:10} | "
        f"Vorhergesagt: {prediction:10} | "
        f"{'RICHTIG' if is_correct else 'FALSCH'} | "
        f"{duration:.2f} s"
    )

accuracy = correct / len(test_documents)

print("\n--- EVALUATION ---")
print(f"Accuracy: {accuracy:.1%}")
print(f"Korrekt: {correct}/{len(test_documents)}")
print(f"Gesamtdauer: {total_duration:.2f} s")
print(f"Input-Tokens: {total_input_tokens}")
print(f"Output-Tokens: {total_output_tokens}")
