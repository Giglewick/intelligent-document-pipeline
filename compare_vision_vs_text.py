
import base64
import io
import json
import os
import re
import time

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. AZURE KONFIGURIEREN
# ============================================================

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")

if not endpoint.endswith("/openai/v1"):
    endpoint += "/openai/v1"

client = OpenAI(
    base_url=endpoint + "/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    timeout=60.0,
    max_retries=1,
)

deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Wir starten mit drei Belegen, um Kosten gering zu halten.
NUM_EXAMPLES = 3

FIELDS = ["company", "date", "address", "total"]

SYSTEM_PROMPT = (
    "Extract receipt information accurately. "
    "Return a JSON object with exactly these keys: "
    "company, date, address, total. "
    "Use strings for all values. "
    "For date, return only DD/MM/YYYY without the time. "
    "For total, return only the numeric amount without currency. "
    "For address, preserve the full address including street, "
    "postal code, city and state when present. "
    "Do not invent missing information."
)


# ============================================================
# 2. HILFSFUNKTIONEN
# ============================================================

def normalize(value):
    """Vereinheitlicht Großschreibung und Leerzeichen."""
    text = str(value or "").upper().strip()
    return re.sub(r"\s+", " ", text)


def call_model(user_content):
    """
    Ruft unser Azure-Modell über das Python-SDK auf.

    user_content kann entweder ein normaler Text
    oder eine multimodale Nachricht mit Bild sein.
    """

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=350,
    )

    duration = time.perf_counter() - start

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Leere Modellantwort")

    prediction = json.loads(content)

    tokens = response.usage.total_tokens if response.usage else 0

    return prediction, duration, tokens


def image_to_data_url(image):
    """
    Wandelt ein PIL-Bild in einen Base64-String um.

    Damit können wir das Bild direkt innerhalb
    unserer API-Anfrage an Azure übertragen.
    """

    buffer = io.BytesIO()

    # PNG ist ein verlustfreies Bildformat.
    image.convert("RGB").save(buffer, format="PNG")

    image_bytes = buffer.getvalue()

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


# ============================================================
# 3. SROIE-DATENSATZ LADEN
# ============================================================

# Für Entwicklungsexperimente verwenden wir train.
# Den Test-Split heben wir für die finale Evaluation auf.
dataset = load_dataset(
    "jsdnrs/ICDAR2019-SROIE",
    split="train",
    streaming=True,
)


# ============================================================
# 4. DIE BEIDEN VARIANTEN VERGLEICHEN
# ============================================================

for index, example in enumerate(
    dataset.take(NUM_EXAMPLES),
    start=1,
):
    print(f"\n========== BELEG {index} ==========")

    ground_truth = example["entities"]

    # ----------------------------------------
    # VARIANTE A: VORHANDENER OCR-TEXT
    # ----------------------------------------

    ocr_text = "\n".join(example["words"])

    text_prediction, text_time, text_tokens = call_model(
        ocr_text
    )

    # ----------------------------------------
    # VARIANTE B: ORIGINALBILD
    # ----------------------------------------

    image = example["image"]

    image_url = image_to_data_url(image)

    # Eine multimodale Nachricht enthält
    # sowohl eine Textanweisung als auch ein Bild.
    vision_content = [
        {
            "type": "text",
            "text": "Extract the fields from this receipt image.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "high",
            },
        },
    ]

    vision_prediction, vision_time, vision_tokens = call_model(
        vision_content
    )

    # ----------------------------------------
    # ERGEBNISSE AUSGEBEN
    # ----------------------------------------

    print("\nGround Truth:")
    print(ground_truth)

    print("\nOCR-Text-Variante:")
    print(text_prediction)

    print("\nVision-Variante:")
    print(vision_prediction)

    print("\nFeldvergleich:")

    for field in FIELDS:
        expected = normalize(ground_truth.get(field))
        text_value = normalize(text_prediction.get(field))
        vision_value = normalize(vision_prediction.get(field))

        print(
            f"{field:10} | "
            f"Text: {'OK' if text_value == expected else 'FEHLER':6} | "
            f"Vision: {'OK' if vision_value == expected else 'FEHLER'}"
        )

    print(
        f"\nText:   {text_time:.2f} s, {text_tokens} Tokens"
    )
    print(
        f"Vision: {vision_time:.2f} s, {vision_tokens} Tokens"
    )
