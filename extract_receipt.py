
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from receipt_schema import Receipt


# ============================================================
# 1. AZURE-VERBINDUNG VORBEREITEN
# ============================================================

# Lädt unsere Zugangsdaten aus der .env-Datei.
# Der API-Key steht NICHT direkt im Python-Code.
load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Das OpenAI-Python-SDK verwendet hier unseren Azure-Endpunkt.
# Unser Endpoint enthält bereits /openai/v1.
client = OpenAI(
    base_url=endpoint + "/",
    api_key=api_key,
    timeout=45.0,
    max_retries=1,
)


# ============================================================
# 2. EXTRAKTIONSFUNKTION
# ============================================================

def extract_receipt(text: str) -> Receipt:
    """
    Extrahiert vier Felder aus einem Belegtext.

    Eingabe:
        text: Unstrukturierter Belegtext.

    Ausgabe:
        Ein validiertes Receipt-Objekt.

    Mögliche Fehler:
        - Azure-API nicht erreichbar
        - Ungültiges JSON
        - Pydantic-Validierungsfehler
    """

    # System-Prompt: Legt die Aufgabe und das Ausgabeformat fest.
    system_prompt = (
        "You extract information from receipts. "
        "Return a JSON object with exactly these keys: "
        "company, date, address, total. "
        "All values should be strings. "
        "Date must be DD/MM/YYYY without a time. "
        "Total must contain only the numeric amount, "
        "for example 193.00, without a currency symbol. "
        "Do not invent missing information."
    )

    # --------------------------------------------------------
    # A. AZURE-LLM AUFRUFEN
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=350,
    )

    # Das SDK liefert ein Response-Objekt zurück.
    # Wir lesen den generierten Inhalt der ersten Antwort aus.
    content = response.choices[0].message.content

    if not content:
        raise ValueError("Das LLM hat eine leere Antwort geliefert.")

    # --------------------------------------------------------
    # B. JSON IN PYTHON-DATEN UMWANDELN
    # --------------------------------------------------------

    # json.loads() wandelt z. B. einen JSON-String
    # in ein Python-Dictionary um.
    data = json.loads(content)

    # --------------------------------------------------------
    # C. MIT PYDANTIC VALIDIEREN
    # --------------------------------------------------------

    # Prüft Pflichtfelder, Datentypen und eigene Validatoren.
    # Falls etwas nicht stimmt, wird ein Fehler ausgelöst.
    receipt = Receipt.model_validate(data)

    return receipt


# ============================================================
# 3. KLEINER MANUELLER TEST
# ============================================================

# Dieser Block wird nur ausgeführt, wenn wir
# extract_receipt.py direkt starten.
#
# Beim späteren Import durch FastAPI läuft er NICHT.
if __name__ == "__main__":

    sample_text = """
    OJC MARKETING SDN BHD
    NO 2 & 4, JALAN BAYU 4
    BANDAR SERI ALAM
    81750 MASAI, JOHOR

    DATE: 15/01/2019
    TOTAL: RM 193.00
    """

    try:
        result = extract_receipt(sample_text)

        print("\nValidierte LLM-Antwort:")
        print(result.model_dump_json(indent=2))

    except ValidationError as error:
        print("\nPydantic hat die Antwort abgelehnt:")
        print(error)

    except json.JSONDecodeError as error:
        print("\nDie Modellantwort war kein gültiges JSON:")
        print(error)
