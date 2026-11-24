
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from extract_receipt import extract_receipt
from receipt_schema import Receipt


# ============================================================
# 1. FASTAPI-ANWENDUNG ERSTELLEN
# ============================================================

app = FastAPI(
    title="Intelligent Document Pipeline",
    description="Extrahiert strukturierte Daten aus Belegtexten mit Azure OpenAI.",
    version="0.1.0",
)


# ============================================================
# 2. SCHEMA FÜR DIE EINGABE DEFINIEREN
# ============================================================

class ExtractRequest(BaseModel):
    """
    Beschreibt den JSON-Body, den unser API-Endpoint erwartet.

    Beispiel:
    {
        "text": "SHOP ABC ... TOTAL 25.50"
    }
    """

    text: str = Field(
        min_length=1,
        description="OCR-Text eines Kassenbelegs",
    )


# ============================================================
# 3. EINFACHER TEST-ENDPOINT
# ============================================================

@app.get("/")
def home():
    """
    GET / zeigt an, dass unser Webservice erreichbar ist.
    Hier wird noch kein LLM aufgerufen.
    """

    return {
        "message": "Intelligent Document Pipeline API is running"
    }


# ============================================================
# 4. ENDPOINT FÜR DIE BELEGEXTRAKTION
# ============================================================

@app.post("/extract", response_model=Receipt)
def extract(request: ExtractRequest):
    """
    POST /extract

    1. FastAPI validiert den eingehenden Request.
    2. Unsere Funktion ruft Azure OpenAI auf.
    3. Pydantic validiert die LLM-Antwort.
    4. FastAPI gibt das Ergebnis als JSON zurück.

    response_model=Receipt beschreibt das Ausgabeformat
    und sorgt für eine zusätzliche Prüfung der Antwort.
    """

    try:
        receipt = extract_receipt(request.text)

        return receipt

    except ValidationError as error:
        # Die Antwort des LLM entspricht nicht unserem Schema.
        raise HTTPException(
            status_code=502,
            detail="Die LLM-Antwort konnte nicht validiert werden.",
        ) from error

    except ValueError as error:
        # Zum Beispiel eine leere LLM-Antwort.
        raise HTTPException(
            status_code=502,
            detail="Die LLM-Antwort konnte nicht verarbeitet werden.",
        ) from error

