
import json
import re
from pathlib import Path


# ============================================================
# 1. GESPEICHERTE EVALUATIONSERGEBNISSE LADEN
# ============================================================

# Wir verwenden die Ergebnisse des letzten Experiments.
# Es werden KEINE neuen Azure-API-Anfragen gesendet.
input_file = Path("results/sroie_predictions.json")

with open(input_file, "r", encoding="utf-8") as file:
    results = json.load(file)


# ============================================================
# 2. ALLGEMEINE TEXTNORMALISIERUNG
# ============================================================

def normalize_text(value):
    """
    Macht Textvergleiche weniger empfindlich gegenüber
    Groß-/Kleinschreibung und mehrfachen Leerzeichen.
    """
    text = str(value or "").upper().strip()
    return re.sub(r"\s+", " ", text)


# ============================================================
# 3. SPEZIELLE NORMALISIERUNG FÜR BETRÄGE
# ============================================================

def normalize_amount(value):
    """
    Entfernt das Währungskürzel RM und Leerzeichen.

    Beispiel:
        "RM 278.80" -> "278.80"

    Dies ist zunächst eine einfache, datensatzspezifische
    Regel und noch kein allgemeiner Währungsparser.
    """
    text = normalize_text(value)
    text = re.sub(r"\bRM\b", "", text)
    return text.strip()


# ============================================================
# 4. SPEZIELLE NORMALISIERUNG FÜR DATUM
# ============================================================

def normalize_date(value):
    """
    Extrahiert ein Datum im Format TT/MM/JJJJ.

    Beispiel:
        "15/01/2019 11:05:16 AM" -> "15/01/2019"

    Andere Datumsformate unterstützen wir später.
    """
    text = normalize_text(value)
    match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)

    return match.group(0) if match else text


# ============================================================
# 5. ORIGINALSCORE UND NORMALISIERTEN SCORE VERGLEICHEN
# ============================================================

fields = ["company", "date", "address", "total"]

normalizers = {
    "company": normalize_text,
    "date": normalize_date,
    "address": normalize_text,
    "total": normalize_amount,
}

original_correct = {field: 0 for field in fields}
normalized_correct = {field: 0 for field in fields}

for result in results:
    expected = result["expected"] if "expected" in result else result["ground_truth"]
    predicted = result.get("predicted", result.get("prediction"))

    if not isinstance(predicted, dict):
        continue

    for field in fields:
        ground_truth = expected.get(field, "")
        prediction = predicted.get(field, "")

        # Vergleich wie in unserer ursprünglichen Evaluation.
        original_match = (
            normalize_text(ground_truth)
            == normalize_text(prediction)
        )

        # Vergleich mit einer passenden Regel pro Feld.
        normalize = normalizers[field]

        normalized_match = (
            normalize(ground_truth)
            == normalize(prediction)
        )

        original_correct[field] += int(original_match)
        normalized_correct[field] += int(normalized_match)

        if not original_match:
            print(f"\nDokument {result['document_id']} – {field}")
            print("Ground Truth:", ground_truth)
            print("Vorhersage:  ", prediction)

            if normalized_match:
                print("Fehlerart: FORMATUNTERSCHIED")
            else:
                print("Fehlerart: INHALTLICHE ABWEICHUNG ODER "
                      "NICHT ABGEDECKTES FORMAT")


# ============================================================
# 6. SCORES AUSGEBEN
# ============================================================

print("\n--- SCORE-VERGLEICH ---")

total = len(results)

for field in fields:
    original_score = original_correct[field] / total
    normalized_score = normalized_correct[field] / total

    print(
        f"{field:10} | "
        f"Original: {original_score:.1%} | "
        f"Normalisiert: {normalized_score:.1%}"
    )
