
import json
import os

import requests
import streamlit as st


# ============================================================
# 1. KONFIGURATION
# ============================================================

# Adresse unseres FastAPI-Backends.
# Über eine Umgebungsvariable können wir diese Adresse
# später beispielsweise für Docker ändern.
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/extract",
)

# Allgemeine Einstellungen der Webseite.
st.set_page_config(
    page_title="Intelligent Document Pipeline",
    page_icon="📄",
    layout="centered",
)


# ============================================================
# 2. TITEL UND BESCHREIBUNG
# ============================================================

st.title("📄 Intelligent Document Pipeline")

st.markdown(
    "AI-powered receipt information extraction "
    "with Azure OpenAI, FastAPI and Pydantic."
)

st.divider()


# ============================================================
# 3. BEISPIELTEXT
# ============================================================

# Damit wir nicht bei jedem Test einen Beleg
# manuell eingeben müssen.
sample_text = """OJC MARKETING SDN BHD
NO 2 & 4, JALAN BAYU 4
BANDAR SERI ALAM
81750 MASAI, JOHOR

DATE: 15/01/2019
TOTAL: RM 193.00"""


# ============================================================
# 4. BENUTZEREINGABE
# ============================================================

# text_area erzeugt ein mehrzeiliges Eingabefeld.
# Der eingegebene Text wird in receipt_text gespeichert.
receipt_text = st.text_area(
    label="Receipt text",
    value=sample_text,
    height=220,
    help="Paste the OCR text of a receipt here.",
)


# ============================================================
# 5. SESSION STATE
# ============================================================

# Streamlit führt das Skript bei Benutzerinteraktionen
# normalerweise erneut von oben nach unten aus.
#
# Session State ermöglicht es uns, Ergebnisse zwischen
# diesen erneuten Ausführungen zu speichern.

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# 6. BUTTON UND API-AUFRUF
# ============================================================

if st.button("Extract information", type="primary"):

    # Alte Ergebnisse entfernen, wenn eine neue
    # Extraktion gestartet wird.
    st.session_state.result = None

    # Prüfen, ob überhaupt Text eingegeben wurde.
    if not receipt_text.strip():
        st.warning("Please enter a receipt text.")

    else:

        # Zeigt während des API-Aufrufs einen Ladeindikator.
        with st.spinner("Extracting receipt information..."):

            try:

                # HTTP-POST-Request an unser FastAPI-Backend.
                # requests wandelt das Dictionary automatisch
                # in einen JSON-Request-Body um.
                response = requests.post(
                    API_URL,
                    json={"text": receipt_text},
                    timeout=120,
                )

                # Löst bei HTTP-Fehlern eine Exception aus.
                response.raise_for_status()

                # JSON-Antwort in ein Python-Dictionary
                # umwandeln und im Session State speichern.
                st.session_state.result = response.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the backend. "
                    "Please make sure FastAPI is running."
                )

            except requests.exceptions.Timeout:
                st.error("The extraction request timed out.")

            except requests.exceptions.HTTPError as error:
                st.error(f"Backend error: {error}")

            except requests.exceptions.RequestException as error:
                st.error(f"Request failed: {error}")


# ============================================================
# 7. ERGEBNISSE DARSTELLEN
# ============================================================

# Die Ergebnisse werden nur angezeigt,
# wenn eine erfolgreiche Extraktion vorliegt.
if st.session_state.result is not None:

    result = st.session_state.result

    st.divider()

    st.success("Extraction successful!")

    st.subheader("Extracted information")

    # Erstellt zwei nebeneinanderliegende Spalten.
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Company**")
        st.write(result["company"])

        st.markdown("**Address**")
        st.write(result["address"])

    with col2:
        st.markdown("**Date**")
        st.write(result["date"])

        st.metric(
            label="Total",
            value=result["total"],
        )

    # Aufklappbarer Bereich für die vollständige JSON-Antwort.
    with st.expander("View raw JSON"):
        st.json(result)

    # Ermöglicht den Download des Extraktionsergebnisses.
    st.download_button(
        label="Download JSON",
        data=json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="extracted_receipt.json",
        mime="application/json",
    )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by Azure GPT-4.1 mini | "
    "FastAPI | Pydantic | Streamlit"
)
