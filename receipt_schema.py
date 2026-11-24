
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator


class Receipt(BaseModel):
    """
    Datenmodell für einen extrahierten Kassenbeleg.

    Pydantic prüft beim Erstellen eines Receipt-Objekts,
    ob alle Pflichtfelder vorhanden sind und ob
    die Werte unseren Regeln entsprechen.
    """

    # Keine zusätzlichen, unbekannten Felder erlauben.
    model_config = ConfigDict(extra="forbid")

    company: str
    date: str
    address: str
    total: Decimal

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        """Prüft das Format TT/MM/JJJJ und das Datum selbst."""

        datetime.strptime(value, "%d/%m/%Y")
        return value

    @field_validator("total")
    @classmethod
    def validate_total(cls, value: Decimal) -> Decimal:
        """Ein negativer Gesamtbetrag ist hier nicht erlaubt."""

        if value < 0:
            raise ValueError("Der Betrag darf nicht negativ sein.")

        return value
