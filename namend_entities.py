"""
Mit diesem Skript werden alle Filmtitel aus allen CSV-Dateien aus dem Verzeichnis "data" extrahiert, die mindestens eine
Named Entity enthalten. Für die NER wird das Modell 'flair/ner-german-large' genutzt. Die CSV-Dateien im Verzeichnis
"NamedEntities" enthalten nur noch Titel, auf welche die Bedingung zutrifft.
"""

import os
import pandas as pd
from tqdm import tqdm

from flair.data import Sentence
from flair.models import SequenceTagger


def contains_named_entity(title: str, tagger) -> bool:
    """
    Prüft, ob ein Titel mindestens eine Named Entity enthält.

    Args:
        title: Der zu prüfende Titel als String.
        tagger: flair.model.SeqeunceTagger

    Returns: True, wenn der Titel mindestens eine Named Entity enthält.
    """

    if pd.isna(title):
        return False

    sentence = Sentence(title)

    tagger.predict(sentence)

    entities = sentence.get_spans("ner")

    return len(entities) > 0


if __name__ == "__main__":

    os.makedirs("NamedEntities", exist_ok=True)

    # Dateien sortieren
    files = sorted(os.listdir("data"))

    # NER-Modell laden
    tagger = SequenceTagger.load("flair/ner-german-large")

    # Alle Filmtitel (Dateinamen) auf Named Entities prüfen
    for file in files:

        path = os.path.join("data", file)

        table = pd.read_csv(path, delimiter=",")

        # Ergebnisliste
        contains_entity = []

        # Fortschrittsanzeige
        for title in tqdm(table["Titel"], desc="Analysiere Titel", unit="Titel"):
            contains_entity.append(contains_named_entity(title, tagger))

        table["NamedEntity"] = contains_entity

        # Nur Titel mit mindestens einer Named Entitiy behalten
        filtered_table = table[table["NamedEntity"]]

        # Index zurücksetzen
        filtered_table = filtered_table.reset_index(drop=True)

        # Gefilterte Tabelle serialisieren
        filtered_table.to_csv(os.path.join("NamedEntities", file), index=False)

        print(
            f"{len(filtered_table)} von {len(table)} Titeln enthalten Named Entities."
        )
