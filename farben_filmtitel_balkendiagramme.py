"""
Mit diesem Skript wird die Farbhäufigkeit pro Titel je Farbe berechnet und als gruppiertes Balkendiagramm visualisiert.
Zu diesem Zweck werden die Terme der Filmtitel vor ihrer Zählung  mittels des Spacy-Modells 'de_core_news_sm'
lemmatisiert. Komposita wurden hier nicht berücksichtigt.

HINWEIS: Diese Skript wurde KI-gestützt erstellt. Die Programmcode wurden im Detail geprüft.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import spacy

if __name__ == "__main__":
    # ─────────────────────────────────────
    # Deutsches spaCy-Modell laden
    # ─────────────────────────────────────

    nlp = spacy.load("de_core_news_sm")

    # ─────────────────────────────────────
    # Ausgabeordner anlegen
    # ─────────────────────────────────────

    # ─────────────────────────────────────
    # CSV-Dateien einlesen
    # ─────────────────────────────────────

    tables = {}

    sorted_names = sorted(os.listdir("data"))

    for file in sorted_names:

        file_path = os.path.join("data", file)

        # Nur CSV-Dateien berücksichtigen
        if not file.lower().endswith(".csv"):
            continue

        tables[file] = pd.read_csv(file_path, delimiter=",")

    # ─────────────────────────────────────
    # Farben definieren
    # ─────────────────────────────────────

    colours = ["rot", "blau", "grün", "gelb", "schwarz", "weiß", "gold"]

    normalized_color_records = {}
    filtered_tables = []

    # ─────────────────────────────────────
    # Titel analysieren
    # ─────────────────────────────────────

    for name, table in tables.items():

        current_table = {colour: 0 for colour in colours}

        # Speichert für jeden Titel die gefundenen Farbhäufigkeiten
        farbhäufigkeiten = []

        # Alle Titel auf einmal durch spaCy verarbeiten
        titel = table["Titel"].fillna("").astype(str).tolist()

        docs = nlp.pipe(titel)

        for doc in docs:

            # Nur vollständige Tokens verwenden.
            #
            # Durch die Verwendung des Lemmas werden z. B.
            # "rote", "roten", "rotem" usw. zu "rot".
            #
            # "Pierrot" bleibt dagegen "Pierrot" bzw. "pierrot"
            # und wird deshalb NICHT als "rot" erkannt.
            lemmata = [token.lemma_.lower() for token in doc if not token.is_space]

            current_row = {}

            for colour in colours:

                count = lemmata.count(colour)

                current_row[colour] = count
                current_table[colour] += count

            farbhäufigkeiten.append(current_row)

        # Farbhäufigkeiten in die Tabelle schreiben
        table["Farbhäufigkeit"] = [str(frequency) for frequency in farbhäufigkeiten]

        # ─────────────────────────────────
        # Normalisierung
        # ─────────────────────────────────

        if len(table) > 0:

            current_table_normalized = {
                colour: count / len(table) for colour, count in current_table.items()
            }

        else:

            current_table_normalized = {colour: 0 for colour in colours}

        normalized_color_records[name] = current_table_normalized

        # ─────────────────────────────────
        # Nur Titel mit mindestens einer
        # gefundenen Farbe behalten
        # ─────────────────────────────────

        filtered_table = table[
            [
                any(frequency > 0 for frequency in row.values())
                for row in farbhäufigkeiten
            ]
        ].copy()

        filtered_tables.append(filtered_table)

    # ─────────────────────────────────────
    # Tabellen wieder mit Dateinamen
    # verknüpfen
    # ─────────────────────────────────────

    csv_names = [name for name in sorted_names if name.lower().endswith(".csv")]

    tables = dict(zip(csv_names, filtered_tables))

    # ─────────────────────────────────────
    # Gruppiertes Balkendiagramm
    # ─────────────────────────────────────

    gruppen = list(normalized_color_records.keys())

    farben = list(next(iter(normalized_color_records.values())).keys())

    plot_farben = {
        "rot": "red",
        "blau": "blue",
        "grün": "green",
        "gelb": "gold",
        "schwarz": "black",
        "weiß": "white",
        "gold": "goldenrod",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    n_farben = len(farben)

    # Gesamtbreite einer Farbgruppe
    gruppenbreite = 0.8

    # Breite eines einzelnen Balkens
    breite = gruppenbreite / n_farben

    x = np.arange(len(gruppen))

    # ─────────────────────────────────────
    # Balken für jede Farbe zeichnen
    # ─────────────────────────────────────

    for j, farbe in enumerate(farben):

        werte = [normalized_color_records[gruppe][farbe] for gruppe in gruppen]

        position = x + (j - (n_farben - 1) / 2) * breite

        ax.bar(
            position,
            werte,
            width=breite,
            label=farbe,
            color=plot_farben[farbe],
            edgecolor="black",
            linewidth=0.8,
        )

    # ─────────────────────────────────────
    # Trennlinien zwischen den Gruppen
    # ─────────────────────────────────────

    for i in range(len(gruppen) - 1):

        ax.axvline(i + 0.5, linestyle="--", linewidth=1, alpha=0.4, zorder=0)

    # ─────────────────────────────────────
    # Achsen
    # ─────────────────────────────────────

    ax.set_xticks(x)
    ax.set_xticklabels(gruppen)

    ax.set_xlabel("Jahr")
    ax.set_ylabel("Normalisierter Wert: Farbe pro Titel")

    ax.set_title("Normalisierte Farbwerte nach Jahren")

    # Horizontales Grid
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Legende
    ax.legend(title="Farbe")

    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────
    # CSV-Export
    # ─────────────────────────────────────

    for name, table in tables.items():

        output_path = os.path.join("Titel_mit_Farben", name)

        table.to_csv(path_or_buf=output_path, index=False)
