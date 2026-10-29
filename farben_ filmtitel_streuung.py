"""
Dieses Skript berechnet und visualisiert die Streuung der Farbennennungen pro Jahr in den CSV‑Dateien im Verzeichnis
"data". Zu diesem Zweck werden die Terme der Filmtitel vor ihrer Zählung  mittels des Spacy-Modells 'de_core_news_sm'
lemmatisiert. Komposita wurden hier nicht berücksichtigt. Die berechneten Werte werden im Anschluss als Balkendiagramm
visualisiert. Die zugrundeliegenden Werte werden außerdem unter "Farbhäufigkeiten" serialisiert. Jede Zeile besitzt
hier eine Zelle mit dem folgenden Vektor: {'rot': a, 'blau': b, 'grün': c, 'gelb': x, 'schwarz': y, 'weiß': z, 'gold': m}.

HINWEIS: Dieses Skript wurde KI-gestützt erstellt. Der Programmcode wurde im Detail geprüft.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    filtered_tables = []

    # ─────────────────────────────────────
    # Zu prüfende Farben
    # ─────────────────────────────────────

    colours = ["rot", "blau", "grün", "gelb", "schwarz", "weiß", "gold"]

    normalized_color_records = {}

    # ─────────────────────────────────────
    # Tabellen analysieren
    # ─────────────────────────────────────

    for name, table in tables.items():

        current_table = {colour: 0 for colour in colours}

        farbhäufigkeiten = []

        # Titel vorbereiten
        titel = table["Titel"].fillna("").astype(str).tolist()

        # Alle Titel mit spaCy verarbeiten
        docs = nlp.pipe(titel)

        for doc in docs:

            # Nur vollständige Tokens verwenden.
            #
            # Durch die Lemmatisierung werden z. B.
            # "rote", "roten", "rotem" zu "rot".
            #
            # "Pierrot" bleibt dagegen "Pierrot" und
            # wird deshalb nicht als "rot" erkannt.
            #
            # "goldene", "goldenen", "goldener" werden
            # entsprechend auf "gold" zurückgeführt.

            lemmata = [token.lemma_.lower() for token in doc if not token.is_space]

            current_row = {}

            for colour in colours:

                count = lemmata.count(colour)

                current_row[colour] = count
                current_table[colour] += count

            farbhäufigkeiten.append(current_row)

        # Farbhäufigkeiten in der Tabelle speichern
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

        filtered_tables.append(table)

    # ─────────────────────────────────────
    # Tabellen wieder mit Dateinamen
    # verknüpfen
    # ─────────────────────────────────────

    csv_names = [name for name in sorted_names if name.lower().endswith(".csv")]

    tables = dict(zip(csv_names, filtered_tables))

    # ─────────────────────────────────────
    # Farbdiagramm
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

    fig, ax = plt.subplots(figsize=(10, 6))

    n_farben = len(farben)
    breite = 0.12

    for i, gruppe in enumerate(gruppen):

        for j, farbe in enumerate(farben):

            x = i + (j - (n_farben - 1) / 2) * breite

            y = normalized_color_records[gruppe][farbe]

            ax.scatter(
                x,
                y,
                color=plot_farben[farbe],
                edgecolor="black",
                s=70,
                zorder=3,
            )

    # ─────────────────────────────────────
    # Trennlinien zwischen den Gruppen
    # ─────────────────────────────────────

    for i in range(len(gruppen) - 1):

        ax.axvline(
            i + 0.5,
            linestyle="--",
            linewidth=1,
            alpha=0.4,
            zorder=1,
        )

    # ─────────────────────────────────────
    # Achsen
    # ─────────────────────────────────────

    ax.set_xticks(range(len(gruppen)))

    ax.set_xticklabels(gruppen)

    ax.set_xlabel("Jahr")
    ax.set_ylabel("Normalisierter Wert: Farbe pro Titel")

    ax.set_title("Normalisierte Farbwerte nach Jahren")

    # Horizontales Grid
    ax.grid(
        axis="y",
        alpha=0.3,
        zorder=0,
    )

    # ─────────────────────────────────────
    # Legende
    # ─────────────────────────────────────

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=farbe,
            markerfacecolor=plot_farben[farbe],
            markeredgecolor="black",
            markersize=8,
        )
        for farbe in farben
    ]

    ax.legend(
        handles=legend_elements,
        title="Farbe",
    )

    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────
    # Balkendiagramm:
    # Anzahl der Einträge pro Tabelle
    # ─────────────────────────────────────

    table_names = list(tables.keys())

    entry_counts = [len(table) for table in tables.values()]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(table_names, entry_counts)

    # Anzahl der Einträge über den Balken
    for bar, count in zip(bars, entry_counts):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Tabelle")
    ax.set_ylabel("Anzahl der Filme")

    ax.set_title("Anzahl der Filme pro Tabelle")

    # Labels lesbar machen
    plt.xticks(rotation=45, ha="right")

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────
    # CSV-Export
    # ─────────────────────────────────────

    for name, table in tables.items():

        table.to_csv(path_or_buf=os.path.join("Farbhäufigkeiten", name), index=False)
