"""
Mit diesem Skript werde alle Filmtitel aus allen CSV-Dateien aus dem Verzeichnis "data" extrahiert, deren Titel aus
genau einem Term besteht. Die CSV-Dateien im Verzeichnis "Einworttiel" enthalten nur noch Titel, auf welche die
Bedingung zutrifft.
"""

import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    # Alle CSV-Dateien im Verzeichnis "data" einlesen.
    tables = {}
    sorted_names = sorted(os.listdir("data"))

    for file in sorted_names:

        with open(os.path.join("data", file), "r", encoding="utf-8") as _:
            tables[file] = pd.read_csv(os.path.join("data", file), delimiter=",")

    filtered_tables = []

    # Alle Titel der CSV-Tabellen auf ihre Länge in Termen prüfen und alle Einworttitel behalten
    for table in tables.values():

        table["Titellänge"] = np.nan

        for i, row in table.iterrows():

            titellänge = len(table.loc[i, "Titel"].split())

            if titellänge > 1:
                table.drop(i, inplace=True)

        filtered_tables.append(table)

    tables = dict(zip(sorted_names, filtered_tables))

    # Die Einworttitel aller Tabellen in neue CSV-Dateien im Verzeichnis "Einworttitel" schreiben.
    for name, table in tables.items():
        table.to_csv(path_or_buf=os.path.join("Einworttitel", name))
