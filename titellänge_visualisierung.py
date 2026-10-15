"""
Mit diesem Skript wird der Mittelwert der Titellänge in Termen pro Jahr visualisiert.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

if __name__ == "__main__":

    tables = []

    sorted_names = sorted(os.listdir("data"))

    print(sorted_names)

    for file in sorted_names:

        with open(os.path.join("data", file), "r", encoding="utf-8") as fp:
            tables.append(pd.read_csv(os.path.join("data", file), delimiter=","))

    for table in tables:

        table["Titellänge"] = np.nan

        for i, row in table.iterrows():
            table.loc[i, "Titellänge"] = len(row["Titel"].split())

    daten = [df["Titellänge"].dropna() for df in tables]
    mittelwerte = [d.mean() for d in daten]

    labels = [title.replace(".csv", "") for title in sorted_names]

    x = np.arange(1, len(daten) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(daten, positions=x)

    ax.plot(x, mittelwerte, color="red", marker="o", linewidth=2, label="Mittelwert")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_xlabel("Jahr")
    ax.set_ylabel("Titellänge")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
