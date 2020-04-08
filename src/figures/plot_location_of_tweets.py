import matplotlib.pyplot as plt
import pandas as pd

from bld.project_paths import project_paths_join as ppj


def plot_states(df):
    fig, ax = plt.subplots(figsize=(12, 8))

    df.state.value_counts().plot.bar(ax=ax)

    plt.tight_layout()

    plt.savefig(ppj("OUT_FIGURES", "fig-location-states.png"))


def plot_cities(df):
    fig, ax = plt.subplots(figsize=(12, 8))

    df.city.value_counts().head(30).plot.bar(ax=ax)

    plt.tight_layout()

    plt.savefig(ppj("OUT_FIGURES", "fig-location-cities.png"))


if __name__ == "__main__":
    df = pd.read_pickle(ppj("OUT_DATA", "data_clean.pkl"))

    plot_states(df)

    plot_cities(df)
