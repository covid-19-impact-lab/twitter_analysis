import matplotlib.pyplot as plt
import pandas as pd

from bld.project_paths import project_paths_join as ppj


def extract_sources():
    df = pd.read_pickle(ppj("OUT_DATA", "data_clean.pkl"))

    sources = (
        df.urls.str.split("/", n=3, expand=True)[2]
        .str.replace("www.", "")
        .value_counts()
    )

    return sources


def plot(sources):
    fig, ax = plt.subplots(figsize=(12, 8))

    sources.head(30).plot.bar(ax=ax)

    plt.tight_layout()

    plt.savefig(ppj("OUT_FIGURES", "fig-favorite-sources.png"))


if __name__ == "__main__":
    sources = extract_sources()
    plot(sources)
