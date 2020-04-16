import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj
from src.shared import read_parquet_in_date_chunks


def extract_sources():
    df = read_parquet_in_date_chunks(ppj("OUT_DATA", "tweets-cleaned"))

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
