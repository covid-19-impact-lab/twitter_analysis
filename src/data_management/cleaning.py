import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
import numpy as np

from bld.project_paths import project_paths_join as ppj


UNNECESSARY_COLUMNS = ["formatted_date", "geo"]


def load_data():
    paths = list(Path(ppj("IN_DATA", "corona_data")).glob("**/*.parquet"))

    dfs = []
    for path in paths:
        table = pq.read_table(path)
        df = table.to_pandas()
        dfs.append(df)

    df = pd.concat(dfs, sort=False)

    return df


def minimal_preprocessing(df):
    replace_to = {None: np.nan, "": np.nan}

    df = df.replace(replace_to)

    df = df.drop_duplicates(subset="id")

    df = df.drop(columns=UNNECESSARY_COLUMNS)

    df.id = df.id.astype(np.uint64)
    df = df.set_index("id")

    return df


def main():
    df = load_data()
    df = minimal_preprocessing(df)

    df.to_pickle(ppj("OUT_DATA", "data_clean.pkl"))


if __name__ == "__main__":
    main()
