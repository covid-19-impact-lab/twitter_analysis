from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bld.project_paths import project_paths_join as ppj
from src.shared import to_parquet_in_date_chunks

UNNECESSARY_COLUMNS = ["formatted_date", "geo"]


def load_data():
    paths = list(Path(ppj("IN_DATA", "corona_data")).glob("**/*.parquet"))

    dfs = []
    for path in paths:
        table = pq.read_table(path)
        df = table.to_pandas()

        # Add state and city from path.
        df["state"] = path.parents[3].name
        df["city"] = path.parents[2].name

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

    to_parquet_in_date_chunks(df, ppj("OUT_DATA", "tweets-cleaned"))


if __name__ == "__main__":
    main()
