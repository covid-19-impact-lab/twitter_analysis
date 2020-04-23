from pathlib import Path

import dask.dataframe as dd
import pandas as pd

TIMESTAMPS = [
    date.strftime("%Y-%m-%d")
    for date in pd.date_range(start="2020-03-10", end="2020-04-14")
]


def to_parquet_in_date_chunks(df, directory):
    """Save a DataFrame to parquet by splitting the data by dates.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame which is stored.
    directory : str
        Directory in which the data is stored, e.g. ``ppj("OUT_DATA",
        "tweets-cleaned")``.

    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    for datetime, sub in df.groupby(df.date.dt.date):
        date_str = datetime.isoformat()
        sub.to_parquet(path / f"{date_str}.parquet")


def read_parquet_in_date_chunks(directory):
    """Read a DataFrame to parquet separated by dates.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame which is stored.
    directory : str
        Directory in which the data is stored, e.g. ``ppj("OUT_DATA",
        "tweets-cleaned")``.

    """
    df = dd.read_parquet(directory)
    return df.compute()
