import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj

if __name__ == "__main__":

    colnames = ["id", "sentiment", "key", "foo", "text"]
    df = pd.read_csv(
        ppj("IN_DATA", "training_data/data.tsv"), sep="\t", header=None, names=colnames
    )

    df = df.replace(to_replace={"Not Available": np.nan})
    df = df.dropna()
    df = df.drop_duplicates()

    df_out = df[["sentiment", "text"]]
    df_out.to_csv(ppj("OUT_DATA", "data_clean.csv"), index=False)
