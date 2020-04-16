import sys
from pathlib import Path

import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.pre_processing.preprocessors import remove_link_only_tweets
from src.pre_processing.preprocessors import replace_hashtags
from src.pre_processing.preprocessors import replace_mentions
from src.pre_processing.preprocessors import replace_urls
from src.pre_processing.translate import apply_to_batch_of_series
from src.pre_processing.translate import translate_text_with_deepl


def main(timestamp):
    path = Path(ppj("OUT_DATA", "translated-tweets", f"{timestamp}.parquet"))

    if not path.exists():
        df = pd.read_parquet(ppj("OUT_DATA", "tweets-cleaned", f"{timestamp}.parquet"))

        df["original_text"] = df.text

        # Replace urls with token, remove link only tweets, remove the token.
        df = replace_urls(df)
        df = remove_link_only_tweets(df)
        df.text = df.text.str.replace("[URL]", "")

        df = replace_mentions(df, "")
        df = replace_hashtags(df, "")

        translation = apply_to_batch_of_series(
            df.text, translate_text_with_deepl, batch_size=50
        )

        df = df[["original_text", "text"]].join(translation)
        df.to_parquet(path)


if __name__ == "__main__":
    timestamp = sys.argv[1]
    main(timestamp)
