from pathlib import Path

import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.pre_processing.preprocessors import detect_language
from src.pre_processing.preprocessors import remove_link_only_tweets
from src.pre_processing.preprocessors import replace_hashtags
from src.pre_processing.preprocessors import replace_mentions
from src.pre_processing.preprocessors import replace_urls


def main():
    if not Path(ppj("OUT_DATA", "detected_languages.pkl")).exists():
        df = pd.read_pickle(ppj("OUT_DATA", "data_clean.pkl"))

        # Replace urls with token, remove link only tweets, remove the token.
        df = replace_urls(df)
        df = remove_link_only_tweets(df)
        df.text = df.text.str.replace("[URL]", "")

        df = replace_mentions(df, "")
        df = replace_hashtags(df, "")

        scores = detect_language(df.text)
        scores.to_pickle(ppj("OUT_DATA", "detected_languages.pkl"))


if __name__ == "__main__":
    main()
