import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.pre_processing.preprocessors import remove_link_only_tweets
from src.pre_processing.preprocessors import replace_hashtags
from src.pre_processing.preprocessors import replace_mentions
from src.pre_processing.preprocessors import replace_urls


def main():
    """Select eligible tweets for labeling.

    The main criteria for tweets to be passed to the annotators are

    1. The tweet must be in English or German.
    2. The tweet should contain at least 50 characters.
    3. The tweet should not respond to someone.

    We do the following before we select the tweets.

    - Remove urls, mentions, hashtags, strip whitespace from the tweet to get the
      corrected length of the tweet.

    """
    df = pd.read_pickle(ppj("OUT_DATA", "data_clean.pkl"))

    df["original_text"] = df.text

    # Replace urls with token, remove link only tweets, remove the token.
    df = replace_urls(df)
    df = remove_link_only_tweets(df)
    df.text = df.text.str.replace("[URL]", "")

    df = replace_mentions(df, "")
    df = replace_hashtags(df, "")

    # Strip whitespace.
    df.text = df.text.str.strip()

    sc = pd.read_pickle(ppj("OUT_DATA", "detected_languages.pkl"))

    # Only keep tweets which are with at least 90% chance either English or German.
    # Select with index.
    at_least_90_sure_de = sc.query("de >= 0.9").index
    de = df.loc[at_least_90_sure_de.intersection(df.index)]
    de = de.loc[de.to.isna()]

    # To save space.
    de = de[["original_text", "permalink"]].sample(n=1_000, random_state=0)
    de.to_csv(ppj("OUT_DATA", "selected_tweets_for_labeling_de.csv"))

    # Only keep tweets which are with at least 90% chance either English or German.
    # Select with index.
    at_least_90_sure_en = sc.query("en >= 0.9").index
    en = df.loc[at_least_90_sure_en.intersection(df.index)]
    en = en.loc[en.to.isna()]

    # To save space.
    en = en[["original_text", "permalink"]].sample(n=1_000, random_state=0)
    en.to_csv(ppj("OUT_DATA", "selected_tweets_for_labeling_en.csv"))


if __name__ == "__main__":
    main()
