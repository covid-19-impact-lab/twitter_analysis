from bld.project_paths import project_paths_join as ppj
from src.pre_processing.preprocessors import remove_link_only_tweets
from src.pre_processing.preprocessors import replace_hashtags
from src.pre_processing.preprocessors import replace_mentions
from src.pre_processing.preprocessors import replace_urls
from src.shared import read_parquet_in_date_chunks


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
    df = read_parquet_in_date_chunks(ppj("OUT_DATA", "tweets-cleaned"))

    df["original_text"] = df.text

    # Replace urls with token, remove link only tweets, remove the token.
    df = replace_urls(df)
    df = remove_link_only_tweets(df)
    df.text = df.text.str.replace("[URL]", "")

    df = replace_mentions(df, "")
    df = replace_hashtags(df, "")

    # Strip whitespace.
    df.text = df.text.str.strip()

    sc = read_parquet_in_date_chunks(ppj("OUT_DATA", "detected-languages"))

    # Only keep tweets which are with at least 90% chance either English or German, and
    # are not a responds to another tweet. Select with index.
    at_least_90_sure = sc.query("de >= 0.9 or en >= 0.9").index
    df = df.loc[at_least_90_sure.intersection(df.index)]
    df = df.loc[df.to.isna()]

    # To save space.
    df = df[["original_text", "permalink"]].sample(n=1_000, random_state=0)
    df.to_csv(ppj("OUT_DATA", "selected_tweets_for_labeling.csv"))


if __name__ == "__main__":
    main()
