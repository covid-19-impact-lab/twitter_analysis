import spacy
import spacy_cld
import numpy as np
import pandas as pd


URL_REGEX = (
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def replace_urls(df, token="[URL]"):
    """Replace urls in tweets with token.

    We replace urls in the tweets by default with the token `[URL]` and do the following
    two steps.

    1. Use the urls scraped with the tweets and replace them.
    2. Use a generic regex to remove other urls

    The url regex is taken from http://www.urlregex.com/.

    """
    url_not_na = df.urls.notna()
    df.loc[url_not_na, "text"] = df.loc[url_not_na].apply(
        lambda x: x.text.replace(x.urls, token), axis=1
    )

    df.text = df.text.str.replace(URL_REGEX, token)

    return df


def replace_mentions(df, token="[MNT]"):
    """Replace mentions in tweets with token.

    Use the mentions scraped with the tweet and replace them.

    After this process, there are still @<whatever> expressions in the tweets which look
    like the users wanted to use hashtags instead. But, only for less than 3% of tweets.

    """

    def _replace_mentions_row(row):
        for mention in row.mentions.split(" "):
            row.text = row.text.replace(mention, token)
        return row

    mention_not_na = df.mentions.notna()
    df.loc[mention_not_na, "text"] = df.loc[mention_not_na].apply(
        lambda x: _replace_mentions_row(x), axis=1
    )

    return df


def replace_hashtags(df, token="[HT]"):
    """Replace hashtags in tweets with token."""

    def _replace_hashtags_row(row):
        if isinstance(row.hashtags, str):
            for hashtag in row.hashtags.split(" "):
                row.text = row.text.replace(hashtag, token)

        return row

    hashtag_not_na = df.hashtags.notna()
    df.loc[hashtag_not_na, "text"] = df.loc[hashtag_not_na].apply(
        lambda x: _replace_hashtags_row(x), axis=1
    )

    return df


def remove_link_only_tweets(df, token="[URL]"):
    """Remove link only tweets.

    There are some tweets which seem only to provide a link which Twitter renders into a
    preview plus the link. We remove these tweets (which are less than 2%) as they are
    not really carrying the sentiment of the twitter user, but that of the article.

    """
    df = df.loc[~df.text.str.endswith(f"... {token}")]
    return df


def detect_language(text, languages=None):
    languages = ["de", "en"] if languages is None else languages
    assert isinstance(languages, list), "'languages' must be a list."

    nlp_cld = spacy.load("en", disable_pipes=["tagger", "ner"])
    language_detector = spacy_cld.LanguageDetector()
    nlp_cld.add_pipe(language_detector)

    def _detect_language(text, nlp, languages):
        try:
            doc = nlp(text)
            language_scores = {
                lang: doc._.language_scores.get(lang, 0) for lang in languages
            }
        except Exception:
            language_scores = {lang: np.nan for lang in languages}

        return tuple(language_scores.values())

    scores = text.apply(_detect_language, nlp=nlp_cld, languages=languages)
    scores = pd.DataFrame(scores.tolist(), columns=["de", "en"], index=scores.index)

    scores["other"] = 1 - scores[languages].sum(axis=1)

    return scores
