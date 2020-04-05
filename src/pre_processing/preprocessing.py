import json
import re
import string

import nltk
import pandas as pd
import preprocessor as p
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

from bld.project_paths import project_paths_join as ppj


def preprocess_tweet(tweet):

    tweet = re.sub(r":", "", tweet)
    tweet = re.sub(r"‚Ä¶", "", tweet)

    # replace numbers
    tweet = re.sub(r"\d+", "", tweet)

    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r"[^\x00-\x7F]+", " ", tweet)

    # remove emojis from tweet
    tweet = emoji_pattern.sub(r"", tweet)

    stop_words = set(stopwords.words("german"))
    word_tokens = word_tokenize(tweet)  # TODO: use other tokenizer?

    # filter using NLTK library append it to a string
    filtered_tweet = []
    for w in word_tokens:
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)

    return " ".join(filtered_tweet)


def preprocess_tweet2(sentence):
    """https://stackoverflow.com/questions/54396405/how-can-i-preprocess-nlp-text-lower
    case-remove-special-characters-remove-numb/54398984"""

    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace("{html}", "")
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", sentence)
    rem_url = re.sub(r"http\S+", "", cleantext)
    rem_num = re.sub("[0-9]+", "", rem_url)
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [
        w for w in tokens if len(w) > 2 if w not in stopwords.words("english")
    ]

    return " ".join(filtered_words)


def preprocess_dataframe(dataframe):
    dff = dataframe.copy()
    dff["text"] = dff["text"].apply(p.clean)
    dff["text"] = dff["text"].apply(preprocess_tweet)
    dff["text"] = dff["text"].apply(preprocess_tweet2)
    dff["text"] = (
        dff["text"].str.lower().str.replace(r"[^\w\s]", " ").str.replace(r"\s\s+", " ")
    )
    return dff


if __name__ == "__main__":
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")

    # load emoticons
    tokens = json.load(open(ppj("PRE_PROCESS", "tokens.json")))
    emoticons_happy = set(tokens["emoticons_happy"])
    emoticons_sad = set(tokens["emoticons_sad"])
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    emoticons = emoticons_happy.union(emoticons_sad)

    # load config
    config = json.load(open(ppj("PRE_PROCESS", "config.json")))

    # load cleaned data
    df = pd.read_pickle(ppj("OUT_DATA", "data_clean.pkl"))

    df_processed = preprocess_dataframe(df)

    seed = 1
    df_train, df_test = train_test_split(
        df_processed, test_size=0.25, random_state=seed
    )

    # save data
    df_processed.to_pickle(ppj("OUT_DATA", "data_processed.pkl"))
    df_train.to_json(
        ppj("OUT_DATA", "data_processed_train.json"), "records", lines=True
    )
    df_test.to_json(ppj("OUT_DATA", "data_processed_test.json"), "records", lines=True)
