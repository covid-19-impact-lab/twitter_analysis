"""This module contains the code to translate with https://www.deepl.com/.

The API allows to process a batch of 50 texts at a time and allows parallel calls to the
API from processes or threads.

The secret API is contained in a file which should never be committed!

The documentation for the API can be found here:
https://www.deepl.com/docs-api/translating-text.

"""
from pathlib import Path

import pandas as pd
import requests
import yaml

from bld.project_paths import project_paths_join as ppj


URL = "https://api.deepl.com/v2/translate"


def translate_text_with_deepl(text, target_lang="EN"):
    """Translate the text.

    `'split_sentences' = 0` tells deepl to translate a string as a whole and without
    splitting it first into sentences. I hope this improves the translation quality
    because it gives more context, but it might be unnecessary.

    Note that translation can be parallelized via threads or processes. The API also
    allows to pass batches of text which cannot be more than 50.

    Parameters
    ----------
    text : str or list
        Text can either be a string or a list of strings.

    Returns
    -------
    translated_texts : list

    """
    assert isinstance(text, (str, list, pd.Series)), "'text' is not str/list/pd.Series."

    if isinstance(text, (list, pd.Series)):
        assert len(text) <= 50, "The DeepL API only allows 50 text per batch request."

    secrets = yaml.safe_load(Path(ppj("PROJECT_ROOT", ".secrets.yaml")).read_text())
    auth_key = secrets["deepl-auth-key"]

    data = {
        "auth_key": auth_key,
        "text": text,
        "target_lang": target_lang,
        "split_sentences": "0",
    }

    response = requests.post(URL, data=data)
    response = response.json()

    detected_source_language = [
        tweet["detected_source_language"] for tweet in response["translations"]
    ]
    translated_texts = [tweet["text"] for tweet in response["translations"]]

    if len(response["translations"]) == 1:
        detected_source_language = detected_source_language[0]
        translated_texts = translated_texts[0]

    return detected_source_language, translated_texts


def apply_to_batch_of_series(series, func, batch_size, **kwargs):
    """Apply function to batch of series.

    This function is a convenience wrapper to process a batch of observations of series
    at a time.

    Parameters
    ----------
    series : pd.Series
        Series containing the text.
    func : callable
        Function which is applied to all values of the series.
    batch_size : int
        Number of elements which can be simultaneously passed to the series.

    """
    assert isinstance(
        series, (pd.Series, list)
    ), "Only use this function with a series or a list."

    df = pd.DataFrame(
        index=series.index, columns=["detected_source_language", "translation"]
    )
    for batch_start in range(0, len(series), batch_size):
        slice_ = slice(batch_start, batch_start + batch_size)
        detected_source_language, translation = func(series[slice_], **kwargs)

        df.iloc[slice_, 0] = detected_source_language
        df.iloc[slice_, 1] = translation

    return df
