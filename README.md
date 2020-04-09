# Twitter Analysis

Analysis of (german) Twitter data related to the Corona crisis using natural language processing and other methods from the data science toolbox.

[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-badge]][license-url]
[![Black Code Style][black-badge]][black-url]

## Project Goal

For the sentiment analysis we aim to classify tweets with the usual labels ``positive``, ``negative`` and ``neutral``.
We extend this standard analysis to more scales, namely we consider the following scales.

1. ``negative`` - ``neutral`` - ``positive``
2. ``pessimistic`` - ``neutral`` - ``optimistic``
3. ``upset`` - ``neutral`` - ``pleased``
4. ``objective`` - ``emotional``

In a first step of the analysis, we are going to explore the capabilities of pre-trained sentiment models ([vader](https://github.com/cjhutto/vaderSentiment), [textBlob](https://textblob.readthedocs.io/en/dev/quickstart.html)), which are designed to classifiy *English* tweets.
Therefore, we will first translate the German tweets to English using [DeepL](https://www.deepl.com/home).

Apart from using existing solutions, we aim at maximizing classification accuracy by building our own natural language model.
For the model we will focus on using architectures from [Hugging Face](https://huggingface.co/).
For the fine-tuning of these models we worked on building our own labeled data set.
This data set contains tweets related to the Corona crisis in Germany, and has been labeled according to the scales above.

## Install

To install the project execute the following commands in a terminal.
(Note that the conda environment is very large as we are using libraries such as ``pytorch``.
This can lead to long waiting times for the environment creation.
If you want to avoid this follow the steps in ``small environment``.)

```
git clone https://github.com/timmens/twitter_analysis.git

conda env create -f environment.yml
conda activate twitter_analysis

python -m spacy download de_core_news_sm

pre-commit install

python waf.py configure
python waf.py build
python waf.py install
```

For more resources on WAF see https://econ-project-templates.readthedocs.io/en/stable/getting_started.html#configuring-your-new-project or https://waf.io/book/.

### Small environment

One can start the project with a small environment and then add packages to the environment when needed.
To do this run
```
conda env create -f small_environment.yml
conda activate twitter_analysis
```
instead of the two conda commands above.

## Entrypoints for engagement / Enhancements

- Improve language detection: Go through the tweets which are categorized as neither
  English nor German. If they are wrongly classified, inspect what needs to change to
  categorize them correctly.

## Literature and Resources

Literature and other resources that we are using can be found in ``src/literature``. 


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/timmens/twitter_analysis
[contributors-url]: https://github.com/timmens/twitter_analysis/graphs/contributors
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/timmens/twitter_analysis/blob/master/LICENSE
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-url]: https://github.com/psf/black
