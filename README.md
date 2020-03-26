# twitter_analysis
Sentiment Analysis of (german) Twitter data using natural language processing

[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-badge]][license-url]
[![Black Code Style][black-badge]][black-url]

## TODO

- [ ] Have a small environment for computation on server?

# Project Goal

We want to classify tweets related to the corona crisis with the usual labels "positive", "negative" and "neutral".
(In a later stage we might extent the model for more categories as "anger", "fear", etcetera.)
We consider german tweets, so we need a model which understands the german language.
In ``data/original-data/`` we have our training data set for german tweets.
We will extend this data set with new labelled data coming from the corona crisis.


## Install

To install the project execute the following commands in a terminal.
(Note that the conda environment is very large as we are using libraries such as ``pytorch``.
This can lead to quite some waiting time for the environment creation.
If you want to avoid this waiting time follow the steps in ``small environment``.)

```
git clone https://github.com/timmens/twitter_analysis.git

conda env create -f environment.yml
conda activate twitter_analysis

python -m spacy download en_core_web_sm
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


### Resources

#### Main Resources
- https://github.com/bentrevett/pytorch-sentiment-analysis
- https://github.com/huggingface/transformers/tree/master/notebooks
- https://www.kdnuggets.com/2019/11/lit-bert-nlp-transfer-learning-3-steps.html


#### Kaggle Resources
- https://www.kaggle.com/bertcarremans/deep-learning-for-sentiment-analysis
- https://www.kaggle.com/menion/sentiment-analysis-with-bert-87-accuracy
- https://www.kaggle.com/ronitmankad/sentiment-analysis-bert-pytorch
- https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
- https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2

#### Other
- https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4
- https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-2-2a92fdc0160d
- https://towardsdatascience.com/fine-grained-sentiment-analysis-part-3-fine-tuning-transformers-1ae6574f25a6
- https://towardsdatascience.com/fasttext-sentiment-analysis-for-tweets-a-straightforward-guide-9a8c070449a2
- https://github.com/kaushaltrivedi/fast-bert

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/timmens/twitter_analysis
[contributors-url]: https://github.com/timmens/twitter_analysis/graphs/contributors
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/timmens/twitter_analysis/blob/master/LICENSE
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-url]: https://github.com/psf/black
