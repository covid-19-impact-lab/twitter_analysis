# Sentiment Analysis of (german) Twitter data using natural language processing

## Install

To install the project execute the following commands in a terminal.
(Note that the conda environment is very large as we are using libraries such as ``pytorch``.
This can lead to quite some waiting time for the environment creation.
If you want to avoid this waiting time follow the steps in ``small environment``.)

```console
git clone https://github.com/timmens/twitter_analysis.git

conda env create -f environment.yml
conda activate twitter_analysis

pre-commit install

python waf.py configure
python waf.py build
python waf.py install
```

For more resources on WAF see https://econ-project-templates.readthedocs.io/en/stable/getting_started.html#configuring-your-new-project or https://waf.io/book/.

### Small environment

One can start the project with a small environment and then add packages to the environment when needed.
To do this run

```console
conda env create -f small_environment.yml
conda activate twitter_analysis
```

instead of the two conda commands above.
