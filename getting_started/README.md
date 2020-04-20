# Getting started - Working with the Data

To get a quick grasp of how our data looks like we included a small example notebook.
Click [here](https://nbviewer.jupyter.org/github/covid-19-impact-lab/twitter_analysis/blob/example_notebook/getting_started/twitter_analysis.ipynb) to open the notebook on nbviewer.

If you want to play around with the data locally you need to install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) and run:

```
conda env create -f environment.yml
conda activate twitter_analysis_example
```

In case you want to run the analysis locally using jupyter notebook or jupyter lab, run the following command in addition to the previous ones:

```
python -m ipykernel install --user --name twitter_analysis_example
```

and select this kernel when working in a notebook.
