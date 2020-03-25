# src folder

## Original data

We differentiate between data used to train the model and data which we want to analyze using the trained model.
The data is not stored on git but will be send to you if you contact one of the authors.

## Data Management

Here we do basic data cleaning such that our training and evaluation data is in the form

sentiment | tweet
----------|------
negative  | Diese corona scheisse geht mir ordentlich auf den senkel!!!!!
neutral   | Das Wetter heute wird bei 14 grad liegen.
positive  | Die neuen folgen GNTM sind wirklich mega gut :) !

## Pre Processing

To get better results of the learning method we use, we pre process our data.
Here we produce the finished data sets which will then be either feed in the network to for training purposes or for evaluation purposes.

## Model Train

Here we use code from ``model_code`` and specifications from ``model_specs`` to train our method.

## Model Evaluation

Here we use the trained method to classify new data.

## Model Code

Here we store model code which defines the method we are using. This code will be used to train the method in ``model_train``.

## Model Specifications

Here we store model specifications which have to be set beforehand and cannot be trained.

## Final

At last we analyze our results.

## Literature

Here we store links and references to literature we are using.
