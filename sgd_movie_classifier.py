# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:26:03 2018

@author: johnb
"""
import numpy as np
from doc_streamer.doc_streamer import DocStreamer
from utils.utils import report

import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import pickle

classes = np.array([(0, 1)])

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**20,
                         preprocessor=None,
                         tokenizer=DocStreamer.tokenizer)

print("Finding best params...")

param_grid = {
    "average": [True, False],
    "l1_ratio": np.linspace(0, 1, num=10),
    "alpha": np.power(10, np.arange(-2, 1, dtype=float)),
}
#penalty="elasticnet",
clf = SGDClassifier(loss="log_loss", fit_intercept=True, random_state=1)

try:
    df = pd.read_csv('data/ratings_shuffled.csv')

except FileNotFoundError:
    print(
        "Couldn't find file. Make sure you unzip data/ratings_shuffled.zip into data directory"
    )

X = df['review']
X = vect.transform(X)

y = df['sentiment']

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)

grid_search.fit(X, y)

report(grid_search.cv_results_)

print(grid_search.best_params_)
print("Building model on best params..\n")

print("Preparing data: train test split")

X_train, X_test, y_train, y_test = train_test_split(df["review"],
                                                    df["sentiment"],
                                                    test_size=.20)

X_train = vect.transform(X_train)
X_test = vect.transform(X_test)

clf = SGDClassifier(loss="log_loss",
                    fit_intercept=True,
                    random_state=1,
                    alpha=grid_search.best_params_.get("alpha"),
                    average=grid_search.best_params_.get("average"),
                    l1_ratio=grid_search.best_params_.get("l1_ratio"))

clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"Accuracy for test: {score}")

print("Preparing evaluation metrics...\n")

preds = clf.predict(X_test)

cm = confusion_matrix(y_test, preds)

hm = sns.heatmap(cm, annot=True, fmt='g')

hm.figure.savefig("visuals/heatmap.png")

rating = "Better Call Saul is one of the best shows on television! Brilliant, poignant, and spectacular!"

X = vect.transform([
    rating
])

pred = clf.predict(X)
if pred > .5:
    score = "pos"
else:
    score = "neg"

print(f"Your rating: \n{rating} is: {score}")