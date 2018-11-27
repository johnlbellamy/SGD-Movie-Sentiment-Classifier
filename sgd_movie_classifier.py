# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:03:35 2018

@author: johnb
"""

import numpy as np
import pyprind

from doc_streamer  import DocStreamer 
from text_cleaner_2000.text_cleaner import TextCleaner

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

cleaner = TextCleaner()
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

vect = HashingVectorizer(decode_error = 'ignore',
                        n_features = 2**21,
                        preprocessor = None,
                        tokenizer = cleaner.streaming_cleaner_and_tokenizer)

clf = SGDClassifier(loss = 'log',
                    random_state = 1,
                    max_iter=5,
                    tol=1e-3)
try:
    doc_stream = DocStreamer.stream_docs(path = 'DATA\\ratings\\ratings.csv')
	
except FileNotFoundError:
    
    try:
	    doc_stream = DocStreamer.stream_docs(path = 'DATA/ratings/ratings.csv')
        
    except FileNotFoundError:
	    print("File wasn't found. Make sure you unzipped results.zip")
	
print("Training model...")

# Gives us 45k for training
for _ in range(45):
    X_train, y_train = DocStreamer.get_mini_batch(doc_stream, size = 1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes = classes)
    pbar.update()

print("Done...")
print("Getting accuracy information...")

X_test, y_test = DocStreamer.get_mini_batch(doc_stream, size = 5000)
X_test = vect.transform(X_test)

print("Accuracy: {}".format(clf.score(X_test, y_test)))