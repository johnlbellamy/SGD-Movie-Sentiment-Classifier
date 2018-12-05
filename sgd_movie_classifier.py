# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:26:03 2018

@author: johnb
"""

import pyprind
import numpy as np
from doc_streamer.doc_streamer  import DocStreamer
#from text_cleaner_2000.text_cleaner import TextCleaner

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pickle

classes = np.array([(0,1)])
pbar = pyprind.ProgBar(45)

vect = HashingVectorizer(decode_error = 'ignore',
                        n_features = 2**20,
                        preprocessor = None,
                        tokenizer = DocStreamer.tokenizer)


clf = SGDClassifier(loss = 'log',
                    random_state = 1,
                    max_iter = 1000,
                    #alpha = 0.0000009,
                    #eta0 = 0.1, 
                    #learning_rate = 'constant'
                   )

try:
    doc_stream = DocStreamer.stream_docs(path = 'DATA\\ratings_shuffled\\ratings_shuffled.csv')


except FileNotFoundError:
    
    try:
	    doc_stream = DocStreamer.stream_docs(path = 'DATA/ratings_shuffled/ratings_shuffled.csv')
        #df = pd.read_csv('DATA/ratings/ratings.csv')
    
    except FileNotFoundError:
	    print("File wasn't found. Make sure you unzipped results.zip")

print("Training model...")

# Gives us 45k for training
for _ in range(45):
    X_train, y_train = DocStreamer.get_mini_batch(doc_stream, size = 1000)
    if not X_train:
        break
    
    X_train = vect.transform(X_train)
    
    clf. partial_fit(X_train, y_train, classes = classes)
    pbar.update()


print("Done...")
print("Getting accuracy information...")
#X_test, y_test = df['review'][44999:],df['sentiment'][44999:]
X_test, y_test = DocStreamer.get_mini_batch(doc_stream, size = 5000)

X_test = vect.transform(X_test)

print("Accuracy: {}".format(clf.score(X_test, y_test)))

print("Updating model with test data...")
clf = clf.partial_fit(X_test, y_test)
print("Dumping model for use in web app...")

try:
    
    pickle.dump(clf,
                open('pkl_objects\\classifier.pkl', 'wb'), 
                protocol = 4)
except FileNotFoundError:
    pickle.dump(clf, 
                open('pkl_objects/classifier.pkl', 'wb'), 
                protocol = 4)