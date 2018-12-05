# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:57:27 2018

@author: johnb
"""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class DocStreamer:
    """"""
    @staticmethod
    def stream_docs(path):
        """Returns data from file one line at a time."""
        with open(path, 'r', encoding="utf8") as csv:
            next(csv) # Skips header
            for line in csv:
                text, label = line[:-3], int(line[-2])
                yield text, label
                
    @staticmethod
    def get_mini_batch(doc_stream, size):
        """ Gets data in mini-batches to process out-of-core."""
        docs, y = [], []
        try:
            for _ in range(size):
                text, label = next(doc_stream)
                docs.append(text)
                y.append(label)
        except StopIteration:
            return None, None
        return docs, y
            
    
    @staticmethod
    def tokenizer(text):
        stops = stopwords.words('english') 
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
        text = re.sub('[\W]+', ' ', text.lower())\
        + ' '.join(emoticons).replace('-','')
        tokenized = [t for t in text.split() if t not in stops]
        return tokenized