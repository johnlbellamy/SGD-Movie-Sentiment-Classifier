#!/usr/bin/env python
import re
import pickle
import os
import string
import pandas as pd

class TextCleaner:
    """
    Takes text where text is an array-like objects and performs:
    alpha_iterator(text) ==>  Returns lower-case letters stripped of 
    punctuation and numbers. 
    Note: By default alpha_iterator removes numbers and emoticons.To keep numbers:
        alpha_iterator(text, remove_numeric = False)
        Likewise, to keep emoticons: 
        alpha_iterator(text, remove_emoticon = False)
        To Keep both:
        alpha_iterator(text, remove_emoticon = False, remove_numeric = False)
    stop_word_iterator(text) ==>  Removes common "stop" words, like "and".
        
    custom_stop_word_iterator(text, stop_words) ==> Custom stop_words in 
    list format. stop_words are words to be removed.
    can use this in-lieu of stop_word_iterator, or in addition to.
    streaming_cleaner_and_tokenizer ==> A text cleaner, tokenizer, and stop-word remover 
    that is designed to be used with bag-of-word, out-of-processor, or other algorithms 
    that have an argument for a tokenizer.
    Method is applied for each row in a list. Pass this as an argument wherever you need a tokenizer/processor
    for algorithms like bag-of-words. Note that this method keeps emoticons, since they can 
    help in determinng polarity of text.
    
     STATIC METHOD:
        TextCleaner.tokenizer(text) ==> Returns tokens (unigrams) from a 
        list of sentences. 
	
    GENERAL USAGE:
        1) Move TextCleaner/TextCleaner folder to root directory of your project, 
        where .py or .ipynb will live.
    Import:
        2) from  TextCleaner2000.TextCleaner import TextCleaner
        Instance Instantiation:
        3) Simply  instantiate a cleaner object with empty call: 
        cleaner = TextCleaner()
   
    METHOD USAGE:
        For the following examples, text refers to an array-like object. 
        For best results, pass text as a list() or a Pandas 
		 DataFrame column: (assuming data_frame is a pandas DataFrame) data_frame["column_name"].
        For stop words used in custom stop word removal, pass stop words as a list().	
    GENERAL NUMBER AND PUNCTUATION REMOVAL:
        alpha_words = cleaner.alpha_iterator(text, remove_emoticon = True, remove_numeric = True)
    COMMON STOPWORD REMOVAL:
        cleaned_of_stops = cleaner.stop_word_iterator(text)
    CUSTOM STOPWORD REMOVAL:
        cleaned_of_custom_stops = cleaner.custom_stop_word_iterator(text, stop_words)
        Remember that stop_words is a comma-separated list().
    STREAMING TEXT:
        Pass TextCleaner.streaming_cleaner_and_tokenizer as an argument to an algorithm
        that calls for a tokenizer. For example, with HashingVectorizer which can
        be used with SGDClassifier:
            HashingVectorizer(decode_error = 'ignore',
                        n_features = 2**21,
                        preprocessor = None,
                        tokenizer = cleaner.streaming_cleaner_and_tokenizer)
            """  
    
    def __init__(self, tc_2000_home = ""):
        self.tc_2000_home = tc_2000_home
        pickle_file_path = 'stops.pkl'
        
        if tc_2000_home == "":
            pkl_file = open(os.path.join(os.getcwd() + '\\text_cleaner_2000',pickle_file_path), 'rb')
            self.stop_words = pickle.load(pkl_file) 
            pkl_file.close() 
        else:
            pkl_file = open(os.path.join(tc_2000_home,pickle_file_path), 'rb')
            self.stop_words = pickle.load(pkl_file) 
            pkl_file.close() 

    def __alphaizer(self, text, remove_numeric, remove_emoticon):
        """Given a string (text), removes all punctuation and numbers.
        Returns lower-case words. Called by the iterator method
        alpha_iterator to apply this to lists, or array-like (pandas dataframe)
        objects."""
        self.text = text
        self.remove_numeric = remove_numeric
        self.remove_emoticon = remove_emoticon
        
        if remove_numeric and remove_emoticon:
            non_numeric = ''.join(i for i in text if not i.isdigit())
            non_numeric = re.sub('<[^>]*>','', non_numeric)
            translation = str.maketrans("","", string.punctuation)
            cleaned = non_numeric.translate(translation)
            cleaned = re.sub("\s+$" ,"" , cleaned) # Removes trailing spaces
            cleaned = re.sub("  "," ", cleaned) #Removes extra spaces
            cleaned = cleaned.replace('\\','').lower()
            return cleaned
        
        elif not remove_numeric and remove_emoticon:
            translation = str.maketrans("","", string.punctuation)
            non_numeric = re.sub('<[^>]*>','', text)
            cleaned = non_numeric.translate(translation)
            cleaned = re.sub("^\s" , "" , cleaned) #Removes leading spaces
            cleaned = re.sub("\s+$" ,"" , cleaned) # Removes trailing spaces
            cleaned = re.sub("  "," ", cleaned) #Removes extra spaces
            cleaned = cleaned.replace('\\','').lower() #Removes back-space
            return cleaned
        
        elif not remove_numeric and not remove_emoticon:
            translation = str.maketrans("","", string.punctuation)
            non_numeric = re.sub('<[^>]*>','', text)
            cleaned = non_numeric.translate(translation)
            cleaned = re.sub("^\s" , "" , cleaned) #Removes leading spaces
            cleaned = re.sub("\s+$" ,"" , cleaned) # Removes trailing spaces
            cleaned = re.sub("  "," ", cleaned) #Removes extra spaces
            cleaned = cleaned.replace('\\','').lower() #Removes back-space
            emoticons = TextCleaner.__emoticon_finder(text)
            emoticons.replace('-','')
            clean =  cleaned + ' ' + emoticons
            return clean.rstrip()
        
        elif remove_numeric and not remove_emoticon:
            non_numeric = ''.join(i for i in text if not i.isdigit())
            translation = str.maketrans("","", string.punctuation)
            non_numeric = re.sub('<[^>]*>','', non_numeric)
            cleaned = non_numeric.translate(translation)
            cleaned = re.sub("^\s" , "" , cleaned) #Removes leading spaces
            cleaned = re.sub("\s+$" ,"" , cleaned) # Removes trailing spaces
            cleaned = re.sub("  "," ", cleaned) #Removes extra spaces
            cleaned = cleaned.replace('\\','').lower()
            emoticons = TextCleaner.__emoticon_finder(text=text)
            emoticons.replace('-','')
            clean =  cleaned + ' ' + emoticons
            return clean.rstrip()
          
    @staticmethod     
    def tokenizer(text): 
        """Given a sentence, splits sentence on blanks and returns a list of ngrams or tokens"""
        if type(text) is str:
            tokenized = text.split(' ')
            clean = [ token for token in tokenized if token != '']
            return clean
        
        elif type(text) is pd.core.series.Series or  type(text) is list:
            clean  = [TextCleaner.tokenizer(t) for t in text if t != None]
            return clean
    
    def __stop_word_remover(self, text, stop):
        """Removes common stop-words like: "and", "or","but", etc. Called by
        stop_word_iterator to apply this to lists, or array-like (pandas dataframe)
        objects. """
        self.text = text
        
        clean = ''
        tokens = [t for t in TextCleaner.tokenizer(text) if t not in stop]
        for t in tokens:
            clean += " " + t
        return clean.lstrip()
        
    def __emoticon_finder(text):
        """Finds emoticons."""
        
        emoticons_ = ""
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        for e in emoticons:
            emoticons_ += ' ' + e
        return emoticons_.lstrip()
		
    def stop_word_iterator(self, text):
        """Calls __stop_word_remover to apply this method to array-like objects.
        Usage: TextCleaner.stop_word_iterator(Text)."""
        self.text = text
        
        clean = [self.__stop_word_remover(t, self.stop_words) for t in text]
        return clean
    
    def alpha_iterator(self, text, remove_numeric = True, remove_emoticon = True):
        """
        Calls __alphaizer to apply this method to array-like objects. Usage:
        TextCleaner.alphaizer(Text).
        Note: By default this method removes numbers from each string.
        To change this behavior pass the flag remove_numerals:
        alphaizer(Text, remove_numerals = False)
        """
        self.text = text
        self.remove_numeric = remove_numeric
        self.remove_emoticon = remove_emoticon
        
        clean = [self.__alphaizer(t, remove_numeric, remove_emoticon) for t in text]
        return clean
    
    def custom_stop_word_iterator(self, text, stop_words):
        """Removes custom stop-words. For cleaned example, "patient", or "medicine", if
        one is dealing with medical Text and do not want to include those words 
        in analysis. Can use this method to pass any set of stop
        words, or in-lieu of common stop-word method stop_word_iterator.Calls 
        __stop_word_remover to apply this method to array-like objects. Usage:
        TextCleaner.custom_stop_word_iterator(Text, stop_words), where 
        stop-words and Text are in a comma-
        separated list, or iterable."""
        self.text = text
        
        clean = [self.__stop_word_remover(t, stop_words) for t in text]
        return clean
    
    def streaming_cleaner_and_tokenizer(self, text, remove_numeric = True, remove_emoticon = False):
        """Called per line of text in a a stream application, such as SGD.
        Can be passed directly to SGD algorithms as TextCleaner.streaming_cleaner_and_tokenizer"""
        self.text = text
        self.remove_emoticon = remove_emoticon
        self.remove_numeric = remove_numeric
        
        alpha = self.__alphaizer(text = text, remove_numeric = remove_numeric, 
                                        remove_emoticon = remove_emoticon)
        clean = self.__stop_word_remover(alpha, self.stop_words)
        tokens = TextCleaner.tokenizer(clean)
        return tokens

        