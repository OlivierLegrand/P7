#! /anaconda3/bin/python

"""Process text: accepts a unicode-encoded paragraph (one or several sentences) as an input and performs the following tasks:
- remove uninformative sentences (sentences found in almost all samples from the corpus, but bearing no information on the sample)
- tokenize
- lemmatize.
Returns the tokens from the sample"""

import warnings
warnings.filterwarnings('ignore')

import re
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import argparse
import time

from sklearn import metrics
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn import manifold
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import cluster
from sklearn.model_selection import ParameterGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools



#nltk.download('wordnet')

# création du set de stopwords
print('Creation du set de stopwords...')
SW = set()
SW.update(nltk.corpus.stopwords.words('english'))
SW.update(STOP_WORDS)
print("Création du set de stopwords terminée")



def clean_sample(sample):
    
    cleaned_sample = re.sub('\n|\r|\t', ' ', sample)
    cleaned_sample = re.sub('\s{2,}', ' ', cleaned_sample)
    
    return cleaned_sample



def prepare_text(sample, **kwargs):
    # création d'un motif regex à partir de quelques phrases récurrentes et non-informatives. 
    sentences_to_remove = ['rs.', 
                           'flipkart.com',
                           'free shipping',
                           'cash on delivery', 
                           'only genuine products', 
                           '30 day replacement guarantee',
                           '\n',
                           '\r',
                           '\t',
                           r'\bcm\b',
                           r'\bbuy\b',
                           r'\bl\b'
                          ]

    sent_rm = kwargs.pop('sentences_to_remove', sentences_to_remove)
    pattern = "|".join(sentences_to_remove)
    cleaned_text = re.sub(pattern, ' ', sample)
    pattern_2 = re.compile(r"\s{2,}")
    cleaned_text = re.sub(pattern_2, ' ', cleaned_text)
    return cleaned_text


# fonctions de lemmatization et tokenization
print("Chargement du lemmatizer...")
lemmatizer = WordNetLemmatizer()

def get_lemma(tokens, lemmatizer, stop_words):
    lemmatized = []
    for item in tokens:
        if item not in stop_words:
            lemmatized.append(lemmatizer.lemmatize(item))
    return lemmatized

def tokenize(sample, **kwargs):
    tokenizer = kwargs.pop('tokenizer', None)
    if tokenizer:
        tokens = tokenizer.tokenize(sample)
        lemma = kwargs.pop('lemmatizer', None)
        stop_w = kwargs.pop('stop_words', None)
        if lemma:
            if stop_w:
                lemmas = get_lemma(tokens, lemmatizer, stop_w)
            else:
                lemmas = get_lemma(tokens, lemmatizer)
        else:
            lemmas = tokens
        
        return lemmas
    else:
        print("Missing tokenizer") 

# motif regex pour la tokenization. Ce motif filtre directement les caractères spéciaux comme 
# \n, \t \r etc., ainsi que les chiffres.
print("Chargement du tokenizer...")
tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]+')

def process_text(sample, script=False):
    cleaned = prepare_text(sample, script=script)
    tokens = tokenize(cleaned, tokenizer=tokenizer,lemmatizer=lemmatizer,stop_words=SW)
    if script:
        print(' '.join(tokens))
    else:
        return ' '.join(tokens)

if __name__ == '__main__':
    sample = input('Entrer le texte à tokenizer: ')
    print()
    print("Résultat:")
    process_text(sample, script=True)