#! /anaconda3/bin/python

"""Process text: accepts a unicode-encoded paragraph (one or several sentences) as an input and performs the following tasks:
- remove uninformative sentences (sentences found in almost all samples from the corpus, but bearing no information on the sample)
- tokenize
- lemmatize.
Returns the tokens from the sample"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import argparse


#nltk.download('wordnet')

# création du set de stopwords
sw = set()
sw.update(nltk.corpus.stopwords.words('english'))
sw.update(STOP_WORDS)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample", help="""Sample of text to analyse.""")
    return parser.parse_args()

def prepare_text(sample):
    # création d'un motif regex à partir de quelques phrases récurrentes et non-informatives. 
    sentences_to_remove = ['rs.', 'flipkart.com', 'free shipping', 'cash on delivery', 'only genuine products', '30 day replacement guarantee']
    pattern = '|'.join(s for s in sentences_to_remove)
    cleaned_text = re.sub(pattern, '', sample)
    return cleaned_text


# fonctions de lemmatization et tokenization
lemmatizer = WordNetLemmatizer()
def get_lemma(tokens, lemmatizer, stop_words):
    lemmatized = []
    for item in tokens:
        if item not in stop_words:
            lemmatized.append(lemmatizer.lemmatize(item))
    return lemmatized


def tokenize(sample, regex):
    tokenizer = nltk.RegexpTokenizer(regex)
    tokens = tokenizer.tokenize(sample)
    lemmas = get_lemma(tokens, lemmatizer, sw)
    return lemmas


# Récupère le texte donné en argument
args = parse_arguments()
sample = args.sample

def process_text(sample=sample, script=False):
    # motif regex pour la tokenization. Ce motif filtre directement les caractères spéciaux comme \n, \t \r etc., ainsi que les chiffres.
    pattern = re.compile(r'[a-zA-Z]+')

    cleaned = prepare_text(sample)
    tokens = tokenize(cleaned, pattern)
    if script:
        print(tokens)
    else:
        return tokens

if __name__ == '__main__':
    process_text(script=True)