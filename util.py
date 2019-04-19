import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import spacy
import re

from tqdm import tqdm
from collections import Counter
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk import tokenize


stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
nlp = spacy.load('en', disable=['parser', 'ner'])

def flatten(l):
    # simple utility to flatten a python list, useful for data loading
    return [item for sublist in l for item in sublist]

def clean(data):
    # Split up and clean the dataset, remove all weird punctuation
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    # Weird spacing to normal space
    data = [re.sub('[^\w\s]', ' ', sent) for sent in data]
    # lowercase it all and split into list so that next step works easil
    data = [sent.lower().split() for sent in data]
    return data

def preprocess(data):
    # remove stopwords, gets rid of confounding in clusters (stopwords have
    # very little semantic meaning)
    nouns = Counter()
    data = [[word for word in doc if word not in stop_words] for doc in data]
    out = []
    allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    # TODO: Come back to this when we have more compute lol
    count = Counter()
    for sent in tqdm(data):
        doc = nlp(" ".join(sent))
        tokenized = []
        for i, token in enumerate(doc):
            if token.pos_ in allowed_postags:
                tokenized.append(token.text) 
                count[token.text] += 1
            if token.pos_ == 'NOUN':
                nouns[token.text] += 1
        out.append(tokenized)
    return out, count, nouns

def merge(data, u, v):
    # remove stopwords, gets rid of confounding in clusters (stopwords have
    # very little semantic meaning)
    data = [[word for word in doc if word not in stop_words] for doc in data]
    out = []
    allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    # TODO: Come back to this when we have more compute lol
    count = Counter()
    for sent in tqdm(data):
        doc = nlp(" ".join(sent))
        tokenized = []
        for i, token in enumerate(doc):
            if token.pos_ in allowed_postags:
                if (token.text == u) or (token.text == v):
                    t = u + "_" + v
                else:
                    t = token.text
                tokenized.append(t) 
                count[t] += 1
        out.append(tokenized)
    return out, count


