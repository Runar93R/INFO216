import numpy as np
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
# importer norskspråkelig modul i spacy
import nb_core_news_sm

from tqdm import tqdm_notebook as tqdm
from pprint import pprint

nrk = pd.read_csv('nrk.csv')
newest_doc = nrk['content']


nlp = spacy.load("nb")

# My list of stop words.
stop_list = ["Mrs.", "Ms.", "say", "WASHINGTON", "'s", "Mr.", ]

# Updates spaCy's default stop words list with my additional words.
nlp.Defaults.stop_words.update(stop_list)

# Iterates over the words in the stop words list and resets the "is_stop" flag.
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True
