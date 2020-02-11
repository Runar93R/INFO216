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
