import numpy as np
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from M2 import *


def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    s = [w for w in s if w not in stopwords_set]
    # s = [w for w in s if not w.isdigit()]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


doc_1 = cleaned_description.iloc[0]
doc_else = cleaned_description.iloc[1:]
doc_1_norm = doc_1.divide(np.square((doc_1 ** 2).sum))
doc_else_norm = doc_else.divide(np.square((doc_else ** 2).sum(axis=1)), axis=0)
