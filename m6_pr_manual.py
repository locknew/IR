import json
import os
import pickle
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer

from m3 import preProcess, BM25
from m6_pr import Pr
from elasticsearch import Elasticsearch

class Indexer:
    def __init__(self):
        self.crawled_folder = Path(os.path.abspath('')).parent / '../crawled/'
        self.stored_file = 'resources/manual_indexer.pkl'
        if os.path.isfile(self.stored_file):
            cached_dict = pickle.load(open(self.stored_file, 'rb'))
            self.__dict__.update(cached_dict)
        else:
            self.run_indexer()
    def run_indexer(self):
        documents = []
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                documents.append(j)
        self.documents = pd.DataFrame.from_dict(documents)
        tfidfVectorizer = TfidfVectorizer(preprocessor=preProcess, stop_words=stopwords.words('english'))
        self.bm25 = BM25(tfidfVectorizer)
        self.bm25.fit(self.documents.apply(lambda s: ' '.join(s[['title', 'text']]), axis = 1))
        pickle.dump(self.__dict__, open(self.stored_file, 'wb'))

    def query(self, q):
        self.pr = Pr(alpha=0.85)
        pr_score_list = self.pr.pr_calc()
        pr_score_list = self.pr.pr_result
        return_score_list = self.bm25.transform(q)
        for i in range(len(pr_score_list)):
            return_score_list[i] = return_score_list[i] * pr_score_list.iloc[i][0]
        hit = (return_score_list>0).sum()
        rank = return_score_list.argsort()[::-1][:hit]
        results = self.documents.iloc[rank].copy().reset_index(drop = True)
        results['score'] = return_score_list[rank]
        return results
