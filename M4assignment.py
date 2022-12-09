import M3
import itertools
import os
from multiprocessing.pool import ThreadPool as Pool
from string import ascii_lowercase

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

topdir = 'resource/WIKIQA'
all_content = []
for dirpath, dirname, filename in os.walk(topdir):
    for name in filename:
        if name.endswith('.txt'):
            with open(os.path.join(dirpath, name)) as f:
                all_content.append(f.read())

processed_content = [M3.preProcess(s) for s in all_content]

vectorizer = CountVectorizer()
vectorizer.fit(processed_content)
freq_wikiqa = vectorizer.transform(processed_content)
freq_wikiqa = pd.DataFrame(freq_wikiqa.todense(), columns=vectorizer.get_feature_names()).sum()

query = ['max', 'man', 'map', 'math']
transformed_query = [vectorizer.inverse_transform(vectorizer.transform([q])) for q in query]
query_freq = pd.Series([freq_wikiqa.T.loc[tq[0]].values[0] if len(tq[0]) > 0 else 0 for tq in transformed_query],
                       index=query)

WIKIQA = pd.DataFrame([['max', 19], ['man', 137], ['map', 67], ['math', 35]], columns=['word', 'frequency'])
WIKIQA_pop = 2.1e6
WIKIQA['P(w)'] = WIKIQA['frequency'] / WIKIQA_pop
WIKIQA['rank'] = WIKIQA['frequency'].rank(ascending=False).astype(int)

freq_iula = vectorizer.transform(processed_content)
