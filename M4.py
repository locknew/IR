import itertools
import os
from multiprocessing.pool import ThreadPool as Pool
from string import ascii_lowercase

import pandas as pd

import M3

topdir = 'resource/iula'
all_content = []
for dirpath, dirname, filename in os.walk(topdir):
    for name in filename:
        if name.endswith('plain.txt'):
            with open(os.path.join(dirpath, name)) as f:
                all_content.append(f.read())

processed_content = [M3.preProcess(s) for s in all_content]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(processed_content)
freq_iula = vectorizer.transform(processed_content)
freq_iula = pd.DataFrame(freq_iula.todense(), columns=vectorizer.get_feature_names()).sum()

query = ['max', 'man', 'map', 'math']
transformed_query = [vectorizer.inverse_transform(vectorizer.transform([q])) for q in query]
query_freq = pd.Series([freq_iula.T.loc[tq[0]].values[0] if len(tq[0]) > 0 else 0 for tq in transformed_query],
                       index=query)

IULA = pd.DataFrame([['max', 27], ['man', 25], ['map', 618], ['math', 4]], columns=['word', 'frequency'])
IULA_pop = 2.1e6
IULA['P(w)'] = IULA['frequency'] / IULA_pop
IULA['rank'] = IULA['frequency'].rank(ascending=False).astype(int)

freq_iula = vectorizer.transform(processed_content)


def read_norvig():
    norvig = pd.read_csv('http://norvig.com/ngrams/count_1edit.txt', sep='\t', encoding=
    "ISO-8859-1", header=None)
    norvig.columns = ['term', 'edit']
    norvig = norvig.set_index('term')
    print(norvig.head())
    return norvig


def read_norvigori():
    norvig_orig = pd.read_csv('http://norvig.com/ngrams/count_big.txt', sep='\t', encoding=
    "ISO-8859-1", header=None)
    norvig_orig = norvig_orig.dropna()
    norvig_orig.columns = ['term', 'freq']
    print(norvig_orig.head())
    return norvig_orig


def get_count(c, norvig_orig):
    return norvig_orig.apply(lambda x: x.term.count(c) * x.freq, axis=1).sum()


pool = Pool(9)
character_set = list(map(''.join, itertools.product(ascii_lowercase, repeat=1))) + list(
    map(''.join, itertools.product(ascii_lowercase, repeat=2)))
freq_list = pool.starmap(get_count, zip(character_set, itertools.repeat(read_norvigori())))
freq_df = pd.DataFrame([character_set, freq_list], index=['char', 'freq']).T
freq_df = freq_df.set_index('char')





def probtablecoca():
    COCA = pd.DataFrame([['defeat', 21940], ['decet', 6], ['defect', 3972], ['deft', 1240],
                         ['defer', 2237], ['Deeft', 0]], columns=['word', 'frequency'])
    COCA_pop = 1001610938
    COCA['P(w)'] = COCA['frequency'] / COCA_pop
    COCA['rank'] = COCA['frequency'].rank(ascending=False, method='min').astype(int)
    COCA['P(x|w)'] = [(read_norvig().loc['e|ea'].values / freq_df.loc['ea'].values)[0],
                      (read_norvig().loc['f|c'].values / freq_df.loc['c'].values)[0],
                      (read_norvig().loc['e|ec'].values / freq_df.loc['ec'].values)[0],
                      (read_norvig().loc['e| '].values / freq_df.loc['e'].values)[0],
                      (read_norvig().loc['t|r'].values / freq_df.loc['r'].values)[0],
                      (read_norvig().loc['fe|ef'].values / freq_df.loc['ef'].values)[0]]

def probtablewiki():
    WIKI = pd.DataFrame([['defeat', 121408], ['decet', 81], ['defect', 7793], ['deft', 814],
                         ['defer', 1416], ['Deeft', 0]], columns=['word', 'frequency'])
    WIKI_pop = 1.9e9
    WIKI['P(w)'] = WIKI['frequency'] / WIKI_pop
    WIKI['rank'] = WIKI['frequency'].rank(ascending=False, method='min').astype(int)
