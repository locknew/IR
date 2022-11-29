import string

import numpy as np
import pandas as pd
import requests as requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import tabulate


def get_and_clean_data():
    data = pd.read_csv('resource/software_developer_united_states_1971_20191023_1.csv')
    description = data['job_description']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(
        lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.drop_duplicates()
    return cleaned_description


def simple_tokenize(data):
    cleaned_description = data.apply(lambda s: [x.strip() for x in s.split()])
    return cleaned_description


def parse_job_description():
    cleaned_description = get_and_clean_data()
    cleaned_description = simple_tokenize(cleaned_description)
    return cleaned_description


def count_python_mysql():
    parsed_description = parse_job_description()
    count_python = parsed_description.apply(lambda s: 'python' in s).sum()
    count_mysql = parsed_description.apply(lambda s: 'mysql' in s).sum()
    print('python: ' + str(count_python) + ' of ' + str(parsed_description.shape[0]))
    print('mysql: ' + str(count_mysql) + ' of ' + str(parsed_description.shape[0]))


def parse_db():
    html_doc = requests.get("https://db-engines.com/en/ranking").content
    soup = BeautifulSoup(html_doc, 'html.parser')
    db_table = soup.find("table", {"class": "dbi"})
    all_db = [''.join(s.find('a').findAll(text=True, recursive=False)).strip() for s in
              db_table.findAll("th", {"class": "pad-l"})]
    all_db = list(dict.fromkeys(all_db))
    db_list = all_db[:10]
    db_list = [s.lower() for s in db_list]
    db_list = [[x.strip() for x in s.split()] for s in db_list]
    return db_list


cleaned_db = parse_db()


def showdb():
    parsed_description = parse_job_description()
    raw = [None] * len(cleaned_db)
    for i, db in enumerate(cleaned_db):
        raw[i] = parsed_description.apply(lambda s: np.all([x in s for x in db])).sum()
        print(' '.join(db) + ': ' + str(raw[i]) + ' of ' + str(parsed_description.shape[0]))


with_python = [None] * len(cleaned_db)


def showlanindb():
    for i, db in enumerate(cleaned_db):
        with_python[i] = parsed_description.apply(lambda s: np.all([x in s for x in db]) and 'java' in s).sum()
        print(' '.join(db) + ' + java: ' + str(with_python[i]) + ' of ' + str(parsed_description.shape[0]))


def showlnper():
    for i, db in enumerate(cleaned_db):
        print(' '.join(db) + ' + java: ' + str(with_python[i]) + ' of ' + str(raw[i]) + ' (' + str(
            np.around(with_python[i] / raw[i] * 100, 2)) + '%)')


def showmap():
    lang = [['java'], ['python'], ['c'], ['kotlin'], ['swift'], ['rust'], ['ruby'], ['scala'], ['julia'],
            ['lua']]
    parsed_description = parse_job_description()
    parsed_db = parse_db()
    all_terms = lang + parsed_db
    query_map = pd.DataFrame(parsed_description.apply(
        lambda s: [1 if np.all([d in s for d in db]) else 0 for db in all_terms]).values.tolist(),
                             columns=[' '.join(d) for d in all_terms])

    query_map[query_map['python'] > 0].apply(lambda s: np.where(s == 1)[0], axis=1).apply(
        lambda s: list(query_map.columns[s]))


def listing():
    nltk.download('stopwords')
    nltk.download('punkt')
    str1 = 'the chosen software developer will be part of a larger engineering team developing software for medical devices.'
    str2 = 'we are seeking a seasoned software developer with strong analytical and technical skills to join our public sector technology consulting team.'
    tokened_str1 = word_tokenize(str1)
    tokened_str2 = word_tokenize(str2)
    tokened_str1 = [w for w in tokened_str1 if len(w) > 2]
    tokened_str2 = [w for w in tokened_str2 if len(w) > 2]
    no_sw_str1 = [word for word in tokened_str1 if not word in stopwords.words()]
    no_sw_str2 = [word for word in tokened_str2 if not word in stopwords.words()]
    ps = PorterStemmer()
    stemmed_str1 = np.unique([ps.stem(w) for w in no_sw_str1])
    stemmed_str2 = np.unique([ps.stem(w) for w in no_sw_str2])
    full_list = np.sort(np.concatenate([stemmed_str1, stemmed_str2]))


def inverse_indexing(parsed_description):
    sw_set = set(stopwords.words()) - {'c'}
    no_sw_description = parsed_description.apply(lambda x: [w for w in x if w not in sw_set])
    ps = PorterStemmer()
    stemmed_description = no_sw_description.apply(lambda x: set([ps.stem(w) for w in x]))
    all_unique_term = list(set.union(*stemmed_description.to_list()))

    invert_idx = {}
    for s in all_unique_term:
        invert_idx[s] = set(stemmed_description.loc[stemmed_description.apply(lambda x: s in x)].index)

    return invert_idx


def search(invert_idx, query):
    ps = PorterStemmer()
    processed_query = [s.lower() for s in query.split()]
    stemmed = [ps.stem(s) for s in processed_query]
    matched = set.intersection(*[invert_idx[s] for s in stemmed])
    return matched


if __name__ == '__main__':
    parsed_description = parse_job_description()
    invert_idx = inverse_indexing(parsed_description)
    query = 'java oracle'
    matched = search(invert_idx, query)
    print(parsed_description.loc[matched].apply(lambda x: ' '.join(x)).head().to_markdown())
