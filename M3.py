from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import M1
from M1 import *
import M2
from M2 import *
import numpy as np
from scipy.sparse import csr_matrix


def preProcess(s):
    ps = M1.PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    s = [w for w in s if w not in stopwords_set]
    # s = [w for w in s if not w.isdigit()]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


def sk_vectorize():
    cleaned_description = M1.get_and_clean_data()
    vectorizer = CountVectorizer(preprocessor=preProcess)
    vectorizer.fit(cleaned_description)
    vectorizer.transform(cleaned_description)
    print(cleaned_description)
    print(cleaned_description[0])
    query = vectorizer.transform(['good at python and java and oracle'])
    print(query)
    print(vectorizer.inverse_transform(query))


def sk_vectorize_block_of_word():
    cleaned_description = M1.get_and_clean_data()
    vectorizer = CountVectorizer(preprocessor=preProcess)
    vectorizer.fit(cleaned_description)
    print(cleaned_description[0])
    vectorizer = CountVectorizer(preprocessor=preProcess, ngram_range=(1, 2))
    vectorizer.fit_transform(cleaned_description)
    print(vectorizer.get_feature_names())


def ranking():
    N = 5
    cleaned_description = M1.get_and_clean_data()
    cleaned_description = cleaned_description.iloc[:N]
    vectorizer = CountVectorizer(preprocessor=preProcess)
    X = vectorizer.fit_transform(cleaned_description)
    print(X)
    print(X.toarray())

    df = np.array((X.todense() > 0).sum(0))[0]
    idf = np.log10(N / df)
    tf = np.log10(X.todense() + 1)
    tf_idf = np.multiply(tf, idf)
    X = csr_matrix(tf_idf)

    print(X.toarray())

    print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))


def doc_as_vector():
    arr = np.array([[100, 90, 5], [200, 200, 200], [200, 300, 10], [50, 0, 200]])
    print(arr)
    data = pd.DataFrame(arr, columns=['DH', 'CD', 'DC'],
                        index=['business', 'computer', 'git', 'parallel'])
    print(data)
    data = np.log10(data + 1)
    print(data['DH'].dot(data['CD']))
    print(data['DH'].dot(data['DC']))
    print(data['CD'].dot(data['DC']))
    data['DH'] /= np.sqrt((data['DH'] ** 2).sum())
    data['CD'] /= np.sqrt((data['CD'] ** 2).sum())
    data['DC'] /= np.sqrt((data['DC'] ** 2).sum())
    print(data.to_markdown())
    print('')
    print(data['DH'].dot(data['CD']))
    print(data['DH'].dot(data['DC']))
    print(data['CD'].dot(data['DC']))


def vector_space_ranking():
    N = 5
    cleaned_description = M1.get_and_clean_data()
    cleaned_description = cleaned_description.iloc[:N]
    vectorizer = TfidfVectorizer(preprocessor=preProcess)
    X = vectorizer.fit_transform(cleaned_description)
    print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))
    # print('')
    # query = 'aws github'
    # query = vectorizer.fit_transform(query)


def q_exercise():
    query = 'aws github'
    cleaned_description = M1.get_and_clean_data()
    vectorizer = TfidfVectorizer(preprocessor=preProcess)
    X = vectorizer.fit(cleaned_description)
    query_tran = X.transform([query])
    clean_data_matrix = X.transform(cleaned_description)

    dataframe_query = pd.DataFrame(query_tran.toarray(), columns=vectorizer.get_feature_names()).iloc[0]
    dataframe_data = pd.DataFrame(clean_data_matrix.toarray(), columns=vectorizer.get_feature_names())

    dataframe_query = dataframe_query.divide(np.sqrt((dataframe_query ** 2).sum()))
    dataframe_data = dataframe_data.divide(np.sqrt((dataframe_data ** 2).sum(axis=1)), axis=0)

    dot_prod = dataframe_query.dot(dataframe_data.T)
    rank = dot_prod.argsort()[::-1]
    print(cleaned_description.iloc[rank[:5]].to_markdown())


class BM25(object):
    def __init__(self, vectorizer, b=0.75, k1=1.6):
        self.vectorizer = vectorizer
        self.b = b
        self.k1 = k1

    def fit(self, X):
        # Fit IDF to documents X
        self.vectorizer.fit(X)
        self.y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = self.y.sum(1).mean()

    def transform(self, q):
        # Calculate BM25 between query q and documents X
        b, k1, avdl = self.b, self.k1, self.avdl
        # apply CountVectorizer
        len_y = self.y.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)
        # convert to csc for better column slicing
        y = self.y.tocsc()[:, q.indices]
        denom = y + (k1 * (1 - b + b * len_y / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = y.multiply(np.broadcast_to(idf, y.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1


def display():
    cleaned_description = M1.get_and_clean_data()
    tf_idf_vectorizer = TfidfVectorizer(preprocessor=preProcess)
    bm25 = BM25(tf_idf_vectorizer)
    bm25.fit(cleaned_description)

    score = bm25.transform('aws devops')
    rank = np.argsort(score)[::-1]
    print(cleaned_description.iloc[rank[:5]].to_markdown())

    score = bm25.transform('aws github')
    rank = np.argsort(score)[::-1]
    print(cleaned_description.iloc[rank[:5]].to_markdown())


def TfIdfRanking():
    cleaned_description = M1.get_and_clean_data()
    vectorizer = TfidfVectorizer(preprocessor=preProcess)
    X = vectorizer.fit(cleaned_description)
    cleandatamatrix = X.transform(cleaned_description)
    cleandatamatrix = pd.DataFrame(cleandatamatrix.toarray(), columns=vectorizer.get_feature_names())
    cleandatamatrix = cleandatamatrix.divide(np.sqrt((cleandatamatrix ** 2).sum(axis=1)), axis=0)
    query = input("Enter query: ")
    while query != ['q']:
        query = input("Enter query: ")
        query = [query]
        querytran = X.transform(query)
        querytran = pd.DataFrame(querytran.toarray(), columns=vectorizer.get_feature_names()).iloc[0]
        querytran = querytran.divide(np.sqrt((querytran ** 2).sum()))
        dotpro = querytran.dot(cleandatamatrix.T)
        rank = dotpro.argsort()[::-1]
        print(cleaned_description.iloc[rank[:5]].to_markdown())


def Bm25Ranking():
    cleaned_description = M1.get_and_clean_data()
    vectorizer = TfidfVectorizer(preprocessor=preProcess, ngram_range=(1, 2))
    bm25 = BM25(vectorizer)
    bm25.fit(cleaned_description)
    query = input("Enter query: ")
    while query != 'q':
        query = input("Enter query: ")
        score = bm25.transform(query)
        rank = np.argsort(score)[::-1]
        print(cleaned_description.iloc[rank[:5]].to_markdown())


def tfRanking():
    cleaned_description = M1.get_and_clean_data()
    vectorizer = CountVectorizer(preprocessor=preProcess)
    X = vectorizer.fit(cleaned_description)
    cleandatamatrix = X.transform(cleaned_description)
    cleandatamatrix = pd.DataFrame(cleandatamatrix.toarray(), columns=vectorizer.get_feature_names())
    cleandatamatrix = cleandatamatrix.divide(np.sqrt((cleandatamatrix ** 2).sum(axis=1)), axis=0)
    query = input("Enter query: ")
    while query != ['q']:
        query = input("Enter query: ")
        query = [query]
        querytran = X.transform(query)
        querytran = pd.DataFrame(querytran.toarray(), columns=vectorizer.get_feature_names()).iloc[0]
        querytran = querytran.divide(np.sqrt((querytran ** 2).sum()))
        dotpro = querytran.dot(cleandatamatrix.T)
        rank = dotpro.argsort()[::-1]
        print(cleaned_description.iloc[rank[:5]].to_markdown())