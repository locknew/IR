from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import M1
from M1 import *
from bm25 import *


def preProcess(s):
    ps = PorterStemmer()
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
    query = vectorizer.transform(['good at java and python'])
    print(query)
    print(vectorizer.inverse_transform(query))

sk_vectorize()
query = vectorizer.transform(['good at java and python'])
print(query)
print(vectorizer.inverse_transform(query))

# slide 22 #
def tf_idfweight():
    N = 5
    cleaned_description = M1.get_and_clean_data()
    cleaned_description = cleaned_description.iloc[:N]
    vectorizer = CountVectorizer(preprocessor=preProcess)
    X = vectorizer.fit_transform(cleaned_description)
    print(X.toarray())

    df = np.array((X.todense() > 0).sum(0))[0]
    idf = np.log10(N / df)
    tf = np.log10(X.todense() + 1)
    tf_idf = np.multiply(tf, idf)
    X = sparse.csr_matrix(tf_idf)

    print(X.toarray())
    print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))


N = 5
cleaned_description = M1.get_and_clean_data()
cleaned_description = cleaned_description.iloc[:N]
tf_idf_vectorizer = TfidfVectorizer(preprocessor=preProcess, ngram_range=(1, 2), use_idf=True)
X = tf_idf_vectorizer.fit(cleaned_description)
transformed_X = X.transform(cleaned_description)
print(pd.DataFrame(transformed_X.toarray(), columns=tf_idf_vectorizer.get_feature_names()))


""" slide 81 pull from bm25"""
cleaned_description = M1.get_and_clean_data()
bm25 = BM25(tf_idf_vectorizer)
bm25.fit(cleaned_description)

score = bm25.transform('aws devops')
rank = np.argsort(score)[::-1]
print(cleaned_description.iloc[rank[:5]].to_markdown())

score = bm25.transform('aws github')
rank = np.argsort(score)[::-1]
print(cleaned_description.iloc[rank[:5]].to_markdown())