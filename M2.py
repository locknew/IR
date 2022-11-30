from scipy.sparse import csr_matrix, dok_matrix, lil_matrix, coo_matrix, csc_matrix

from M1 import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import timeit


def cleandataframe():
    cleaned_description = get_and_clean_data()
    cleaned_description = cleaned_description

    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
    stop_dict = {s: 1 for s in stopwords.words()}
    sw_removed_description = tokenized_description.apply(lambda s: [word for word in s if word not in
                                                                    stop_dict])
    sw_removed_description = tokenized_description.apply(lambda s: [word for word in s if word not in
                                                                    stopwords.words()])
    sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if
                                                                     len(word) > 2])

    ps = PorterStemmer()
    stemmed_description = sw_removed_description.apply(lambda s: [ps.stem(w) for w in s])

    cv = CountVectorizer(analyzer=lambda x: x)
    X = cv.fit_transform(stemmed_description)
    print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names()))


print('time of todok', timeit.timeit(lambda: X.todok() * X.T.todok(), number=1))
print('time of tolil', timeit.timeit(lambda: X.tolil() * X.T.tolil(), number=1))
print('time of tocoo', timeit.timeit(lambda: X.tocoo() * X.T.tocoo(), number=1))
print('time of tocsc', timeit.timeit(lambda: X.tocsc() * X.T.tocsc(), number=1))

B = X.todense()
times = 100

compCsr = (timeit.timeit(lambda: csr_matrix(B), number=times) / times)
compDok = (timeit.timeit(lambda: dok_matrix(B), number=times) / times)
compLil = (timeit.timeit(lambda: lil_matrix(B), number=times) / times)
compCoo = (timeit.timeit(lambda: coo_matrix(B), number=times) / times)
compCsc = (timeit.timeit(lambda: csc_matrix(B), number=times) / times)

print('comCsr :', compCsr)
print('comDok :', compDok)
print('comLil :', compLil)
print('coCoo :', compCoo)
print('comCsc :', compCsc)
