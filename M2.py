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




