import itertools
import pickle
import string
from multiprocessing.pool import ThreadPool as Pool
import re
import lightgbm as lgb
import pandas as pd
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from optuna.multi_objective import trial
from sklearn import model_selection, metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack

dataset = pd.read_json('resource/embold_train.json')
dataset.loc[dataset['label'] > 0, 'label'] = 1
stopword_set = set(stopwords.words())
stemmer = PorterStemmer()
pool = Pool(8)


def preprocess(text, stopword_set, stemmer):
    cleaned_text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,.<=>?@[]^`{|}~' + u'\xa0'))
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))

    cleaned_text = ' '.join(['_variable_with_underscore' if '_' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_variable_with_dash' if '-' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_long_variable_name' if len(t) > 8 and t[0] != '#' else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_weburl' if t.startswith('http') and '/' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_number' if re.sub('[\\/;:_-]', '', t).isdigit() else t for t in cleaned_text.split()])
    # cleaned_text = ' '.join(['_base16_number' if re.match('[0-9a-f].*', t) else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_variable_with_address' if re.match('.*0x[0-9a-f].*', t) else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_name_with_number' if re.match('.*[a-f]*:[0-9]*', t) else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_number_starts_with_one_character' if re.match('[a-f][0-9].*', t) else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_number_starts_with_three_characters' if re.match('[a-f]{3}[0-9].*', t) else t for t
                             in cleaned_text.split()])
    cleaned_text = ' '.join(['_version' if any(i.isdigit() for i in t) and t.startswith('v') else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_localpath' if ('\\' in t or '/' in t) and ':' not in t else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_image_size' if t.endswith('px') else t for t in cleaned_text.split()])
    tokenized_text = word_tokenize(cleaned_text)
    sw_removed_text = [word for word in tokenized_text if word not in stopword_set]
    sw_removed_text = [word for word in sw_removed_text if len(word) > 2]
    stemmed_text = ' '.join([stemmer.stem(w) for w in sw_removed_text])
    return stemmed_text


cleaned_title = pool.starmap(preprocess, zip(dataset.title, itertools.repeat(stopword_set),
                                             itertools.repeat(stemmer)))
cleaned_body = pool.starmap(preprocess, zip(dataset.body, itertools.repeat(stopword_set),
                                            itertools.repeat(stemmer)))

data_texts = pd.DataFrame([cleaned_title, cleaned_body], index=['title', 'body']).T
y = dataset['label']
# ------------------------------------------------------------------------------------------------------------------#
data_fit, data_blindtest, y_fit, y_blindtest = model_selection.train_test_split(data_texts, y, test_size=0.1)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
tfidf_vectorizer.fit(cleaned_title + cleaned_body)
X_tfidf_fit = tfidf_vectorizer.transform(data_fit['title'])
X_tfidf_blindtest = tfidf_vectorizer.transform(data_blindtest['title'])

gbm_model = lgb.LGBMClassifier()
precision_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5,
                                                     n_jobs=-2, scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                                  scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()
print('CV: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))

# -----------------------------------------------------------------------------------------------------#
data_fit_train, data_fit_test, y_fit_train, y_fit_test = model_selection.train_test_split(data_fit, y_fit,
                                                                                          test_size=0.3)
X_tfidf_fit_train = tfidf_vectorizer.transform(data_fit_train['title'])
X_tfidf_fit_test = tfidf_vectorizer.transform(data_fit_test['title'])
X_tfidf_blindtest = tfidf_vectorizer.transform(data_blindtest['title'])

gbm_model.fit(X_tfidf_fit_train, y_fit_train, eval_set=[(X_tfidf_fit_test, y_fit_test)],
              eval_metric='AUC')
precision_test_score = metrics.precision_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                               average='macro')
recall_test_score = metrics.recall_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                         average='macro')
f1_test_score = metrics.f1_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                 average='macro')
print('test: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_test_score, recall_test_score,
                                                   f1_test_score))
pickle.dump(tfidf_vectorizer, open('resource/github_bug_prediction_tfidf_vectorizer.pkl', 'wb'))
pickle.dump(gbm_model, open('resource/github_bug_prediction_model.pkl', 'wb'))
# ----------------------------------------------LSA------------------------------------------------------------#
# from sklearn.decomposition import TruncatedSVD
#
# lsa = TruncatedSVD(n_components=100, n_iter=100, random_state=0)
# lsa.fit(X_tfidf_fit)
# X_lsa_fit = lsa.transform(X_tfidf_fit)
#
# gbm_model_with_lsa = lgb.LGBMClassifier()
#
# precision_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5,
#                                                      n_jobs=-2, scoring='precision_macro').mean()
# recall_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5,
#                                                   n_jobs=-2, scoring='recall_macro').mean()
# f1_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_lsa_fit, y_fit, cv=5, n_jobs=-2,
#                                               scoring='f1_macro').mean()
#
# print('fit: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))
#
# X_fit_with_lsa = hstack([X_tfidf_fit, X_lsa_fit]).tocsr()
#
# precision_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5,
#                                                      n_jobs=-2, scoring='precision_macro').mean()
# recall_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5,
#                                                   n_jobs=-2, scoring='recall_macro').mean()
# f1_cv_score = model_selection.cross_val_score(gbm_model_with_lsa, X_fit_with_lsa, y_fit, cv=5,
#                                               n_jobs=-2, scoring='f1_macro').mean()
#
# print('fit: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))
# pickle.dump(tfidf_vectorizer, open('resource/github_bug_prediction_tfidf_vectorizer.lsa', 'wb'))
# pickle.dump(gbm_model, open('resource/github_bug_prediction_model.lsa', 'wb'))
# --------------------------------------------------LDA---------------------------------------------------------#
from sklearn.decomposition import LatentDirichletAllocation

count_vectorizer = CountVectorizer(ngram_range=(1, 1))
count_vectorizer.fit(cleaned_title + cleaned_body)
X_tf_fit = count_vectorizer.transform(data_fit['title'])
X_tf_blindtest = count_vectorizer.transform(data_blindtest['title'])
lda = LatentDirichletAllocation(n_components=100, random_state=0)
lda.fit(X_tf_fit)
X_lda_fit = lda.transform(X_tf_fit)
gbm_model_with_lda = lgb.LGBMClassifier()

precision_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_lda_fit, y_fit, cv=5,
                                                     n_jobs=-2, scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_lda_fit, y_fit, cv=5,
                                                  n_jobs=-2, scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_lda_fit, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()

print('fit: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))

X_fit_with_lda = hstack([X_tfidf_fit, X_lda_fit]).tocsr()

precision_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_fit_with_lda, y_fit, cv=5,
                                                     n_jobs=-2, scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_fit_with_lda, y_fit, cv=5,
                                                  n_jobs=-2, scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_fit_with_lda, y_fit, cv=5,
                                              n_jobs=-2, scoring='f1_macro').mean()

print('fit: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))
pickle.dump(tfidf_vectorizer, open('resource/github_bug_prediction_tfidf_vectorizer.lda', 'wb'))
pickle.dump(gbm_model, open('resource/github_bug_prediction_model.lda', 'wb'))
# ------------------------------------------------optuna-------------------------------------------------------------#
import optuna



def objective(trial):
    dtrain = lgb.Dataset(X_tfidf_fit_train, label=y_fit_train)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_tfidf_fit_test)
    pred_labels = np.rint(preds)
    accuracy = metrics.roc_auc_score(y_fit_test, pred_labels)
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
gbm_model = lgb.LGBMClassifier(**trial.params)
precision_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5,
                                                     n_jobs=-2, scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                                  scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()
print('CV: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score,
                                                 f1_cv_score))
gbm_model.fit(X_tfidf_fit_train, y_fit_train, eval_set=[(X_tfidf_fit_test, y_fit_test)],
              eval_metric='AUC')
precision_test_score = metrics.precision_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                               average='macro')
recall_test_score = metrics.recall_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                         average='macro')
f1_test_score = metrics.f1_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                 average='macro')
print('test optuna: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_test_score, recall_test_score,
                                                          f1_test_score))
# --------------------------------------------------------------------------------------------------------------#
