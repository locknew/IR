import itertools
import pickle
import re
import string
from multiprocessing.pool import ThreadPool as Pool

from sklearn import model_selection, metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import lightgbm as lgb
import optuna
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from flask import Flask, request
from scipy.sparse import hstack
import pickle

app = Flask(__name__)
# app.vecterizer = pickle.load(open('resource/github_bug_prediction_vectorizer.pkl', 'rb'))
# app.model = pickle.load(open('resource/github_bug_prediction_model.pkl', 'rb'))


import M1


def preprocess(text, stopword_set, stemmer):
    cleaned_text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,.<=>?@[]^`{|}~' + u'\xa0'))
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))

    cleaned_text = ' '.join(['_variable_with_underscore' if '_' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_variable_with_dash' if '-' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_long_variable_name' if len(t) > 10 and t[0] != '#' else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_weburl' if t.startswith('http') and '/' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_number' if re.sub('[\\/;:_-]', '', t).isdigit() else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_base16_number' if re.match('[0-9a-f].*', t) else t for t in cleaned_text.split()])
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


dataset = pd.read_json('resource/embold_train.json')
dataset.loc[dataset['label'] > 0, 'label'] = 1
stopword_set = set(stopwords.words())
stemmer = M1.PorterStemmer()
pool = Pool(6)

# cleaned_title = pool.starmap(preprocess, zip(dataset.title, itertools.repeat(stopword_set), itertools.repeat(stemmer)))
# cleaned_body = pool.starmap(preprocess, zip(dataset.body, itertools.repeat(stopword_set), itertools.repeat(stemmer)))
dataset['combined'] = dataset['title'] + '. ' + dataset['body']
cleaned_title = pool.starmap(preprocess, zip(dataset.title, itertools.repeat(stopword_set), itertools.repeat(stemmer)))
cleaned_body = pool.starmap(preprocess, zip(dataset.body, itertools.repeat(stopword_set), itertools.repeat(stemmer)))
combine = pool.starmap(preprocess, zip(dataset.combined, itertools.repeat(stopword_set), itertools.repeat(stemmer)))
data_texts = pd.DataFrame([cleaned_title, cleaned_body, combine], index=['title', 'body', 'combine']).T

y = dataset['label']

data_fit, data_blindtest, y_fit, y_blindtest = model_selection.train_test_split(data_texts, y,
                                                                                test_size=1)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
tfidf_vectorizer.fit(cleaned_title + cleaned_body)

X_tfidf_fit = tfidf_vectorizer.transform(data_fit['combine'])
X_tfidf_blindtest = tfidf_vectorizer.transform(data_blindtest['combine'])

gbm_model = lgb.LGBMClassifier()

data_fit_train, data_fit_test, y_fit_train, y_fit_test = model_selection.train_test_split(data_fit, y_fit,
                                                                                          test_size=1)
X_tfidf_fit_train = tfidf_vectorizer.transform(data_fit_train['combine'])
X_tfidf_fit_test = tfidf_vectorizer.transform(data_fit_test['combine'])
X_tfidf_blindtest = tfidf_vectorizer.transform(data_blindtest['combine'])


def objective(trial):
    dtrain = lgb.Dataset(X_tfidf_fit_train, label=y_fit_train)
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "dart",
        'num_boost_round': 300,
        'is_unbalance': True,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 25),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.2, 0.5),
        "max_depth": trial.suggest_int("max_depth", 2, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "max_bin": trial.suggest_int("max_bin", 2, 64),
        "min_data_in_leaf": 20
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_tfidf_fit_test)
    pred_labels = np.rint(preds)
    accuracy = metrics.roc_auc_score(y_fit_test, pred_labels)
    return accuracy
    #


def optuna_cv():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    trial = study.best_trial
    gbm_model = lgb.LGBMClassifier(**trial.params)
    precision_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                                         scoring='precision_macro').mean()
    recall_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                                      scoring='recall_macro').mean()
    f1_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                                  scoring='f1_macro').mean()
    print('CV: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score, f1_cv_score))
    gbm_model.fit(X_tfidf_fit_train, y_fit_train, eval_set=[(X_tfidf_fit_test, y_fit_test)],
                  eval_metric='AUC')
    precision_test_score = metrics.precision_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                                   average='macro')
    recall_test_score = metrics.recall_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                             average='macro')
    f1_test_score = metrics.f1_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                     average='macro')
    print('test: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_test_score, recall_test_score, f1_test_score))
    # pickle.dump(tfidf_vectorizer, open('resources/github_bug_prediction_tfidf_vectorizer_assignment.pkl', 'wb'))
    # pickle.dump(gbm_model, open('resources/github_bug_prediction_assignment_model.pkl', 'wb'))
