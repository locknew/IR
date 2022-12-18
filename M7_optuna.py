
import optuna
import lightgbm as lgb

from IR.M7 import X_tfidf_fit_train, y_fit_train, X_tfidf_fit_test, y_fit_test, X_tfidf_fit, y_fit, X_tfidf_blindtest, \
    y_blindtest
import numpy as np
import pandas as pd
from sklearn import model_selection, metrics

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
trial = study.best_trial
gbm_model = lgb.LGBMClassifier(trial.params)
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