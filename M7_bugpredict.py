from flask import Flask, request
from nltk import PorterStemmer, app
from nltk.corpus import stopwords
from scipy.sparse import hstack
from M7 import preprocess
import pickle


class Flask:
    app = Flask(__name__)
    app.vecterizer = pickle.load(open('resource/github_bug_prediction_tfidf_vectorizer.pkl', 'rb'))
    app.model = pickle.load(open('resource/github_bug_prediction_model.pkl', 'rb'))
    app.stopword_set = set(stopwords.words())
    app.stemmer = PorterStemmer()

    @app.route('/predict', methods=['GET'])
    def search():
        response_object = {'status': 'success'}
        argList = request.args.to_dict(flat=False)
        title = argList['title'][0]
        body = argList['body'][0]
        predict = app.model.predict_proba(
            hstack([app.vecterizer.transform([preprocess(title, app.stopword_set, app.stemmer)])]))
        response_object['predict_as'] = 'bug' if 1 - predict[0][1] >= 0.5 else 'not bug'
        response_object['bug_prob'] = 1 - predict[0][1]
        return response_object

    pass


if __name__ == '__main__':
    app.run(debug=True)
