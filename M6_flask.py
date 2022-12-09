from datetime import time
from urllib import request

import pandas as pd
from nltk import app

from M6_elastic import Indexer
from flask import Flask, request

app = Flask(__name__)
app.indexer = Indexer()


class Flask:
    @app.route('/search', methods=['GET'])
    def search():
        start = time.time()
        response_object = {'status': 'success'}
        argList = request.args.to_dict(flat=False)
        query_term = argList['query'][0]
        results = app.indexer.es_client.search(index='simple', source_excludes=['url_lists'], size=100,
                                               query={"match": {"text": query_term}})
        end = time.time()
        total_hit = results['hits']['total']['value']
        results_df = pd.DataFrame(
            [[hit["_source"]['title'], hit["_source"]['url'], hit["_source"]['text'][:100], hit["_score"]] for hit in
             results['hits']['hits']], columns=['title', 'url', 'text', 'score'])

        response_object['total_hit'] = total_hit
        response_object['results'] = results_df.to_dict('records')
        response_object['elapse'] = end - start

        return response_object

    pass


if __name__ == '__main__':
    app.run(debug=True)
