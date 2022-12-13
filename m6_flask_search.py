from flask import Flask, request
from elasticsearch import Elasticsearch
from m6_pr_solr import Indexer
import pandas as pd
import numpy as np
import time
import pysolr
from m6_pr_manual import Indexer

app = Flask(__name__)
app.es_client = Elasticsearch("http://localhost:9200")
app.solr = pysolr.Solr("http://localhost:8983/solr/simple")
app.manual_indexer = Indexer()
@app.route('/search', methods=['GET'])
def search_elastic():
    start = time.time()
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    query_term=argList['query'][0]
    results = app.es_client.search(index='simple', source_excludes=['url_lists'], size=100,
        query={
            "match":
                {"text": query_term},
        })
    end = time.time()
    total_hit = results['hits']['total']['value']
    results_df = pd.DataFrame([[hit["_source"]['title'], hit["_source"]['url'], hit["_source"]
['text'][:100], hit["_score"]] for hit in results['hits']['hits']], columns=['title', 'url', 'text',
'score'])

    response_object['total_hit'] = total_hit
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start

    return response_object

@app.route('/search_solr', methods=['GET'])
def search_solr():
    start = time.time()
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    query_term=argList['query'][0]
    results = app.solr.search('text:'+query_term, **{'fl' : '*, score'})
    end = time.time()
    results_df = pd.DataFrame(np.hstack([[result['title'] for result in results], [result['url'] for result in results],
                                        [[result['text'][0][:100]] for result in results], [[result['score']] for result in results]]),
                              columns=['title', 'url', 'text', 'score'])

    response_object['total_hit'] = results.hits
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start

    return response_object

@app.route('/search_manual', methods=['GET'])
def search_manual():
    start = time.time()
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    results = app.manual_indexer.query(query_term)
    end = time.time()
    total_hit = len(results)
    results_df = results

    response_object['total_hit'] = total_hit
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start

    return response_object

if __name__ == '__main__':
    app.run(debug=True)
