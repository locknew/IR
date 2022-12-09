import pysolr
import json
import os
import pickle
from pathlib import Path


# from M6_crawled_pr import PR
class Indexer:
    def __init__(self):
        self.crawled_folder = Path(os.path.abspath('')).parent / '../crawled/'
        with open(self.crawled_folder / 'url_list.pickle', 'rb') as f:
            self.file_mapper = pickle.load(f)
        self.solr = pysolr.Solr('http://localhost:8983/solr/simple', always_commit=True, timeout=10)

    def run_indexer(self):
        self.solr.delete(q='*:*')
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                j['id'] = j['url']
                print(j)
                self.solr.add(j)

    pass


if __name__ == '__main__':
    s = Indexer()
    s.run_indexer()
    results = s.solr.search('text:camt')
    for result in results:
        print("The title is '{0} ({1})'.".format(result['title'], result['url']))