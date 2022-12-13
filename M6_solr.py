
import json
import os
import pickle
from pathlib import Path
import pysolr as pysolr
from IR.M6_pagerank import Pr


# from M6_crawled_pr import PR
class Indexer:
    def __init__(self):
        self.crawled_folder = Path(os.path.abspath('')).parent / '../crawled/'
        with open(self.crawled_folder / 'url_list.pickle', 'rb') as f:
            self.file_mapper = pickle.load(f)
        self.solr = pysolr.Solr('http://localhost:8983/solr/simple', always_commit=True, timeout=10)

    def run_indexer(self):
        self.pr = Pr(alpha=0.85)
        self.pr.pr_calc()
        self.solr.delete(q='*:*')
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                j['id'] = j['url']
                j['pagerank'] = self.pr.pr_result.loc[j['id']].score
                print(j)
                self.solr.add(j)

    pass


if __name__ == '__main__':
    s = Indexer()
    s.run_indexer()
    results = s.solr.search('text:camt', **{'defType': 'edismax',
                                            'boost': 'mul(query($q),field(pagerank,min))'})
    for result in results:
        print("The title is '{0} ({1})'.".format(result['title'], result['url']))