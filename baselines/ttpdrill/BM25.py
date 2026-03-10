import numpy as np
from rank_bm25 import BM25Okapi as _BM25Okapi, BM25L as _BM25L, BM25Plus as _BM25Plus

class BM25Okapi:
    def __init__(self, corpus):
        self.bm25 = _BM25Okapi(corpus)
        self.corpus = corpus

    def get_top_n(self, query, ontology_list, n=5):
        # query is expected to be a list of tokens (bow)
        scores = self.bm25.get_scores(query)
        # Handle cases where scores might be all zeros or empty
        if len(scores) == 0:
            return [], [], []
        
        # Get indices of top n scores
        # argsort gives ascending, so [::-1] for descending
        top_n_indices = np.argsort(scores)[::-1][:n]
        
        # Ensure we don't index out of bounds of ontology_list if it's different from corpus
        # TTPDrill seems to pass ontology_list here
        match_ttp = [ontology_list[i] if i < len(ontology_list) else None for i in top_n_indices]
        match_scores = [scores[i] for i in top_n_indices]
        
        return top_n_indices, match_ttp, match_scores

class BM25L(BM25Okapi):
    def __init__(self, corpus):
        self.bm25 = _BM25L(corpus)
        self.corpus = corpus

class BM25Plus(BM25Okapi):
    def __init__(self, corpus):
        self.bm25 = _BM25Plus(corpus)
        self.corpus = corpus

# Alias BM25 to BM25Okapi for compatibility
BM25 = BM25Okapi
