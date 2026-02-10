import logging
import torch
from contextlib import nullcontext
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import numpy as np 

logger = logging.getLogger(__name__)

class CleanInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    def __init__(self, queries, corpus, relevant_docs, query_exclusions, **kwargs):
        super().__init__(queries, corpus, relevant_docs, **kwargs)
        self.corpus_ids_map = {cid: idx for idx, cid in enumerate(self.corpus_ids)}
        
        # Ensure every query excludes itself from the haystack
        self.query_exclusions = {}
        for qid, exclusions in query_exclusions.items():
            self.query_exclusions[qid] = list(exclusions)
            # Self-exclusion logic: if the query exists in the corpus, ignore it
            if qid in self.corpus_ids_map and qid not in self.query_exclusions[qid]:
                self.query_exclusions[qid].append(qid)

    def compute_metrics(self, model, corpus_model=None, corpus_embeddings=None):
        """
        Calculates the IR metrics. The internal attributes for strings are 
        'queries_list' and 'corpus_list'.
        """
        if corpus_model is None:
            corpus_model = model

        # 1. Encode Queries
        # Note: self.queries_list is the internal list of strings
        query_embeddings = model.encode(
            self.queries_list, 
            show_progress_bar=self.show_progress_bar, 
            batch_size=self.batch_size, 
            convert_to_tensor=True
        )

        # 2. Encode Corpus
        if corpus_embeddings is None:
            # Note: self.corpus_list is the internal list of strings
            corpus_embeddings = corpus_model.encode(
                self.corpus_list, 
                show_progress_bar=self.show_progress_bar, 
                batch_size=self.batch_size, 
                convert_to_tensor=True
            )

        # 3. Compute Similarity & Mask Exclusions
        from sentence_transformers.util import cos_sim, dot_score
        score_func = cos_sim if self.main_score_function == 'cosine' else dot_score
        scores = score_func(query_embeddings, corpus_embeddings)

        for i, qid in enumerate(self.queries_ids):
            if qid in self.query_exclusions:
                for cid_to_exclude in self.query_exclusions[qid]:
                    if cid_to_exclude in self.corpus_ids_map:
                        idx = self.corpus_ids_map[cid_to_exclude]
                        scores[i][idx] = -float('inf')

        # 4. Use the parent's logic to calculate the final dict of metrics
        return self.compute_metrics_from_scores(scores)

    def compute_metrics_from_scores(self, scores):
        """
        Utility to turn the masked scores back into the standard MRR/Accuracy dict.
        """
        scores = scores.cpu().numpy()
        query_results = {}
        max_k = max(max(self.mrr_at_k), max(self.accuracy_at_k))

        for i, qid in enumerate(self.queries_ids):
            # Find Top-K indices
            top_hits = np.argpartition(scores[i], -max_k)[-max_k:]
            top_hits = top_hits[np.argsort(scores[i][top_hits])][::-1]
            
            query_results[qid] = [
                {'corpus_id': self.corpus_ids[idx], 'score': scores[i][idx]} 
                for idx in top_hits
            ]
            
        return self.calculate_metrics(query_results)