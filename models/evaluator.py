import logging
from contextlib import nullcontext

import torch
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logger = logging.getLogger(__name__)


class CleanInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    def __init__(self, queries, corpus, relevant_docs, query_exclusions, **kwargs):
        super().__init__(queries, corpus, relevant_docs, **kwargs)
        self.query_exclusions = query_exclusions
        self.corpus_ids_map = {cid: idx for idx, cid in enumerate(self.corpus_ids)}

    def compute_metrices(self, model, corpus_model=None, corpus_embeddings=None):
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Encode queries
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            query_embeddings = model.encode(
                self.queries,
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )

        # Encode full corpus
        if corpus_embeddings is None:
            with nullcontext() if self.truncate_dim is None else corpus_model.truncate_sentence_embeddings(self.truncate_dim):
                corpus_embeddings = corpus_model.encode(
                    self.corpus,
                    show_progress_bar=self.show_progress_bar,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                )

        queries_result_list = {}
        for name, score_function in self.score_functions.items():
            pair_scores = score_function(query_embeddings, corpus_embeddings)

            # Apply exclusions: set excluded corpus scores to -inf before ranking
            for i, qid in enumerate(self.queries_ids):
                if qid in self.query_exclusions:
                    for cid_to_exclude in self.query_exclusions[qid]:
                        if cid_to_exclude in self.corpus_ids_map:
                            pair_scores[i][self.corpus_ids_map[cid_to_exclude]] = -float('inf')

            # Get top-k results
            pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
            )
            pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
            pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

            queries_result_list[name] = []
            for query_itr in range(len(query_embeddings)):
                queries_result_list[name].append([
                    {"corpus_id": self.corpus_ids[idx], "score": score}
                    for idx, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr])
                ])

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        # Use parent's compute_metrics for MRR, NDCG, etc.
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])

        return scores
