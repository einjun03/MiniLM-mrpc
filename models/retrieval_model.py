print("loading imports...")
from pprint import pprint
from datasets import load_dataset
print("loading sentence transformer imports...")
from .evaluator import CleanInformationRetrievalEvaluator
print("loaded 1/3")
from sentence_transformers import (
    SentenceTransformerTrainer
)
print("loaded 2/3")
from sentence_transformers.losses import OnlineContrastiveLoss, CoSENTLoss
print("loaded 3/3")
from sentence_transformers.evaluation import InformationRetrievalEvaluator


class RetrievalModel:
    LOSS_FUNCTIONS = {
        "CoSENTLoss": lambda model, **kwargs: CoSENTLoss(model=model),
        "OnlineContrastiveLoss": lambda model, **kwargs: OnlineContrastiveLoss(model=model, margin=kwargs.get("margin", 0.5)),
    }

    def __init__(self, model, train_dt, eval_repo, debug=False, seed=42, margin=0.5, use_custom_evaluator=False, loss_fn="CoSENTLoss"):
        self.model = model
        self.debug = debug
        self.use_custom_evaluator = use_custom_evaluator

        # 1. Load Training Data (Augmented Pairs)
        self.train_dt = train_dt
        if self.debug:
            self.train_dt = self.train_dt.shuffle(seed=seed).select(range(50))

        # 2. Store Repo paths for Evaluation
        self.eval_repo = eval_repo

        # 3. Initialize Loss Function
        if loss_fn not in self.LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss function '{loss_fn}'. Must be one of: {list(self.LOSS_FUNCTIONS.keys())}")
        self.loss_fn = self.LOSS_FUNCTIONS[loss_fn](model=self.model, margin=margin)

    def prepare_ir_evaluator(self, split_name):
        """
        Loads retrieval datasets and returns the appropriate evaluator
        based on the use_custom_evaluator flag.
        """
        queries_dt = load_dataset(self.eval_repo, "queries", split=split_name)
        corpus_dt = load_dataset(self.eval_repo, "corpus", split=split_name)
        qrels_dt = load_dataset(self.eval_repo, "qrels", split=split_name)

        queries = {row['id']: row['text'] for row in queries_dt}
        corpus = {row['id']: row['text'] for row in corpus_dt}
        relevant_docs = {row['id']: set(row['matches']) for row in qrels_dt}

        if self.use_custom_evaluator:
            query_exclusions = {row['id']: row['exclude_from_corpus'] for row in queries_dt}
            return CleanInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                query_exclusions=query_exclusions,
                name=f"mrpc-{split_name}-clean-v2",
                main_score_function="cosine"
            )

        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"mrpc-{split_name}-retrieval",
            main_score_function="cosine"
        )

    def train(self, args):
        # Prepare the Val Evaluator
        val_evaluator = self.prepare_ir_evaluator("validation")

        print("--- Pre-train Retrieval Metrics ---")
        pprint(val_evaluator(self.model))

        # Initialize Trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dt,
            loss=self.loss_fn,
            evaluator=val_evaluator, # Trainer will run this every 'eval_strategy'
        )
        
        trainer.train()

    def test(self):
        # Prepare and run the Test Evaluator
        test_evaluator = self.prepare_ir_evaluator("test")
        print("--- Post-train Test Results ---")
        results = test_evaluator(self.model)
        pprint(results)
        return results
    
    @staticmethod
    def diagnose_ranking_failures(model, evaluator, num_queries=5):
        """
        Prints the Top 5 results for queries where the model failed to get Rank #1.
        """
        from sentence_transformers.util import cos_sim

        model.eval()
        # 1. Get embeddings (evaluator.corpus and evaluator.queries are lists)
        corpus_embeddings = model.encode(evaluator.corpus, convert_to_tensor=True)
        query_embeddings = model.encode(evaluator.queries[:num_queries], convert_to_tensor=True)

        # 2. Compute Cosine Similarity
        scores = cos_sim(query_embeddings, corpus_embeddings)

        # 3. Apply exclusions if using custom evaluator
        if hasattr(evaluator, 'query_exclusions'):
            for i, qid in enumerate(evaluator.queries_ids[:num_queries]):
                if qid in evaluator.query_exclusions:
                    for cid_to_exclude in evaluator.query_exclusions[qid]:
                        if cid_to_exclude in evaluator.corpus_ids_map:
                            scores[i][evaluator.corpus_ids_map[cid_to_exclude]] = -1.0

        # 4. Analyze results
        for i, qid in enumerate(evaluator.queries_ids[:num_queries]):
            print(f"\n{'='*50}")
            print(f"QUERY {qid}: {evaluator.queries[i]}")

            gold_cids = evaluator.relevant_docs[qid]
            top_values, top_indices = scores[i].topk(5)

            print("\nTOP 5 RETRIEVED:")
            for rank, (score, idx) in enumerate(zip(top_values, top_indices), 1):
                cid = evaluator.corpus_ids[idx]
                text = evaluator.corpus[idx]
                is_gold = " [GOLD]" if cid in gold_cids else ""
                print(f"  {rank}. [{score:.4f}] {text}{is_gold}")

            if evaluator.corpus_ids[top_indices[0]] not in gold_cids:
                print(f"\n  FAILED: Gold match was not #1.")