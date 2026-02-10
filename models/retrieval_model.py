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
from sentence_transformers.losses import OnlineContrastiveLoss
print("loaded 3/3")
#from sentence_transformers.evaluation import InformationRetrievalEvaluator


class RetrievalModel:
    def __init__(self, model, train_dt, eval_repo, debug=False, seed=42, margin=0.5):
        self.model = model
        self.debug = debug
        
        # 1. Load Training Data (Augmented Pairs)
        # Using your 'training' config which has sentence1, sentence2, label
        self.train_dt = train_dt
        if self.debug:
            self.train_dt = self.train_dt.shuffle(seed=seed).select(range(50))

        # 2. Store Repo paths for Evaluation
        self.eval_repo = eval_repo
        
        # 3. Initialize Loss Function
        # OnlineContrastiveLoss is great for pairs; it effectively mines hard negatives within the batch
        self.loss_fn = OnlineContrastiveLoss(model=self.model, margin=margin)

    def prepare_ir_evaluator(self, split_name):
        """
        Loads the 'v2 clean-retrieval' datasets and uses the 
        custom CleanInformationRetrievalEvaluator.
        """
        # Load the 3 parts of the v2 config
        queries_dt = load_dataset(self.eval_repo, "queries", split=split_name)
        corpus_dt = load_dataset(self.eval_repo, "corpus", split=split_name)
        qrels_dt = load_dataset(self.eval_repo, "qrels", split=split_name)

        # Format 1: Dict[qid, query_text]
        queries = {row['id']: row['text'] for row in queries_dt}
        
        # Format 2: Dict[cid, corpus_text]
        corpus = {row['id']: row['text'] for row in corpus_dt}
        
        # Format 3: Dict[qid, Set[cid]]
        relevant_docs = {row['id']: set(row['matches']) for row in qrels_dt}
        
        # Format 4: NEW - Dict[qid, List[cid]] for exclusions
        query_exclusions = {row['id']: row['exclude_from_corpus'] for row in queries_dt}

        return CleanInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            query_exclusions=query_exclusions,
            name=f"mrpc-{split_name}-clean-v2",
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