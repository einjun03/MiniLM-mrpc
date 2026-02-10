import argparse
from sentence_transformers import SentenceTransformer
from models import RetrievalModel


def main():
    parser = argparse.ArgumentParser(description="Diagnose ranking failures for a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model or HF Hub name")
    parser.add_argument("--eval_repo", type=str, default="ejun26/mrpc-clean-retrieval-v2")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--num_queries", type=int, default=5)
    parser.add_argument("--use_custom_evaluator", action="store_true")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = SentenceTransformer(args.model_path)
    print(f"Model loaded successfully")

    evaluator_type = "CleanInformationRetrievalEvaluator" if args.use_custom_evaluator else "InformationRetrievalEvaluator"
    print(f"\nPreparing {evaluator_type} on '{args.split}' split from {args.eval_repo}...")
    pipeline = RetrievalModel(model, train_dt=None, eval_repo=args.eval_repo,
                              use_custom_evaluator=args.use_custom_evaluator)
    evaluator = pipeline.prepare_ir_evaluator(args.split)
    print(f"Evaluator ready: {len(evaluator.queries)} queries, {len(evaluator.corpus)} corpus docs")

    print(f"\nRunning diagnosis on {args.num_queries} queries...")
    RetrievalModel.diagnose_ranking_failures(model, evaluator, num_queries=args.num_queries)
    print("\nDone.")


if __name__ == "__main__":
    main()
