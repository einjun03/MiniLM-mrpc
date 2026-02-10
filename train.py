import argparse
import os
import wandb
from models import RetrievalModel
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainingArguments
)
from datasets import load_dataset

def setup_wandb(run_name=None):
    os.environ["WANDB_PROJECT"] = "minilm-mrpc-retrieval"
    # Allow custom run names for better experiment tracking
    wandb.init(project="minilm-mrpc-retrieval", name=run_name)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM on MRPC with IR Evaluation")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for OnlineContrastiveLoss")
    parser.add_argument("--output_dir", type=str, default="../models/finetune-mrpc-ir")
    
    # Eval & Save Strategy
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_limit", type=int, default=2)
    
    # Dataset Repos
    parser.add_argument("--train_repo", type=str, default="nyu-mll/glue") 
    parser.add_argument("--eval_repo", type=str, default="ejun26/mrpc-clean-retrieval-v2")
    
    # Debug & Tracking
    parser.add_argument("--debug", action="store_true", help="Run a fast sanity check with tiny data")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for WandB run")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")

    # Hugging Face Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Push the trained model to Hugging Face Hub")
    parser.add_argument("--hub_model_name", type=str, default=None, help="Repo name on HF Hub (e.g. 'username/model-name')")

    args_cmd = parser.parse_args()

    # 1. Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Configure Training Arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args_cmd.output_dir,
        num_train_epochs=args_cmd.epochs,
        learning_rate=args_cmd.lr,
        per_device_train_batch_size=args_cmd.batch_size,
        warmup_ratio=args_cmd.warmup_ratio,
        
        # Strategy: Evaluate and Save every X steps instead of every epoch
        eval_strategy="steps",
        eval_steps=args_cmd.eval_steps,
        save_strategy="steps",
        save_steps=args_cmd.eval_steps,
        save_total_limit=args_cmd.save_limit,
        
        # Checkpointing logic
        load_best_model_at_end=True,
        metric_for_best_model="mrpc-validation-retrieval_cosine_mrr@10",
        greater_is_better=True,  # Crucial: MRR is "higher is better"
        
        report_to="wandb" if args_cmd.use_wandb else "none",
        run_name=args_cmd.run_name,
        logging_steps=10,
    )

    # 3. Load ORIGINAL training data
    print(f"Loading original training data from {args_cmd.train_repo}...")
    train_ds = load_dataset(args_cmd.train_repo, "mrpc", split="train")

    rename={"sentence1": "text1", "sentence2": "text2"}
    train_ds = train_ds.rename_columns(rename).remove_columns(['idx'])

    # Ensure labels are floats for OnlineContrastiveLoss
    train_ds = train_ds.map(lambda x: {"label": float(x["label"])})

    if args_cmd.debug:
        print("🛠️ DEBUG MODE: Shuffling and selecting 100 samples.")
        train_ds = train_ds.shuffle(seed=42).select(range(100))
        training_args.num_train_epochs = 1
    if args_cmd.use_wandb and not args_cmd.debug:
        setup_wandb(args_cmd.run_name)

    # 4. Initialize Pipeline and Start Training
    # (The RetrievalModel uses your custom IR evaluator internally)
    pipeline = RetrievalModel(model, train_ds, args_cmd.eval_repo, debug=args_cmd.debug, margin=args_cmd.margin)
    
    print("Starting Training...")
    pipeline.train(training_args)
    
    print("Running Final Test...")
    pipeline.test()

    # 5. Push to Hugging Face Hub
    if args_cmd.push_to_hub:
        if not args_cmd.hub_model_name:
            raise ValueError("--hub_model_name is required when using --push_to_hub")
        print(f"Pushing model to Hugging Face Hub: {args_cmd.hub_model_name}")
        model.push_to_hub(args_cmd.hub_model_name)

if __name__ == "__main__":
    main()