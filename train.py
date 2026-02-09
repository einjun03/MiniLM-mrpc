print("loading imports...")
import wandb
from datasets import load_dataset
import os
from pprint import pprint
#import numpy as np
print("loading sentence transformer imports...")
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
#from scipy.stats import spearmanr
print("loaded 1/3")
from sentence_transformers.losses import CoSENTLoss, CosineSimilarityLoss
print("loaded 2/3")
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
print("loaded 3/3")

class EmbeddingModel:
    
    def __init__(self, model, dataset, loss='CoSENTLoss', debug=False, seed=42):

        self.dataset = dataset

        self.train_dt = self.create_dataset('train')

        self.debug = debug
        if self.debug:
            #running to sanity check
            self.train_dt = self.train_dt.shuffle(seed=seed).select(range(50))
        
        self.test_dt = self.create_dataset('test')
        self.val_dt = self.create_dataset('validation')

        self.model = model
        if loss == 'CoSENTLoss':
            self.loss_fn = CoSENTLoss(self.model)
        elif loss == 'CosineSimilarityLoss':
            self.loss_fn = CosineSimilarityLoss(self.model)
        else:
            raise Exception("Invalid Loss Function")

    def create_dataset(self, split, rename={"sentence1": "text1", "sentence2": "text2"}):
        dt = load_dataset(*self.dataset, split=split)
        return dt.rename_columns(rename).remove_columns(['idx'])

    def train(self, args):
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=self.val_dt['text1'],
            sentences2=self.val_dt['text2'],
            scores=self.val_dt['label'],
            name='mrpc-eval'
        )

        #evaluate on the val set before training
        print("Pre-train val set results:")
        #os.makedirs("./val_logs", exist_ok=True)  # ← Create directory
        res = evaluator(self.model)
        pprint(res)

        # 7. Create a trainer & train
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dt,
            eval_dataset=self.val_dt,
            loss=self.loss_fn,
            evaluator=evaluator,
        )
        trainer.train()

    def eval(self, stage="pretrain"):
        os.makedirs("./test_logs", exist_ok=True)  # ← Create directory
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=self.test_dt['text1'],
            sentences2=self.test_dt['text2'],
            scores=self.test_dt['label'],
            name=f"mrpc-test-{stage}"
        )

        pprint("Test set results:")
        #evaluate on the test set before training
        res = evaluator(self.model, output_path="./test_logs")
        pprint(res)

def setup_wandb():
    os.environ["WANDB_PROJECT"]="minilm-finetune"
    os.environ["WANDB_LOG_MODEL"]="true"
    os.environ["WANDB_WATCH"]="false"

if __name__ == '__main__':
    #print("loading dataset...")
    #ds = load_dataset("SetFit/mrpc")
    #print("loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("loaded model")

    debugging = False

    if debugging: 
        args = SentenceTransformerTrainingArguments(
            output_dir="models/debug/finetuned-mrpc",
            num_train_epochs=20,
            per_device_train_batch_size=8, # number of samples/batch
            learning_rate=1e-4,
            warmup_ratio=0.0,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=3, #log for each batch
            report_to="none"
        )
    else:
        args = SentenceTransformerTrainingArguments(
            output_dir="models/finetune-mrpc",
            report_to="wandb",
            num_train_epochs=5,
            per_device_train_batch_size=32,
            learning_rate=2e-5,
            warmup_ratio=0.1
        )
    
    setup_wandb()

    ds = ["nyu-mll/glue", "mrpc"]
    embd_model = EmbeddingModel(model, ds, debug=debugging)
    embd_model.eval()
    embd_model.train(args)
    embd_model.eval(stage="post-train")
    model = embd_model.model
    model.save_pretrained("models/finetune-mrpc/final")
    model.push_to_hub("miniLM-mrpc-finetune")
    wandb.finish()

"""
Pre-training 

Evaluation Results: 0.386307988532814 | p_value: 1.6860126991028433e-62

Correlation factor is between -1, 1 where -1.0 is perfectly negative relationship and 1.0 is perfectly positive relationship
Correlation factor 0.3 < x < 0.5 implies weak - bad signaling

p value: probability that the correlation happened by random chance
p < 0.05: Correlation is statistically significant (not random)
p > 0.05: Correlation could be by chance/random
"""