# MiniLM on MRPC, as retrieval

Fine-tuning `all-MiniLM-L6-v2` for sentence retrieval, using MRPC — and fixing
the labelling problem that makes MRPC misleading as a retrieval benchmark.

**Models** · [`ejun26/minilm-mrpc-clean-retrieval`](https://huggingface.co/ejun26/minilm-mrpc-clean-retrieval) · [`ejun26/miniLM-mrpc-finetune`](https://huggingface.co/ejun26/miniLM-mrpc-finetune)
**Datasets** · [`mrpc-clean-retrieval-v2`](https://huggingface.co/datasets/ejun26/mrpc-clean-retrieval-v2) · [`mrpc-clean-retrieval`](https://huggingface.co/datasets/ejun26/mrpc-clean-retrieval) · [`mrpc-transitive-suite`](https://huggingface.co/datasets/ejun26/mrpc-transitive-suite)

---

## The problem

MRPC is a paraphrase **classification** set: pairs of sentences labelled
equivalent or not. Reusing it for **retrieval** — encode a query, rank a corpus,
score MRR@10 — breaks on an assumption the original labels never made.

Paraphrase is transitive. If A ≡ B and B ≡ C, then A ≡ C. MRPC never labelled
the A–C pair, because it was annotating pairs rather than building equivalence
classes. So a retrieval evaluator sees the model rank C highly for query A,
finds no relevance label, and scores it as a **false positive** — punishing the
model for being right.

The better the embeddings get, the worse this gets.

## The fix

**1. Recover the equivalence classes.** Treat positive pairs as edges and take
the transitive closure, giving paraphrase cliques:

```
Train   7,052 unique sentences   2,338 cliques   1 inter-group conflict
Val       807 unique sentences     276 cliques   0 inter-group conflicts
```

The conflict count matters — it's the check that closure didn't merge two
groups that MRPC explicitly labelled *not* equivalent. One conflict across
2,338 train cliques means the transitive assumption essentially holds.

**2. Augment from the closure.** Add the pairs transitivity implies, and
propagate negatives across cliques:

```
New positive rows (transitive)    149
New negative rows (propagated)     72
Total augmented dataset size    3,889
```

**3. Exclude, don't count wrong.**
[`models/evaluator.py`](models/evaluator.py) subclasses
`InformationRetrievalEvaluator` so each query drops from its own haystack any
document that is semantically equivalent but unlabelled — plus the query itself,
which otherwise retrieves as its own perfect match:

```python
class CleanInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    def __init__(self, queries, corpus, relevant_docs, query_exclusions, **kwargs):
        ...
        # Ensure every query excludes itself from the haystack
```

The resulting evaluation set is small and honest:

```
Corpus (haystack)          807
Queries                    555
Avg relevant docs/query   1.02
```

## Training

```bash
./setup.sh
./run_train.sh --use_wandb --use_custom_evaluator --run_name clean-v2
```

`run_train.sh` handles W&B and HuggingFace login from `WANDB_API_KEY` and
`HF_TOKEN` (both optional — it runs without them) and passes everything else
through to `train.py`.

| Flag | Default | |
| --- | --- | --- |
| `--loss_fn` | `CoSENTLoss` | or `OnlineContrastiveLoss` |
| `--epochs` | 4 | |
| `--lr` | 2e-5 | |
| `--batch_size` | 32 | |
| `--margin` | 0.5 | OnlineContrastiveLoss only |
| `--eval_steps` | 50 | evaluate and checkpoint by step, not epoch |
| `--use_custom_evaluator` | off | the exclusion-aware evaluator above |
| `--push_to_hub` | off | with `--hub_model_name` |
| `--debug` | off | fast sanity run on tiny data |

Checkpointing selects on `cosine_mrr@10` with `load_best_model_at_end`, so the
saved model is the best retriever seen, not the last epoch.

## Layout

```
train.py                     training entry point
run_train.sh  setup.sh       login + launch
diagnose.py                  inspection helper
models/
  retrieval_model.py         model wrapper
  correlation_model.py
  evaluator.py               exclusion-aware IR evaluator
data_processing/
  parser.py  data_utils.py
  analyze_groups.py          clique construction, conflict detection
data_augmentation/
  create_datasets.py         builds the published HF datasets
  mrpc_retrieval_data.py
  mrpc_retrieval_dedup.py
  analyze_contradictions.py  finds label conflicts across cliques
  MRPC.ipynb
```

Used in production by
[prediction-market-inference](https://github.com/einjun03/prediction-market-inference),
where the embedding service matches equivalent prediction markets across Kalshi
and Polymarket — the same retrieval problem, with money attached.
