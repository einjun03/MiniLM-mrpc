from data_processing import DatasetParser, augment_data
from datasets import load_dataset, Dataset, DatasetDict

def create_train_rows(dataset):
    """
    formatting train dataset (augmented) using group relations
    """ 
    #id to group mappings (np array), group: conflict group mappings, id to conflict group mappings, pure neg pairs set 
    parser = DatasetParser(dataset, is_train=True)
    parser.parse_dataset()
    df = augment_data(parser)
    return df
    

def create_retrieval_rows(dataset):
    """
    formatting val/test data using group relations 
    sample/ (all other groups + all negative pairs not in any group)
    corpus: negative samples + "relevant_docs" (items belonging to the same group) samples
    """
    parser = DatasetParser(dataset, is_train=False)
    parser.parse_dataset()

    # 1. Build the structures
    eval_queries = {}
    eval_corpus = {str(i): sent for i, sent in enumerate(parser.id_to_sentence)}
    eval_relevant_docs = {}

    for gid, members in parser.group_to_ids.items():
        members_list = list(members)
        for query_sid in members_list:
            qid = f"q_{query_sid}"
            eval_queries[qid] = parser.get_sentence(query_sid)
            
            # All OTHER members of the group are relevant matches
            other_members = [str(m) for m in members_list if m != query_sid]
            if other_members:
                eval_relevant_docs[qid] = other_members

    # 2. Calculate and Print Stats
    num_queries = len(eval_relevant_docs)
    corpus_size = len(eval_corpus)
    total_relevant_pairs = sum(len(docs) for docs in eval_relevant_docs.values())
    avg_relevant = total_relevant_pairs / num_queries if num_queries > 0 else 0

    print(f"--- Evaluator Statistics ---")
    print(f"Corpus Size (Haystack): {corpus_size}")
    print(f"Number of Queries:      {num_queries}")
    print(f"Avg Relevant Docs/Query: {avg_relevant:.2f}")
    print(f"---------------------------")

    # Convert dicts to lists of dicts for HF compatibility
    queries_ds = Dataset.from_list([{"id": k, "text": v} for k, v in eval_queries.items()])
    corpus_ds = Dataset.from_list([{"id": k, "text": v} for k, v in eval_corpus.items()])
    # Map sets to lists for JSON serialization
    qrels_ds = Dataset.from_list([{"id": k, "matches": list(v)} for k, v in eval_relevant_docs.items()])

    ir_dataset_dict = DatasetDict({
        "queries": queries_ds,
        "corpus": corpus_ds,
        "relevant_docs": qrels_ds
    })

    return ir_dataset_dict

if __name__ == '__main__':
    #train_dt = load_dataset("nyu-mll/glue", "mrpc", split="train")
    #augmented_train = create_train_rows(train_dt)

    val_dt = load_dataset("nyu-mll/glue", "mrpc", split="validation")
    reformatted_val = create_retrieval_rows(val_dt)

    test_dt = load_dataset("nyu-mll/glue", "mrpc", split="test")
    reformatted_test = create_retrieval_rows(test_dt)

    # 1. Training
    #augmented_train.push_to_hub("ejun26/mrpc-transitive-suite", config_name="training", split="train")

    # 2. Split retrieval into three distinct configurations
    for split_type in ["val", "test"]:
        data = reformatted_val if split_type == "val" else reformatted_test
        
        # Push Queries
        data["queries"].push_to_hub("ejun26/mrpc-transitive-suite", config_name="queries", split=split_type)
        
        # Push Corpus
        data["corpus"].push_to_hub("ejun26/mrpc-transitive-suite", config_name="corpus", split=split_type)
        
        # Push Relevant Docs (Qrels)
        data["relevant_docs"].push_to_hub("ejun26/mrpc-transitive-suite", config_name="qrels", split=split_type)