from data_processing import RetrievalParser
from datasets import Dataset, DatasetDict, load_dataset

def create_clean_retrieval_dataset(dataset):
    # 1. Parse the groups but don't augment
    parser = RetrievalParser(dataset)
    parser.build_original_mapping()
    # Runs BFS and handles poisoned group purging
    group_to_ids, _, _, _ = parser.parse_dataset()
    
    # Create a mapping of sentence_id -> group_id for fast exclusion
    sent_to_group = {}
    for gid, members in group_to_ids.items():
        for sid in members:
            sent_to_group[sid] = gid

    eval_queries = []
    eval_relevant_docs = []
    
    # 2. Build Queries and Relevant Docs based ONLY on original pairs
    # Each query will have its own unique corpus (exclusion list)
    for q_sid, pos_partners in parser.original_pos_pairs.items():
        qid = f"q_{q_sid}"
        query_text = parser.get_sentence(q_sid)
        
        # Ground Truth: Only the original partners
        relevant_ids = [str(p) for p in pos_partners]
        
        # Exclusion List: Everyone in the same group (the 'Transitive' relatives)
        q_group = sent_to_group.get(q_sid, -1)
        exclude_ids = set()
        if q_group != -1:
            exclude_ids = {str(m) for m in group_to_ids[q_group] if str(m) not in relevant_ids and m != q_sid}

        eval_queries.append({"id": qid, "text": query_text, "exclude_from_corpus": list(exclude_ids)})
        eval_relevant_docs.append({"id": qid, "matches": relevant_ids})

    # 3. Global Corpus (All sentences)
    eval_corpus = [{"id": str(i), "text": sent} for i, sent in enumerate(parser.id_to_sentence)]

    return DatasetDict({
        "queries": Dataset.from_list(eval_queries),
        "corpus": Dataset.from_list(eval_corpus),
        "relevant_docs": Dataset.from_list(eval_relevant_docs)
    })

def main():
    repo = "nyu-mll/glue"
    config = "mrpc"
    
    # Process Val and Test
    results = {}
    for split in ["validation", "test"]:
        print(f"\n--- Creating Clean Retrieval Set for {split} ---")
        dt = load_dataset(repo, config, split=split)
        results[split] = create_clean_retrieval_dataset(dt)

    # Standardize for Upload
    # Since each split has different exclusion needs, we store them as:
    # config: queries | split: val, test
    # config: corpus | split: val, test
    # config: qrels | split: val, test
    
    for split in ["validation", "test"]:
        results[split]["queries"].push_to_hub("ejun26/mrpc-clean-retrieval", config_name="queries", split=split)
        results[split]["corpus"].push_to_hub("ejun26/mrpc-clean-retrieval", config_name="corpus", split=split)
        results[split]["relevant_docs"].push_to_hub("ejun26/mrpc-clean-retrieval", config_name="qrels", split=split)

if __name__ == "__main__":
    main()