from data_processing import RetrievalParser
from datasets import Dataset, DatasetDict, load_dataset


def create_deduplicated_retrieval_dataset(dataset):
    parser = RetrievalParser(dataset)
    parser.build_original_mapping()
    # BFS grouping and paradox purging
    group_to_ids, _, _, _ = parser.parse_dataset()
    
    sent_to_group = {sid: gid for gid, members in group_to_ids.items() for sid in members}

    eval_queries = []
    eval_relevant_docs = []
    seen_relationships = set() # To track (min_id, max_id) pairs
    
    # Identify unique sentences for the corpus
    eval_corpus = [{"id": str(i), "text": sent} for i, sent in enumerate(parser.id_to_sentence)]

    for q_sid, pos_partners in parser.original_pos_pairs.items():
        # Filter partners to ensure we only create one direction per pair
        valid_partners = []
        for p_sid in pos_partners:
            relationship = tuple(sorted((q_sid, p_sid)))
            if relationship not in seen_relationships:
                valid_partners.append(p_sid)
                seen_relationships.add(relationship)

        if not valid_partners:
            continue

        qid = f"q_{q_sid}"
        relevant_ids = [str(p) for p in valid_partners]
        
        # Exclusion List: Transitive relatives (same group but not direct partners)
        q_group = sent_to_group.get(q_sid, -1)
        exclude_ids = set()
        if q_group != -1:
            exclude_ids = {
                str(m) for m in group_to_ids[q_group] 
                if str(m) not in relevant_ids and m != q_sid
            }

        eval_queries.append({
            "id": qid, 
            "text": parser.get_sentence(q_sid), 
            "exclude_from_corpus": list(exclude_ids)
        })
        eval_relevant_docs.append({"id": qid, "matches": relevant_ids})

    return DatasetDict({
        "queries": Dataset.from_list(eval_queries),
        "corpus": Dataset.from_list(eval_corpus),
        "relevant_docs": Dataset.from_list(eval_relevant_docs)
    })

def run_pipeline(repo_name="ejun26/mrpc-clean-retrieval-v2"):
    splits = ["validation", "test"]
    configs = ["queries", "corpus", "qrels"]
    
    for split in splits:
        print(f"📦 Processing split: {split}")
        raw_dt = load_dataset("nyu-mll/glue", "mrpc", split=split)
        processed_dict = create_deduplicated_retrieval_dataset(raw_dt)
        
        # Mapping our internal keys to the Hub's naming convention
        processed_dict["queries"].push_to_hub(repo_name, config_name="queries", split=split)
        processed_dict["corpus"].push_to_hub(repo_name, config_name="corpus", split=split)
        processed_dict["relevant_docs"].push_to_hub(repo_name, config_name="qrels", split=split)

    print(f"✨ Pipeline complete! View at: https://huggingface.co/datasets/{repo_name}")

if __name__ == "__main__":
    run_pipeline()