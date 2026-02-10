from data_processing import DatasetParserFineGrained
from datasets import load_dataset, Dataset, DatasetDict

if __name__ == '__main__':
    # Define the splits we want to process
    splits = ["train", "validation", "test"]
    repo_id = "nyu-mll/glue"
    config_id = "mrpc"

    print(f"🚀 Starting clique extraction for {repo_id}/{config_id}...\n")

    for split_name in splits:
        print(f"--- Processing Split: {split_name.upper()} ---")
        
        # 1. Load the specific split
        dataset = load_dataset(repo_id, config_id, split=split_name)
        
        # 2. Initialize Parser (is_train=True to parse negative relations for analysis)
        parser = DatasetParserFineGrained(dataset, is_train=True)
        
        # 3. Parse the graph and identify cliques
        parser.parse_dataset()

