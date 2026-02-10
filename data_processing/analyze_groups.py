from parser import DatasetParser
from datasets import load_dataset

def export_large_groups(parser, output_file="large_groups.txt"):
    """
    Identifies groups with more than 2 sentences and writes them to a file.
    """
    # 1. Filter groups with size > 2
    large_groups = {
        gid: members for gid, members in parser.group_to_ids.items() 
        if len(members) > 2
    }
    
    # 2. Print count
    print(f"Number of groups with more than 2 items: {len(large_groups)}")
    
    # 3. Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        for gid, members in large_groups.items():
            f.write(f"--- Group ID: {gid} (Size: {len(members)}) ---\n")
            for sid in members:
                sentence = parser.get_sentence(sid)
                f.write(f"- {sentence}\n")
            f.write("\n")  # Add a spacer between groups

    print(f"Successfully exported large groups to {output_file}")

# --- Example Usage ---
# parser = DatasetParser(dataset)
# parser.parse_dataset()
# export_large_groups(parser)

def main():
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
        parser = DatasetParser(dataset, is_train=True)
        
        # 3. Parse the graph and identify cliques
        parser.parse_dataset()
        
        # 4. Export groups with more than 2 items to a text file
        output_filename = f"cliques_{split_name}.txt"
        
        # Filter and count large groups
        large_groups = {
            gid: members for gid, members in parser.group_to_ids.items() 
            if len(members) > 2
        }
        
        print(f"✅ Found {len(large_groups)} groups with size > 2.")
        
        # Write to split-specific file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"=== MRPC {split_name.upper()} CLIQUES (Size > 2) ===\n")
            f.write(f"Total Large Groups: {len(large_groups)}\n\n")
            
            for gid, members in large_groups.items():
                f.write(f"Group ID {gid} | Size: {len(members)}\n")
                for sid in members:
                    f.write(f"  - {parser.get_sentence(sid)}\n")
                f.write("\n")
        
        print(f"💾 Exported to: {output_filename}\n")

    print("✨ All splits processed successfully.")

if __name__ == "__main__":
    main()
