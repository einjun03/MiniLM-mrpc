import pandas as pd
from datasets import Dataset
from collections import defaultdict

def augment_data(parser, negative_limit=10):
    original_dataset = parser.dataset
    groups = parser.group_to_ids  # Now using the class attribute
    conflicts = parser.group_conflicts
    isolated_to_group = parser.isolated_neg_to_group

    new_positives = 0
    new_negatives = 0
    
    augmented_rows = []
    existing_pairs = set()

    # 1. Load original data
    for row in original_dataset:
        s1_text, s2_text = row['sentence1'], row['sentence2']
        augmented_rows.append({
            "sentence1": s1_text,
            "sentence2": s2_text,
            "label": row['label']
        })
        existing_pairs.add(tuple(sorted([s1_text, s2_text])))

    # 3. Augment Positives (Transitivity)
    for gid, members in groups.items():
        members_list = list(members) # Convert set to list for indexing
        n = len(members_list)
        if n < 2: continue
        
        for i in range(n):
            for j in range(i + 1, n):
                s1_text = parser.id_to_sentence[members_list[i]]
                s2_text = parser.id_to_sentence[members_list[j]]
                pair = tuple(sorted([s1_text, s2_text]))
                
                if pair not in existing_pairs:
                    augmented_rows.append({"sentence1": s1_text, "sentence2": s2_text, "label": 1})
                    existing_pairs.add(pair)
                    new_positives += 1

    # 4. Augment Negatives (Group-to-Group Propagation)
    for g1, conflicting_groups in conflicts.items():
        for g2 in conflicting_groups:
            if g1 > g2: continue 
            
            count_for_this_conflict = 0
            for s1_id in groups[g1]:
                for s2_id in groups[g2]:
                    if negative_limit and count_for_this_conflict >= negative_limit:
                        break
                    
                    s1_text = parser.id_to_sentence[s1_id]
                    s2_text = parser.id_to_sentence[s2_id]
                    pair = tuple(sorted([s1_text, s2_text]))
                    
                    if pair not in existing_pairs:
                        augmented_rows.append({"sentence1": s1_text, "sentence2": s2_text, "label": 0})
                        existing_pairs.add(pair)
                        new_negatives += 1
                        count_for_this_conflict += 1

    # 5. NEW: Augment Negatives (Isolated Sentence to Group)
    for sent_id, target_groups in isolated_to_group.items():
        s1_text = parser.id_to_sentence[sent_id]
        
        for gid in target_groups:
            count_for_this_isolated = 0
            for s2_id in groups[gid]:
                if negative_limit and count_for_this_isolated >= negative_limit:
                    break
                
                s2_text = parser.id_to_sentence[s2_id]
                pair = tuple(sorted([s1_text, s2_text]))
                
                if pair not in existing_pairs:
                    augmented_rows.append({"sentence1": s1_text, "sentence2": s2_text, "label": 0})
                    existing_pairs.add(pair)
                    new_negatives += 1
                    count_for_this_isolated += 1

    # Final Logging
    print(f"--- Augmentation Summary ---")
    print(f"New Positive Rows (Transitive): {new_positives}")
    print(f"New Negative Rows (Propagated): {new_negatives}")
    print(f"Total Augmented Dataset Size: {len(augmented_rows)}")
    print(f"----------------------------")

    return Dataset.from_pandas(pd.DataFrame(augmented_rows))