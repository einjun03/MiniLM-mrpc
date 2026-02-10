import numpy as np
from collections import defaultdict, deque

class DatasetParser:

    def __init__(self, dataset, is_train=True):
        # Global Symbol Table
        self.dataset = dataset
        self.sentence_to_id = {}
        self.id_to_sentence = []
        self.group_to_ids = defaultdict(set)

        # Group A vs Group B conflicts
        self.group_conflicts = defaultdict(set)
        # Sentence X vs Group A conflicts
        self.isolated_neg_to_group = defaultdict(set)
        # Sentences with NO positive pairs that are negative to each other
        self.pure_negative_pairs = set()

        self.parse_negative = is_train

    def get_id(self, sentence):
        """Maps sentence strings to unique integer IDs."""
        if sentence not in self.sentence_to_id:
            sid = len(self.id_to_sentence)
            self.sentence_to_id[sentence] = sid
            self.id_to_sentence.append(sentence)
            return sid
        return self.sentence_to_id[sentence]
    
    def get_sentence(self, id):
        return self.id_to_sentence[id]

    def process_dataset_to_ids(self):
        """
        Pre-processes the entire dataset into a list of (id1, id2, label).
        This avoids repeated dictionary lookups during graph traversal.
        """
        processed_data = []
        for row in self.dataset:
            u = self.get_id(row['sentence1'])
            v = self.get_id(row['sentence2'])
            processed_data.append((u, v, row['label']))
        return processed_data

    def create_graphs(self, processed_data):
        """Builds adjacency lists using integer IDs."""
        adj = defaultdict(set)
        neg_adj = defaultdict(set)
        all_positive_ids = set()
        
        for u, v, label in processed_data:
            if label == 1:
                adj[u].add(v)
                adj[v].add(u)
                all_positive_ids.add(u)
                all_positive_ids.add(v)
            elif self.parse_negative:
                neg_adj[u].add(v)
                neg_adj[v].add(u)
                
        return all_positive_ids, adj, neg_adj

    def create_groups(self, all_positive_ids, adj, total_sentences):
        """
        Uses BFS to find connected components.
        Returns a NumPy array where index is sentence_id and value is group_id.
        """
        # Initialize with -1 (meaning not in a positive group)
        group_map = np.full(total_sentences, -1, dtype=np.int32)
        visited = set()
        curr_group = 0

        for root_id in all_positive_ids:
            if root_id not in visited:
                queue = deque([root_id])
                visited.add(root_id)
                while queue:
                    u = queue.popleft()
                    group_map[u] = curr_group
                    self.group_to_ids[curr_group].add(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            queue.append(v)
                curr_group += 1
                
        return group_map, curr_group

    def analyze_negative_relations(self, group_map, neg_adj):
        """
        Maps how groups conflict with each other and identifies 
        isolated negative distractors.
        """

        for u, neighbors in neg_adj.items():
            u_group = group_map[u]
            
            for v in neighbors:
                v_group = group_map[v]
                
                if u_group != -1 and v_group != -1:
                    if u_group != v_group:
                        self.group_conflicts[u_group].add(v_group)
                elif u_group != -1 and v_group == -1:
                    self.isolated_neg_to_group[v].add(u_group)
                elif u_group == -1 and v_group == -1:
                    # Both are isolated, but we know they are not equivalent
                    pair = tuple(sorted((u, v)))
                    self.pure_negative_pairs.add(pair)
        
        return self.group_conflicts, self.isolated_neg_to_group, self.pure_negative_pairs

    def parse_dataset(self):
        # 1. Map to IDs
        processed_data = self.process_dataset_to_ids()
        num_sents = len(self.id_to_sentence)
        # 2. Build Adjacency
        pos_ids, pos_adj, neg_adj = self.create_graphs(processed_data)

        # 3. Create Group Map (The "Search Index")
        group_map, total_groups = self.create_groups(pos_ids, pos_adj, num_sents)

        # 4. Analyze Logic Gaps
        conflicts, isolated, pure_negs = self.analyze_negative_relations(group_map, neg_adj)

        print(f"Total Unique Sentences: {num_sents}")
        print(f"Total Positive Groups (Cliques): {total_groups}")
        print(f"Inter-Group Negative Conflicts: {sum(len(v) for v in conflicts.values()) // 2}")

        return self.group_to_ids, conflicts, isolated, pure_negs

class RetrievalParser(DatasetParser):
    """
    Subclass of your DatasetParser specifically for 
    creating clean IR Benchmarks.
    """
    def __init__(self, dataset):
        super().__init__(dataset, is_train=False)
        # We need a direct lookup of original positive pairs
        self.original_pos_pairs = defaultdict(set)

    def build_original_mapping(self):
        """Map every sentence to its direct Label 1 partners."""
        for row in self.dataset:
            if row['label'] == 1:
                u = self.get_id(row['sentence1'])
                v = self.get_id(row['sentence2'])
                self.original_pos_pairs[u].add(v)
                self.original_pos_pairs[v].add(u)

class DatasetParserFineGrained:
    def __init__(self, dataset, is_train=True):
        self.dataset = dataset
        self.sentence_to_id = {}
        self.id_to_sentence = []
        self.group_to_ids = defaultdict(set)
        self.group_conflicts = defaultdict(set)
        self.isolated_neg_to_group = defaultdict(set)
        self.pure_negative_pairs = set()
        self.parse_negative = is_train

    def get_id(self, sentence):
        if sentence not in self.sentence_to_id:
            sid = len(self.id_to_sentence)
            self.sentence_to_id[sentence] = sid
            self.id_to_sentence.append(sentence)
            return sid
        return self.sentence_to_id[sentence]
    
    def get_sentence(self, id):
        return self.id_to_sentence[id]

    def process_dataset_to_ids(self):
        processed_data = []
        for row in self.dataset:
            u = self.get_id(row['sentence1'])
            v = self.get_id(row['sentence2'])
            processed_data.append((u, v, row['label']))
        return processed_data

    def create_graphs(self, processed_data):
        adj = defaultdict(set)
        neg_adj = defaultdict(set)
        all_positive_ids = set()
        for u, v, label in processed_data:
            if label == 1:
                adj[u].add(v)
                adj[v].add(u)
                all_positive_ids.add(u)
                all_positive_ids.add(v)
            elif self.parse_negative:
                neg_adj[u].add(v)
                neg_adj[v].add(u)
        return all_positive_ids, adj, neg_adj

    def create_groups(self, all_positive_ids, adj, total_sentences):
        group_map = np.full(total_sentences, -1, dtype=np.int32)
        visited = set()
        curr_group = 0
        for root_id in all_positive_ids:
            if root_id not in visited:
                queue = deque([root_id])
                visited.add(root_id)
                while queue:
                    u = queue.popleft()
                    group_map[u] = curr_group
                    self.group_to_ids[curr_group].add(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            queue.append(v)
                curr_group += 1
        return group_map, curr_group

    def find_and_purge_contradictions(self, group_map, neg_adj):
        """
        Identifies groups where a 'Label 0' exists between members.
        Removes these groups so no transitive augmentation occurs.
        """
        poisoned_groups = set()
        for u, neighbors in neg_adj.items():
            u_group = group_map[u]
            if u_group == -1: continue
            
            for v in neighbors:
                v_group = group_map[v]
                # If a negative edge exists within the SAME positive group
                if u_group == v_group:
                    poisoned_groups.add(u_group)
        
        if poisoned_groups:
            print(f"🚩 Found {len(poisoned_groups)} poisoned groups with internal contradictions.")
            for gid in poisoned_groups:
                # Discarding the group prevents new (A,C) pairs if (A,B)=1 and (B,C)=1 but (A,C)=0
                del self.group_to_ids[gid]
        
        return poisoned_groups

    def analyze_negative_relations(self, group_map, neg_adj, poisoned_groups):
        """Updated to ignore poisoned groups during relation mapping."""
        for u, neighbors in neg_adj.items():
            u_group = group_map[u]
            if u_group in poisoned_groups: u_group = -1 # Treat as isolated
            
            for v in neighbors:
                v_group = group_map[v]
                if v_group in poisoned_groups: v_group = -1
                
                if u_group != -1 and v_group != -1:
                    if u_group != v_group:
                        self.group_conflicts[u_group].add(v_group)
                elif u_group != -1 and v_group == -1:
                    self.isolated_neg_to_group[v].add(u_group)
                elif u_group == -1 and v_group == -1:
                    pair = tuple(sorted((u, v)))
                    self.pure_negative_pairs.add(pair)
        
        return self.group_conflicts, self.isolated_neg_to_group, self.pure_negative_pairs

    def parse_dataset(self):
        processed_data = self.process_dataset_to_ids()
        num_sents = len(self.id_to_sentence)
        pos_ids, pos_adj, neg_adj = self.create_graphs(processed_data)
        group_map, total_groups = self.create_groups(pos_ids, pos_adj, num_sents)

        # NEW STEP: Remove groups that have internal 'Label 0' markers
        poisoned = self.find_and_purge_contradictions(group_map, neg_adj)

        conflicts, isolated, pure_negs = self.analyze_negative_relations(group_map, neg_adj, poisoned)

        print(f"Total Unique Sentences: {num_sents}")
        print(f"Clean Positive Groups: {len(self.group_to_ids)} (Discarded {len(poisoned)})")
        return self.group_to_ids, conflicts, isolated, pure_negs