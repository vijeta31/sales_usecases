# OPTIMIZED SOLUTION with better time complexity:
class OptimizedNameMatcher:
    def __init__(self, global_name_dict=None):
        self.global_name_dict = global_name_dict or {}
        self.next_id = max(self.global_name_dict.values()) + 1 if self.global_name_dict else 1
        
        # Build inverted index for O(1) lookups
        self.word_to_id = {}
        self.prefix_to_id = {}
        self._build_indexes()
    
    def _build_indexes(self):
        """Build lookup indexes - O(N * L) preprocessing"""
        for name, id_val in self.global_name_dict.items():
            name_lower = name.lower()
            
            # Index by words
            for word in name_lower.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = []
                self.word_to_id[word].append(id_val)
            
            # Index by prefixes (for substring matching)
            for i in range(len(name_lower)):
                prefix = name_lower[:i+1]
                if prefix not in self.prefix_to_id:
                    self.prefix_to_id[prefix] = []
                self.prefix_to_id[prefix].append(id_val)
    
    def find_matching_id_optimized(self, new_name):
        """Find matching ID using indexes - O(L) average case"""
        new_name_lower = new_name.lower()
        
        # Try word-based matching first (fastest)
        for word in new_name_lower.split():
            if word in self.word_to_id:
                return self.word_to_id[word][0]
        
        # Fallback to prefix matching for substring cases
        for i in range(len(new_name_lower)):
            prefix = new_name_lower[:i+1]
            if prefix in self.prefix_to_id:
                return self.prefix_to_id[prefix][0]
        
        return None
    
    def process_new_names(self, new_names):
        """Process names with optimized lookup - O(M * L) average case"""
        results = {}
        
        for name in new_names:
            matching_id = self.find_matching_id_optimized(name)
            
            if matching_id:
                results[name] = matching_id
            else:
                results[name] = self.next_id
                self.next_id += 1
        
        # Update global dictionary and rebuild indexes
        self.global_name_dict.update(results)
        self._build_indexes()  # Rebuild for new entries
        return results

# TIME COMPLEXITY COMPARISON:
"""
Original Solution: O(M * N * L)
Optimized Solution: O(M * L) average case, O(M * N * L) worst case

Space Complexity: O(N * L) for indexes

The optimization works best when:
- Names have common words/prefixes
- Global dictionary is large (N >> M)
- Most lookups find matches quickly via indexes
"""
