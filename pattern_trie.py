from typing import Tuple, Dict, Optional

class PatternTrieNode:
    def __init__(self):
        self.children: Dict[str, 'PatternTrieNode'] = {}
        self.is_end: bool = False

class PatternTrie:
    def __init__(self):
        self.root = PatternTrieNode()

    def _key(self, attr: str, val: str) -> str:
        return f"{attr}={val}"

    def insert(self, pattern: Tuple[Tuple[str, str]]):
        node = self.root
        for attr, val in sorted(pattern):  # 保证顺序一致
            key = self._key(attr, val)
            if key not in node.children:
                node.children[key] = PatternTrieNode()
            node = node.children[key]
        node.is_end = True

    def has_more_general_pattern(self, pattern: Tuple[Tuple[str, str]]) -> bool:
        """
        Determine whether a rule that is more generalized (with more _) than the current pattern already exists
        For example, when there is already (A=_, B=1), do not insert (A=val, B=1).
        """
        def dfs(node, idx):
            if node.is_end:
                return True
            if idx >= len(pattern):
                return False
            attr, val = pattern[idx]
            candidates = []

            # General matching: The current path can be followed by _ or a specific value
            key_exact = self._key(attr, val)
            key_wild = self._key(attr, '_')
            if key_exact in node.children:
                candidates.append(node.children[key_exact])
            if key_wild in node.children:
                candidates.append(node.children[key_wild])

            for child in candidates:
                if dfs(child, idx + 1):
                    return True
            return False

        sorted_pattern = sorted(pattern)
        return dfs(self.root, 0)

    def debug_print(self):
        def dfs(node, path):
            if node.is_end:
                print("Rule:", " AND ".join(path))
            for key, child in node.children.items():
                dfs(child, path + [key])
        dfs(self.root, [])
