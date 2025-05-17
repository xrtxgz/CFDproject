from typing import Tuple, List, Set, Dict
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# Internal Prefix Tree used for fast minimality checking
class PrefixTreeNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class PrefixTree:
    def __init__(self):
        self.root = PrefixTreeNode()

    def insert(self, itemset: Tuple[str]):
        node = self.root
        for item in sorted(itemset):
            if item not in node.children:
                node.children[item] = PrefixTreeNode()
            node = node.children[item]
        node.is_end = True

    def delete(self, itemset: Tuple[str]) -> bool:
        path = []
        node = self.root
        for item in sorted(itemset):
            if item not in node.children:
                return False
            path.append((node, item))
            node = node.children[item]
        node.is_end = False
        for parent, label in reversed(path):
            child = parent.children[label]
            if not child.children and not child.is_end:
                del parent.children[label]
            else:
                break
        return True

    def has_subset(self, itemset: Tuple[str]) -> bool:
        def dfs(node, idx):
            if node.is_end:
                return True
            if idx >= len(itemset):
                return False
            for j in range(idx, len(itemset)):
                item = itemset[j]
                if item in node.children:
                    if dfs(node.children[item], j + 1):
                        return True
            return False

        sorted_itemset = tuple(sorted(itemset))
        return dfs(self.root, 0)


# RuleRepository now uses PrefixTree internally
class RuleRepository:
    def __init__(self):
        self.trees: Dict[str, PrefixTree] = defaultdict(PrefixTree)
        self.rules: Dict[str, List[Tuple[Tuple[str, ...], float]]] = defaultdict(list)

    def add_rule(self, lhs: Tuple[str, ...], rhs: str, conf: float):
        sorted_lhs = tuple(sorted(lhs))
        self.trees[rhs].insert(sorted_lhs)
        self.rules[rhs].append((lhs, conf))

    def has_subset(self, lhs: Tuple[str, ...], rhs: str) -> bool:
        return self.trees[rhs].has_subset(lhs)

    def has_superset(self, lhs: Tuple[str, ...], rhs: str) -> bool:
        for e_lhs, _ in self.rules[rhs]:
            if set(lhs).issubset(e_lhs):
                return True
        return False

    def delete(self, lhs: Tuple[str, ...], rhs: str) -> bool:
        """从规则集合与 prefix tree 中移除该条规则"""
        sorted_lhs = tuple(sorted(lhs))
        if rhs not in self.rules:
            return False
        self.rules[rhs] = [rule for rule in self.rules[rhs] if rule[0] != lhs]
        return self.trees[rhs].delete(sorted_lhs)

    def visualize_rhs_tree(self, rhs: str):
        """Visualize the LHS prefix tree structure under a certain RHS (with attribute names and termination tags), and return the matplotlib Figure"""
        if rhs not in self.trees:
            print(f"No rules for RHS: {rhs}")
            return None

        G = nx.DiGraph()
        node_id = [0]
        node_labels = {}

        def add_nodes(node, parent_id=None, incoming_label="ROOT"):
            current_id = node_id[0]
            G.add_node(current_id)
            if parent_id is not None:
                G.add_edge(parent_id, current_id)
            node_labels[current_id] = f"{incoming_label} {'(*)' if node.is_end else ''}"

            for label, child in node.children.items():
                node_id[0] += 1
                add_nodes(child, current_id, label)

        root = self.trees[rhs].root
        add_nodes(root)

        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=264)
        nx.draw(G, pos, ax=ax, with_labels=False, node_size=600, arrows=True)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)
        ax.set_title(f"Prefix Tree for RHS: {rhs}")
        ax.axis('off')

        return fig


class TrashRepository:
    def __init__(self):
        self.rules: Set[Tuple[Tuple[str, ...], str]] = set()

    def add_rejected(self, lhs: Tuple[str, ...], rhs: str):
        self.rules.add((lhs, rhs))

    def is_rejected(self, lhs: Tuple[str, ...], rhs: str) -> bool:
        return (lhs, rhs) in self.rules
