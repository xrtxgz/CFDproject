from typing import Tuple, List
import numpy as np
from collections import defaultdict, Counter

class ConfidenceCalculator:
    """
    A unified module for the three confidence calculation methods:
    overall: Global confidence level (current default)
    min: Minimum confidence level for each group
    avg: Average confidence level per group
    """
    def __init__(self, db: List[dict]):
        self.db = db

    def compute(self, lhs_partition, lhs: Tuple[str], rhs: str, method: str = "overall") -> float:
        lhs_rhs_partition = defaultdict(list)
        for lhs_key, tids in lhs_partition.items():
            for i in tids:
                rhs_val = self.db[i][rhs]
                lhs_rhs_partition[(lhs_key, rhs_val)].append(i)

        if method == "overall":
            total_support = 0
            total_error = 0
            for lhs_key, tids in lhs_partition.items():
                total_support += len(tids)
                subclass_sizes = [
                    len(lhs_rhs_partition[(lhs_key, rhs_val)])
                    for rhs_val in set(self.db[i][rhs] for i in tids)
                ]
                max_class_size = max(subclass_sizes) if subclass_sizes else 0
                total_error += len(tids) - max_class_size
            return 1 - total_error / total_support if total_support > 0 else 0.0

        elif method == "min":
            confidences = []
            for lhs_key, tids in lhs_partition.items():
                counts = Counter(self.db[i][rhs] for i in tids)
                conf = max(counts.values()) / len(tids)
                confidences.append(conf)
            return min(confidences) if confidences else 0.0

        elif method == "avg":
            confidences = []
            for lhs_key, tids in lhs_partition.items():
                counts = Counter(self.db[i][rhs] for i in tids)
                conf = max(counts.values()) / len(tids)
                confidences.append(conf)
            return float(np.mean(confidences)) if confidences else 0.0

        else:
            raise ValueError(f"Unknown confidence method: {method}")
