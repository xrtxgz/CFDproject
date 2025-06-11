import pandas as pd
import numpy as np
import time
import tracemalloc
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.preprocessing import KBinsDiscretizer
from rule_repository import RuleRepository, TrashRepository
from pattern_trie import PatternTrie
from confidence import ConfidenceCalculator

def with_time_and_memory_tracking(func):
    """
    Decorator: Measures function running time + peak memory usage (accurate and not affected by garbage collection)
    """
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.time()

        result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        log = {
            "time_sec": round(elapsed, 3),
            "memory_peak_mb": round(peak / 1024 / 1024, 2)
        }

        return result, log
    return wrapper


class CFDDiscovererWithFD:
    def __init__(
        self,
        df: pd.DataFrame,
        maxsize: int = 10,
        minconf: float = 0.95,
        rhs_index: Optional[int] = None,
        n_bins: Optional[int] = None,
        conf_method="overall"
    ):
        self.original_df = df.copy()
        self.maxsize = maxsize
        self.minconf = minconf
        self.rhs_index = rhs_index
        self.n_bins = n_bins
        self.processed_df = self._preprocess_dataframe()
        self.db = self.processed_df.to_dict(orient='records')
        self.attributes = list(self.processed_df.columns)
        self.fd_candidates = []
        self.minimal_fds = []
        self.lhs_partitions: Dict[Tuple[str, ...], Dict[Tuple[str, ...], List[int]]] = {}
        self.repo = RuleRepository()
        self.conf_method = conf_method
        self.conf_calc = ConfidenceCalculator(self.db)
        
    def _preprocess_dataframe(self) -> pd.DataFrame:
        df = self.original_df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if self.n_bins is not None and numeric_cols:
            binner = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
            df[numeric_cols] = binner.fit_transform(df[numeric_cols])
            df[numeric_cols] = df[numeric_cols].astype(int).astype(str)
        else:
            df[numeric_cols] = df[numeric_cols].astype(str)
        df[categorical_cols] = df[categorical_cols].astype(str)
        return df

    def _normalize_support(self, min_support):
        if min_support < 1:
            return int(np.ceil(min_support * len(self.processed_df)))
        return int(min_support)

    def build_lhs_partitions(self):
        """Construct the equivalence class partitioning of all possible LHS: lhs -> {key_tuple: [row_indices]}"""
        self.lhs_partitions = {}
        for size in range(1, self.maxsize + 1):
            for lhs in combinations(self.attributes, size):
                partition = defaultdict(list)
                for idx, row in enumerate(self.db):
                    key = tuple(row[attr] for attr in lhs)
                    partition[key].append(idx)
                self.lhs_partitions[lhs] = partition
                
    def compute_fd_confidence(self, lhs: Tuple[str], rhs: str) -> float:
        lhs_partition = self.lhs_partitions.get(lhs)
        if lhs_partition is None:
            return 0.0
        return self.conf_calc.compute(lhs_partition, lhs, rhs, method=self.conf_method)

    def discover_fds(self) -> List[Tuple[Tuple[str], str, float]]:
        self.build_lhs_partitions()
        result = []
        rhs_attributes = [self.attributes[self.rhs_index]] if self.rhs_index is not None else self.attributes
        for size in range(1, self.maxsize + 1):
            for lhs in combinations(self.attributes, size):
                for rhs in rhs_attributes:
                    if rhs in lhs:
                        continue
                    conf = self.compute_fd_confidence(lhs, rhs)
                    if conf >= self.minconf:
                        result.append((lhs, rhs, conf))
        self.fd_candidates = result
        return result

    def visualize_fd_candidates(self, rhs: Optional[str] = None):
        """
        Construct the FD rule prefix tree and visualize the specified RHS.
        Automatically save the constructed rule tree as self.repo.
        """
        if not self.fd_candidates:
            self.discover_fds()

        self.repo = RuleRepository()
        for lhs, r, conf in self.fd_candidates:
            self.repo.add_rule(lhs, r, conf)

        if rhs:
            return self.repo.visualize_rhs_tree(rhs)
        else:
            figs = []
            for r in self.repo.rules:
                figs.append(self.repo.visualize_rhs_tree(r))
            return figs

    def visualize_minimal_fd_candidates(self, rhs: Optional[str] = None):
        """
        Build the prefix tree of minimal FD.
        """
        if not self.minimal_fds:
            self.discover_minimal_fds()

        self.repo = RuleRepository()
        for lhs, r, conf in self.minimal_fds:
            self.repo.add_rule(lhs, r, conf)

        if rhs:
            return self.repo.visualize_rhs_tree(rhs)

    def discover_minimal_fds(self, direct: bool = False) -> List[Tuple[Tuple[str], str, float]]:
        """
        Return the minimized FD. Pruning is performed by default using the existing candidate FDS.
        When setting direct=True, directly enumerate and prune simultaneously without first generating all FDS.
        """
        self.minimal_fds = []
        self.repo = RuleRepository()
        trash = TrashRepository()
    
        if direct:
            self.build_lhs_partitions()
            rhs_attributes = [self.attributes[self.rhs_index]] if self.rhs_index is not None else self.attributes
    
            for size in range(1, self.maxsize + 1):
                for lhs in combinations(self.attributes, size):
                    for rhs in rhs_attributes:
                        if rhs in lhs:
                            continue
                        sorted_lhs = tuple(sorted(lhs))
                        if trash.is_rejected(sorted_lhs, rhs):
                            continue
                        if self.repo.has_subset(sorted_lhs, rhs):
                            trash.add_rejected(sorted_lhs, rhs)
                            continue
                        conf = self.compute_fd_confidence(lhs, rhs)
                        if conf >= self.minconf:
                            self.repo.add_rule(sorted_lhs, rhs, conf)
                            self.minimal_fds.append((lhs, rhs, conf))
    
        else:
            if not self.fd_candidates:
                self.discover_fds()
            for lhs, rhs, conf in self.fd_candidates:
                sorted_lhs = tuple(sorted(lhs))
                if trash.is_rejected(sorted_lhs, rhs):
                    continue
                if self.repo.has_subset(sorted_lhs, rhs):
                    trash.add_rejected(sorted_lhs, rhs)
                    continue
                self.repo.add_rule(sorted_lhs, rhs, conf)
                self.minimal_fds.append((lhs, rhs, conf))
    
        return self.minimal_fds

    def discover_cfds(self, min_support: int = 5, rhs_index: Optional[int] = None) -> List[Tuple]:
        min_support = self._normalize_support(min_support)
        if not self.minimal_fds:
            self.discover_minimal_fds(direct=True)
    
        if rhs_index is not None:
            rhs_name = self.attributes[rhs_index]
            filtered_minfds = [fd for fd in self.minimal_fds if fd[1] == rhs_name]
        else:
            filtered_minfds = self.minimal_fds
    
        cfd_rules = []
        for lhs, rhs, _ in filtered_minfds:
            grouped = self.processed_df.groupby(list(lhs), observed=True)
            for pattern_values, group in grouped:
                if isinstance(pattern_values, str):
                    pattern_values = (pattern_values,)
                pattern_support = len(group)
                if pattern_support < min_support:
                    continue
                rhs_counts = Counter(group[rhs])
                most_common_value, count = rhs_counts.most_common(1)[0]
                confidence = count / pattern_support
    
                if confidence >= self.minconf:
                    lhs_pattern = tuple((attr, val) for attr, val in zip(lhs, pattern_values))
                    cfd_rules.append(((lhs_pattern, rhs), confidence, pattern_support))
    
        self.cfd_rules = sorted(cfd_rules, key=lambda x: (-x[2], -x[1]))
        return self.cfd_rules

    def discover_variable_cfds(self, min_support: int = 5, allow_overlap: bool = False, rhs_index: Optional[int] = None) -> List[Tuple]:
        min_support = self._normalize_support(min_support)
        if not self.minimal_fds:
            self.discover_minimal_fds(direct=True)
    
        if rhs_index is not None:
            rhs_name = self.attributes[rhs_index]
            filtered_minfds = [fd for fd in self.minimal_fds if fd[1] == rhs_name]
        else:
            filtered_minfds = self.minimal_fds
    
        variable_cfds = []
        for lhs, rhs, _ in filtered_minfds:
            grouped = self.processed_df.groupby(list(lhs), observed=True)
            for pattern_values, group in grouped:
                if isinstance(pattern_values, str):
                    pattern_values = (pattern_values,)
                support = len(group)
                if support < min_support:
                    continue
                rhs_counts = Counter(group[rhs])
                most_common_value, count = rhs_counts.most_common(1)[0]
                confidence = count / support
    
                if confidence >= self.minconf:
                    for mask in range(1, 2 ** len(lhs)):
                        pattern = []
                        for i, attr in enumerate(lhs):
                            if (mask >> i) & 1:
                                pattern.append((attr, pattern_values[i]))
                            else:
                                pattern.append((attr, '_'))
                        variable_cfds.append(((tuple(pattern), rhs), confidence, support))
    
        final_result = []
        if not allow_overlap:
            pattern_trie = PatternTrie()
    
        for (lhs_pattern, rhs), conf, supp in sorted(variable_cfds, key=lambda x: (-x[2], -x[1])):
            if allow_overlap or not pattern_trie.has_more_general_pattern(lhs_pattern):
                if not allow_overlap:
                    pattern_trie.insert(lhs_pattern)
                final_result.append(((lhs_pattern, rhs), conf, supp))
    
        self.variable_cfds = final_result
        return self.variable_cfds

    @with_time_and_memory_tracking
    def discover_cfds_tracked(self, min_support: int = 5, rhs_index: Optional[int] = None):
        min_support = self._normalize_support(min_support)
        rules = self.discover_cfds(min_support=min_support, rhs_index=rhs_index)  # 获取规则列表
        return {
            "type": "CFD",
            "rule_count": len(rules),
            "min_support": min_support,
            "min_confidence": self.minconf
        }

    @with_time_and_memory_tracking
    def discover_variable_cfds_tracked(self, min_support: int = 5, allow_overlap: bool = False, rhs_index: Optional[int] = None):
        min_support = self._normalize_support(min_support)
        rules = self.discover_variable_cfds(min_support=min_support, allow_overlap=allow_overlap, rhs_index=rhs_index)
        return {
            "type": "vCFD",
            "rule_count": len(rules),
            "min_support": min_support,
            "min_confidence": self.minconf,
            "allow_overlap": allow_overlap
        }

    def detect_cfd_violations(self, topk: int = 30, min_support: int = 5, allow_overlap: bool = False,
                              rhs_index: Optional[int] = None) -> List[int]:
        """
        Detect the index of the sample rows violated by the topk CFDS before (only detect variable CFDS)
        """
        if not hasattr(self, "variable_cfds") or not self.variable_cfds:
            min_support = self._normalize_support(min_support)
            self.discover_variable_cfds(min_support=min_support, allow_overlap=allow_overlap, rhs_index=rhs_index)

        violations = set()

        for (lhs_pattern, rhs_attr), conf, supp in self.variable_cfds[:topk]:
            for idx, row in enumerate(self.db):
                match = True
                for attr, val in lhs_pattern:
                    if val == "_":
                        continue
                    if row[attr] != val:
                        match = False
                        break
                if match:
                    if row[rhs_attr] != self._get_expected_rhs_value(lhs_pattern, rhs_attr):
                        violations.add(idx)

        return sorted(violations)

    def _get_expected_rhs_value(self, lhs_pattern: Tuple[Tuple[str, str]], rhs_attr: str) -> str:
        """
        Look for the most common RHS values in the samples that match the lhs_pattern in processed_df
        """
        cond = np.ones(len(self.processed_df), dtype=bool)
        for attr, val in lhs_pattern:
            if val == "_":
                continue
            cond &= (self.processed_df[attr] == val)
        subset = self.processed_df[cond]
        if subset.empty:
            return None
        return subset[rhs_attr].mode().iloc[0]

    def repair_errors(self, topk: int = 30, min_support: int = 5, allow_overlap: bool = False,
                      rhs_index: Optional[int] = None) -> pd.DataFrame:
        """
        Based on Top-K variable CFD, the RHS values that violate the rules in the data are repaired.
        Return the repaired DataFrame.
        """
        if not hasattr(self, "variable_cfds") or not self.variable_cfds:
            min_support = self._normalize_support(min_support)
            self.discover_variable_cfds(min_support=min_support, allow_overlap=allow_overlap, rhs_index=rhs_index)

        repaired_df = self.processed_df.copy()
        for (lhs_pattern, rhs_attr), conf, supp in self.variable_cfds[:topk]:
            if conf < self.minconf:
                continue
            expected_rhs_val = self._get_expected_rhs_value(lhs_pattern, rhs_attr)
            if expected_rhs_val is None:
                continue

            for idx, row in self.processed_df.iterrows():
                match = True
                for attr, val in lhs_pattern:
                    if val == "_":
                        continue
                    if row[attr] != val:
                        match = False
                        break
                if match and row[rhs_attr] != expected_rhs_val:
                    repaired_df.at[idx, rhs_attr] = expected_rhs_val  # 进行修复
        return repaired_df

    def get_top_fds(self, topk: int = 30):
        """
        Return the topk FD candidate rules (non-minimum), including the confidence level
        The first line indicates the total number, and the rest are rule texts
        """
        if not self.fd_candidates:
            self.discover_fds()

        header = f"FDs - Total: {len(self.fd_candidates)}"
        rules = [f"IF {' AND '.join(lhs)} THEN {rhs} (conf = {conf:.4f})"
                 for lhs, rhs, conf in self.fd_candidates[:topk]]
        return [header] + rules

    def get_top_minimal_fds(self, topk: int = 30, direct: bool = False):
        """
        Return the list of the topk Minimal FDs strings.
        Parameters:
            - topk: Number of top items to return
            - direct: If True, force discovery of new minimal FDs (overwrites current ones)
        Returns:
            - list[str]
        """
        if direct:
            self.discover_minimal_fds(direct=True)
        else:
            if not self.fd_candidates:
                self.discover_fds()
            self.discover_minimal_fds(direct=False)

        header = f"Minimal Functional Dependencies - Total: {len(self.minimal_fds)}"
        rules = [
            f"IF {' AND '.join(lhs)} THEN {rhs} (conf = {conf:.4f})"
            for lhs, rhs, conf in self.minimal_fds[:topk]
        ]
        return [header] + rules

    def get_top_cfds(self, topk: int = 30, min_support: int = 5, rhs_index: Optional[int] = None, force: bool = False):
        if force or not hasattr(self, "cfd_rules") or not self.cfd_rules:
            self.cfd_rules = []
            min_support = self._normalize_support(min_support)
            self.discover_cfds(min_support=min_support, rhs_index=rhs_index)

        header = f"Constant CFDs - Total: {len(self.cfd_rules)}"
        rules = [
            f"IF {' AND '.join(f'{a}={v}' for a, v in lhs)} THEN {rhs} (conf = {conf:.4f}, supp = {supp})"
            for (lhs, rhs), conf, supp in self.cfd_rules[:topk]
        ]
        return [header] + rules

    def get_top_variable_cfds(self, topk: int = 30, min_support: int = 5, allow_overlap: bool = False,
                              rhs_index: Optional[int] = None, force: bool = False):
        if force or not hasattr(self, "variable_cfds") or not self.variable_cfds:
            self.variable_cfds = []
            min_support = self._normalize_support(min_support)
            self.discover_variable_cfds(min_support=min_support, allow_overlap=allow_overlap, rhs_index=rhs_index)

        header = f"Variable CFDs - Total: {len(self.variable_cfds)}"
        rules = []
        for (lhs_pattern, rhs), conf, supp in self.variable_cfds[:topk]:
            lhs_str = " AND ".join(f"{a}={v}" for a, v in lhs_pattern)
            wildcard_count = sum(1 for _, v in lhs_pattern if v == "_")
            tag = "[general-retained]" if wildcard_count > 0 else "[specific-matched]"
            rules.append(f"IF {lhs_str} THEN {rhs} (conf = {conf:.4f}, supp = {supp}) {tag}")

        return [header] + rules


