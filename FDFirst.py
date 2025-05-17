import pandas as pd
import numpy as np
import time
import psutil
import os
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
    装饰器：测量函数运行时间 + 内存使用峰值（准确、不受垃圾回收影响）
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
        if min_support < 1:  # 比如 0.05，表示百分比
            return int(np.ceil(min_support * len(self.processed_df)))
        return int(min_support)

    def build_lhs_partitions(self):
        """构造所有可能 LHS 的等价类划分：lhs -> {key_tuple: [row_indices]}"""
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
        可视化 discover_fds() 后得到的所有候选 FD（未最小化）结构树。
        默认显示所有 RHS；也可指定 rhs。
        """
        if not self.fd_candidates:
            print("⚠️ 请先调用 discover_fds() 生成 fd_candidates。")
            return
    
        repo = RuleRepository()
        for lhs, r, conf in self.fd_candidates:
            repo.add_rule(lhs, r, conf)
    
        if rhs:
            repo.visualize_rhs_tree(rhs)
        else:
            for r in repo.rules:
                print(f"Visualizing FD candidates for RHS: {r}")
                repo.visualize_rhs_tree(r)

        
    def discover_minimal_fds(self, direct: bool = False) -> List[Tuple[Tuple[str], str, float]]:
        """
        返回最小化的 FD。默认使用已有候选 FD 进行剪枝。
        设置 direct=True 时，直接边枚举边剪枝，无需先生成所有 FD。
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
    
    def detect_cfd_violations(self, topk: int = 30) -> List[int]:
        """
        检测前 topk 条 CFD 所违反的样本行索引（仅检测 variable CFD）
        """
        if not hasattr(self, "variable_cfds") or not self.variable_cfds:
            self.discover_variable_cfds()
    
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
        在 processed_df 中查找与 lhs_pattern 匹配样本中最常见的 RHS 值
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

    def repair_errors(self, topk: int = 30) -> pd.DataFrame:
        """
        基于 Top-K variable CFD，对数据中违反规则的 RHS 值进行修复。
        返回修复后的 DataFrame。
        """
        if not hasattr(self, "variable_cfds") or not self.variable_cfds:
            self.discover_variable_cfds()
    
        repaired_df = self.processed_df.copy()
        for (lhs_pattern, rhs_attr), conf, supp in self.variable_cfds[:topk]:
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
        打印前 topk 条 FD 候选规则（非最小），包含置信度
        """
        if not self.fd_candidates:
            self.discover_fds()
        print(f"Functional Dependencies (FDs) - Total: {len(self.fd_candidates)}")
        for i, (lhs, rhs, conf) in enumerate(self.fd_candidates[:topk], 1):
            lhs_str = " AND ".join(lhs)
            print(f"{i}. IF {lhs_str} THEN {rhs} (conf = {conf:.4f})")

    def get_top_minimal_fds(self, topk: int = 30, direct: bool = False):
        """
        打印前 topk 条 Minimal FDs。
        参数：
            - topk: int，显示条数
            - direct: bool，是否直接生成 minimal FDs（跳过 discover_fds）
        """
        if not self.minimal_fds:
            self.discover_minimal_fds(direct=direct)
    
        print(f"Minimal Functional Dependencies - Total: {len(self.minimal_fds)}")
        for i, (lhs, rhs, conf) in enumerate(self.minimal_fds[:topk], 1):
            lhs_str = " AND ".join(lhs)
            print(f"{i}. IF {lhs_str} THEN {rhs} (conf = {conf:.4f})")
    
    def get_top_cfds(self, topk: int = 30):
        """
        打印前 topk 条 constant CFDs：IF A=val AND B=val THEN C=val
        """
        if not hasattr(self, "cfd_rules") or not self.cfd_rules:
            self.discover_cfds()
        print(f"Constant CFDs - Total: {len(self.cfd_rules)}")
        for i, ((lhs_pattern, rhs), conf, supp) in enumerate(self.cfd_rules[:topk], 1):
            lhs_str = " AND ".join(f"{a}={v}" for a, v in lhs_pattern)
            print(f"{i}. IF {lhs_str} THEN {rhs} (conf = {conf:.4f}, supp = {supp})")

    
    def get_top_variable_cfds(self, topk: int = 30):
        """
        打印前 topk 条 variable CFDs：IF A=val OR A=_ THEN ...
        并添加标记：是否为 general retained 或 specific matched 规则
        """
        if not hasattr(self, "variable_cfds") or not self.variable_cfds:
            self.discover_variable_cfds()
        print(f"Variable CFDs - Total: {len(self.variable_cfds)}")
        for i, ((lhs_pattern, rhs), conf, supp) in enumerate(self.variable_cfds[:topk], 1):
            lhs_str = " AND ".join(f"{a}={v}" for a, v in lhs_pattern)
            wildcard_count = sum(1 for a, v in lhs_pattern if v == "_")
            if wildcard_count > 0:
                tag = "[general-retained]"
            else:
                tag = "[specific-matched]"
            print(f"{i}. IF {lhs_str} THEN {rhs} (conf = {conf:.4f}, supp = {supp}) {tag}")