"""
Isomorphic Neuron Matcher.
Solves the Linear Sum Assignment Problem (LSAP) to recover neuron permutations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def solve_bipartite_matching(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Finds the optimal permutation mapping target row indices to base column indices to minimize total cost.
    Uses SciPy's Jonker-Volgenant/Hungarian solver if available, with a fast NumPy fallback.
    Returns:
        perm: 1D array of length N where perm[k] is the matched index in base for neuron k in target.
    """
    N = cost_matrix.shape[0]
    if HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        perm = np.zeros(N, dtype=np.int64)
        perm[row_ind] = col_ind
        return perm

    # Fallback: Greedy Bipartite Matching with 2-opt Refinement
    perm = np.zeros(N, dtype=np.int64)
    assigned_cols = set()
    
    flat_indices = np.argsort(cost_matrix, axis=None)
    rows, cols = np.unravel_index(flat_indices, cost_matrix.shape)
    
    assigned_rows = set()
    for r, c in zip(rows, cols):
        if r not in assigned_rows and c not in assigned_cols:
            perm[r] = c
            assigned_rows.add(r)
            assigned_cols.add(c)
            if len(assigned_rows) == N:
                break

    unassigned_cols = list(set(range(N)) - assigned_cols)
    for r in range(N):
        if r not in assigned_rows:
            c = unassigned_cols.pop()
            perm[r] = c
            assigned_rows.add(r)

    for _ in range(3):
        improved = False
        for i in range(N):
            for j in range(i + 1, min(N, i + 64)):
                pi, pj = perm[i], perm[j]
                current_cost = cost_matrix[i, pi] + cost_matrix[j, pj]
                swapped_cost = cost_matrix[i, pj] + cost_matrix[j, pi]
                if swapped_cost < current_cost:
                    perm[i], perm[j] = pj, pi
                    improved = True
        if not improved:
            break

    return perm


class IsomorphicMatcher:
    """
    Computes optimal neuron permutation between base and target checkpoint layers.
    """

    def __init__(self, confidence_threshold: float = 0.05):
        self.confidence_threshold = confidence_threshold

    def compute_cost_matrix(
        self,
        base_out: np.ndarray,
        target_out: np.ndarray,
        base_in: Optional[np.ndarray] = None,
        target_in: Optional[np.ndarray] = None,
        base_vectors: Optional[List[np.ndarray]] = None,
        target_vectors: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs pairwise distance matrix M where M[k, j] is the distance between
        neuron k in TARGET model and neuron j in BASE model.
        Returns:
            total_cost_matrix: weighted sum of distance matrices
            primary_out_cost: pure primary output layer cosine distance (for confidence calculation)
        """
        N = target_out.shape[0]

        # Flatten spatial and input dimensions: shape [N, D_out]
        target_out_flat = target_out.reshape(N, -1).astype(np.float32)
        base_out_flat = base_out.reshape(N, -1).astype(np.float32)

        # Normalize features
        norm_target_out = np.linalg.norm(target_out_flat, axis=1, keepdims=True) + 1e-8
        norm_base_out = np.linalg.norm(base_out_flat, axis=1, keepdims=True) + 1e-8
        
        # Primary Cosine distance: M[k, j] = 1 - cosine_similarity(target[k], base[j])
        out_sim = (target_out_flat / norm_target_out) @ (base_out_flat / norm_base_out).T
        primary_out_cost = 1.0 - np.clip(out_sim, -1.0, 1.0)
        total_cost = primary_out_cost.copy()

        # Incorporate bias and norm vectors (e.g. running stats, scaling factors)
        if base_vectors and target_vectors:
            for bv, tv in zip(base_vectors, target_vectors):
                tv_flat = tv.reshape(N, 1).astype(np.float32)
                bv_flat = bv.reshape(1, N).astype(np.float32)
                diff = (tv_flat - bv_flat) ** 2
                max_diff = np.max(diff) + 1e-8
                total_cost += 0.5 * (diff / max_diff)

        # Incorporate subsequent layer input weights with small weight if provided
        if base_in is not None and target_in is not None:
            target_in_re = np.swapaxes(target_in, 0, 1).reshape(N, -1).astype(np.float32)
            base_in_re = np.swapaxes(base_in, 0, 1).reshape(N, -1).astype(np.float32)

            norm_target_in = np.linalg.norm(target_in_re, axis=1, keepdims=True) + 1e-8
            norm_base_in = np.linalg.norm(base_in_re, axis=1, keepdims=True) + 1e-8

            in_sim = (target_in_re / norm_target_in) @ (base_in_re / norm_base_in).T
            total_cost += 0.1 * (1.0 - np.clip(in_sim, -1.0, 1.0))

        return total_cost, primary_out_cost

    def align_group(
        self,
        base_out: np.ndarray,
        target_out: np.ndarray,
        base_in: Optional[np.ndarray] = None,
        target_in: Optional[np.ndarray] = None,
        base_vectors: Optional[List[np.ndarray]] = None,
        target_vectors: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Aligns a single permutation group.
        Returns:
            perm: 1D array of permutation indices for base tensor: base[perm] -> target
            confidence: float score between 0.0 and 1.0
            is_alignable: bool indicating if match is statistically meaningful
        """
        total_cost, primary_out_cost = self.compute_cost_matrix(
            base_out=base_out,
            target_out=target_out,
            base_in=base_in,
            target_in=target_in,
            base_vectors=base_vectors,
            target_vectors=target_vectors,
        )

        perm = solve_bipartite_matching(total_cost)

        N = total_cost.shape[0]
        aligned_out_cost = np.sum(primary_out_cost[np.arange(N), perm])
        mean_out_cost = aligned_out_cost / (N + 1e-8)

        confidence = float(np.clip(1.0 - mean_out_cost, 0.0, 1.0))
        is_alignable = confidence >= self.confidence_threshold

        return perm, confidence, is_alignable
