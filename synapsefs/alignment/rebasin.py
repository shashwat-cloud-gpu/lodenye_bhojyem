"""
Git Re-Basin: Model Merging Modulo Permutation Symmetries.
Implementation of Algorithm 1 (Coordinate Ascent Weight Matching) from Ainsworth et al. (NeurIPS 2022, arXiv:2209.04836).
"""

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from synapsefs.alignment.matcher import solve_bipartite_matching
from synapsefs.alignment.residual import permute_tensor
from synapsefs.alignment.topology import ModelTopology, PermutationGroup


class GitReBasinEngine:
    """
    Implements multi-pass Coordinate Ascent Weight Matching (Algorithm 1 in Ainsworth et al.).
    Iteratively optimizes layer permutation matrices P_1, P_2, ... P_L to minimize distance in weight space.
    """

    def __init__(self, max_iter: int = 15, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def flatten_conv_out(tensor: np.ndarray) -> np.ndarray:
        """Flattens (C_out, C_in, Kh, Kw) -> (C_out, C_in * Kh * Kw)"""
        if tensor.ndim == 4:
            return tensor.reshape(tensor.shape[0], -1)
        elif tensor.ndim == 2:
            return tensor
        else:
            return tensor.reshape(tensor.shape[0], -1)

    @staticmethod
    def flatten_conv_in(tensor: np.ndarray) -> np.ndarray:
        """Flattens (C_next, C_in, Kh, Kw) -> (C_in, C_next * Kh * Kw)"""
        if tensor.ndim == 4:
            # swap axes so C_in is first
            swapped = np.swapaxes(tensor, 0, 1)
            return swapped.reshape(tensor.shape[1], -1)
        elif tensor.ndim == 2:
            return np.swapaxes(tensor, 0, 1)
        else:
            return np.swapaxes(tensor, 0, 1).reshape(tensor.shape[1], -1)

    def coordinate_ascent_weight_matching(
        self,
        tensors_a: Dict[str, np.ndarray],
        tensors_b: Dict[str, np.ndarray],
        groups: List[PermutationGroup],
    ) -> Dict[str, np.ndarray]:
        """
        Runs iterative Coordinate Ascent over permutation groups holding other layers fixed.
        Returns:
            discovered_perms: Dict[group_id -> 1D permutation array]
        """
        # Initialize all permutations to Identity
        perms: Dict[str, np.ndarray] = {
            grp.group_id: np.arange(grp.dim_size, dtype=np.int64) for grp in groups
        }

        # Build reverse index for quick lookup of which groups affect which tensor axes
        # tensor_name -> {axis: group_id}
        tensor_group_map: Dict[str, Dict[int, str]] = {}
        for grp in groups:
            for tname, axis in grp.output_tensors:
                if tname not in tensor_group_map:
                    tensor_group_map[tname] = {}
                tensor_group_map[tname][axis] = grp.group_id

            for vname in grp.vector_tensors:
                if vname not in tensor_group_map:
                    tensor_group_map[vname] = {}
                tensor_group_map[vname][0] = grp.group_id

            for tname, axis in grp.input_tensors:
                if tname not in tensor_group_map:
                    tensor_group_map[tname] = {}
                tensor_group_map[tname][axis] = grp.group_id

        # Coordinate ascent iteration loop
        for iteration in range(self.max_iter):
            progress = 0
            perm_changed = False

            for grp in groups:
                gid = grp.group_id
                N = grp.dim_size

                # Construct Affinity Matrix M of shape (N, N)
                # M[i, j] measures affinity between neuron i in Model A and neuron j in Model B
                affinity_matrix = np.zeros((N, N), dtype=np.float64)

                # 1. Output weights term: W_l^(A) * (W_l^(B) with other axis perms applied)^T
                for out_tname, out_axis in grp.output_tensors:
                    if out_tname not in tensors_a or out_tname not in tensors_b:
                        continue
                    w_a = tensors_a[out_tname]
                    w_b = tensors_b[out_tname]

                    # Apply other active permutations to w_b (except current group out_axis)
                    other_perms = {}
                    for ax, other_gid in tensor_group_map.get(out_tname, {}).items():
                        if ax != out_axis and other_gid in perms:
                            other_perms[ax] = perms[other_gid]

                    if other_perms:
                        w_b_perm = permute_tensor(w_b, other_perms)
                    else:
                        w_b_perm = w_b

                    w_a_flat = self.flatten_conv_out(w_a).astype(np.float64)
                    w_b_flat = self.flatten_conv_out(w_b_perm).astype(np.float64)

                    # Matrix product: (N, D) @ (D, N) -> (N, N)
                    affinity_matrix += (w_a_flat @ w_b_flat.T)

                # 2. Vector parameters term (bias, norm scales, stats)
                for vname in grp.vector_tensors:
                    if vname not in tensors_a or vname not in tensors_b:
                        continue
                    v_a = tensors_a[vname].reshape(N, 1).astype(np.float64)
                    v_b = tensors_b[vname].reshape(1, N).astype(np.float64)
                    # Outer product adds affinity for matching values
                    affinity_matrix += (v_a @ v_b)

                # 3. Subsequent input weights term: (W_{l+1}^(A))^T * (W_{l+1}^(B) with other perms)
                for in_tname, in_axis in grp.input_tensors:
                    if in_tname not in tensors_a or in_tname not in tensors_b:
                        continue
                    w_next_a = tensors_a[in_tname]
                    w_next_b = tensors_b[in_tname]

                    other_perms = {}
                    for ax, other_gid in tensor_group_map.get(in_tname, {}).items():
                        if ax != in_axis and other_gid in perms:
                            other_perms[ax] = perms[other_gid]

                    if other_perms:
                        w_next_b_perm = permute_tensor(w_next_b, other_perms)
                    else:
                        w_next_b_perm = w_next_b

                    w_next_a_in = self.flatten_conv_in(w_next_a).astype(np.float64)
                    w_next_b_in = self.flatten_conv_in(w_next_b_perm).astype(np.float64)

                    affinity_matrix += (w_next_a_in @ w_next_b_in.T)

                # Convert maximization of Tr(P * M) to minimization of Cost Matrix
                cost_matrix = -affinity_matrix
                cost_matrix -= np.min(cost_matrix)  # ensure non-negative

                # Solve Linear Sum Assignment Problem (LSAP)
                new_perm = solve_bipartite_matching(cost_matrix)

                if not np.array_equal(perms[gid], new_perm):
                    perms[gid] = new_perm
                    perm_changed = True
                    progress += 1

            if not perm_changed:
                break

        return perms

    def rebasin(
        self,
        tensors_a: Dict[str, np.ndarray],
        tensors_b: Dict[str, np.ndarray],
        topology: Optional[ModelTopology] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Re-basins Model B to align with Model A modulo permutation symmetries.
        Returns:
            rebasined_b: Model B weights permuted into Model A's basin (B~ = pi*(B))
            perms: Discovered permutation maps
        """
        shapes = {k: list(v.shape) for k, v in tensors_a.items()}
        if topology is None:
            topology = ModelTopology()
        if not topology.groups:
            groups = topology.build_from_tensors(shapes)
        else:
            groups = topology.groups

        # Run Coordinate Ascent Weight Matching
        perms = self.coordinate_ascent_weight_matching(tensors_a, tensors_b, groups)


        # Build tensor axis permutations map
        tensor_axis_perms: Dict[str, Dict[int, np.ndarray]] = {}
        for grp in groups:
            p = perms.get(grp.group_id)
            if p is None:
                continue
            for tname, axis in grp.output_tensors:
                if tname not in tensor_axis_perms:
                    tensor_axis_perms[tname] = {}
                tensor_axis_perms[tname][axis] = p

            for vname in grp.vector_tensors:
                if vname not in tensor_axis_perms:
                    tensor_axis_perms[vname] = {}
                tensor_axis_perms[vname][0] = p

            for tname, axis in grp.input_tensors:
                if tname not in tensor_axis_perms:
                    tensor_axis_perms[tname] = {}
                tensor_axis_perms[tname][axis] = p

        # Permute Model B tensors into Model A's coordinate basin
        rebasined_b: Dict[str, np.ndarray] = {}
        for tname, arr in tensors_b.items():
            if tname in tensor_axis_perms:
                rebasined_b[tname] = permute_tensor(arr, tensor_axis_perms[tname])
            else:
                rebasined_b[tname] = arr.copy()

        return rebasined_b, perms

    def interpolate(
        self,
        tensors_a: Dict[str, np.ndarray],
        rebasined_b: Dict[str, np.ndarray],
        alpha: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Performs linear mode interpolation in weight space:
        W_alpha = (1 - alpha) * W_A + alpha * W_B_rebasined
        """
        merged: Dict[str, np.ndarray] = {}
        for tname in tensors_a.keys():
            if tname in rebasined_b:
                arr_a = tensors_a[tname]
                arr_b = rebasined_b[tname]
                if arr_a.shape == arr_b.shape:
                    # Linear interpolation
                    merged[tname] = ((1.0 - alpha) * arr_a.astype(np.float32) + alpha * arr_b.astype(np.float32)).astype(arr_a.dtype)
                else:
                    merged[tname] = arr_a.copy()
            else:
                merged[tname] = tensors_a[tname].copy()
        return merged
