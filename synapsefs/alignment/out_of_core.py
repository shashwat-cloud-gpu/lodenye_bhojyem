"""
Out-of-Core Layer-by-Layer Alignment & Diffing Engine.
Processes checkpoints from 50M to 7B parameters strictly within consumer RAM limits (16GB max).
Applies forward permutation propagation across sequential layers.
"""

import gc
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from synapsefs.alignment.matcher import IsomorphicMatcher
from synapsefs.alignment.residual import ResidualEngine, permute_tensor
from synapsefs.alignment.topology import ModelTopology
from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.dag import Manifest
from synapsefs.utils.safetensors_helper import (
    DTYPE_TO_NUMPY,
    load_tensor_lazy,
    read_safetensors_header,
)


class OutOfCoreAlignmentEngine:
    """
    Sequentially streams tensors from disk to align permutations and compute delta manifests.
    Propagates input permutations forward to achieve near-zero residuals on fine-tuned models.
    """

    def __init__(self, cas: ContentAddressableStore, confidence_threshold: float = 0.05):
        self.cas = cas
        self.matcher = IsomorphicMatcher(confidence_threshold=confidence_threshold)

    def align_and_diff(
        self,
        base_checkpoint_path: Path,
        target_checkpoint_path: Path,
        config_path: Optional[Path] = None,
        base_commit_id: Optional[str] = None,
    ) -> Tuple[Manifest, Dict[str, Any]]:
        """
        Performs out-of-core alignment between base and target checkpoints.
        Returns:
            manifest: Generated Manifest object referencing CAS blobs
            stats: Dict containing alignment metrics, residual ratios, and memory stats
        """
        # 1. Parse headers lazily without loading weights into RAM
        base_h_size, base_header, base_data_start = read_safetensors_header(base_checkpoint_path)
        tgt_h_size, tgt_header, tgt_data_start = read_safetensors_header(target_checkpoint_path)

        base_info = (base_h_size, base_header, base_data_start)
        tgt_info = (tgt_h_size, tgt_header, tgt_data_start)

        tgt_shapes = {
            k: v["shape"] for k, v in tgt_header.items() if k != "__metadata__" and isinstance(v, dict)
        }

        # 2. Build topology dependency graph
        topology = ModelTopology.from_config_file(config_path) if config_path else ModelTopology()
        groups = topology.build_from_tensors(tgt_shapes)

        # 3. Layer-by-Layer Forward Permutation Propagation & Recovery
        discovered_perms: Dict[str, np.ndarray] = {}
        group_confidences: Dict[str, float] = {}
        tensor_axis_perms: Dict[str, Dict[int, np.ndarray]] = {}
        all_alignable = True

        for grp in groups:
            if not grp.output_tensors:
                continue
            out_tname, out_axis = grp.output_tensors[0]
            if out_tname not in base_header or out_tname not in tgt_header:
                continue

            # Lazy load primary output tensor pair
            base_out = load_tensor_lazy(str(base_checkpoint_path), out_tname, base_info)
            tgt_out = load_tensor_lazy(str(target_checkpoint_path), out_tname, tgt_info)

            # Apply any previously discovered input permutations (e.g. columns / input channels)
            if out_tname in tensor_axis_perms:
                base_out = permute_tensor(base_out, tensor_axis_perms[out_tname])

            # Lazy load vector parameters (bias, norm stats)
            base_vecs, tgt_vecs = [], []
            for vname in grp.vector_tensors:
                if vname in base_header and vname in tgt_header:
                    bv = load_tensor_lazy(str(base_checkpoint_path), vname, base_info)
                    tv = load_tensor_lazy(str(target_checkpoint_path), vname, tgt_info)
                    if vname in tensor_axis_perms:
                        bv = permute_tensor(bv, tensor_axis_perms[vname])
                    base_vecs.append(bv)
                    tgt_vecs.append(tv)

            # Run alignment
            perm, conf, is_alignable = self.matcher.align_group(
                base_out=base_out,
                target_out=tgt_out,
                base_in=None,
                target_in=None,
                base_vectors=base_vecs,
                target_vectors=tgt_vecs,
            )

            discovered_perms[grp.group_id] = perm
            group_confidences[grp.group_id] = conf
            if not is_alignable:
                all_alignable = False

            # Propagate discovered permutation to tensor axes
            for tname, axis in grp.output_tensors:
                if tname not in tensor_axis_perms:
                    tensor_axis_perms[tname] = {}
                tensor_axis_perms[tname][axis] = perm

            for vname in grp.vector_tensors:
                if vname not in tensor_axis_perms:
                    tensor_axis_perms[vname] = {}
                tensor_axis_perms[vname][0] = perm

            for tname, axis in grp.input_tensors:
                if tname not in tensor_axis_perms:
                    tensor_axis_perms[tname] = {}
                tensor_axis_perms[tname][axis] = perm

            del base_out, tgt_out, base_vecs, tgt_vecs
            gc.collect()

        # 4. Out-of-Core Residual Delta Computation & CAS Blob Storage
        manifest_tensors: Dict[str, Dict[str, Any]] = {}
        total_target_bytes = 0
        total_residual_bytes = 0
        energy_ratios: List[float] = []

        target_metadata = tgt_header.get("__metadata__", {})

        for tname, tmeta in tgt_header.items():
            if tname == "__metadata__":
                continue

            shape = tmeta["shape"]
            dtype_str = tmeta["dtype"]
            byte_len = tmeta["data_offsets"][1] - tmeta["data_offsets"][0]
            total_target_bytes += byte_len

            # Load target tensor lazily
            target_arr = load_tensor_lazy(str(target_checkpoint_path), tname, tgt_info)

            if tname in base_header and base_header[tname]["shape"] == shape:
                base_arr = load_tensor_lazy(str(base_checkpoint_path), tname, base_info)
                axis_perms = tensor_axis_perms.get(tname)

                compressed_delta, energy_ratio = ResidualEngine.compute_residual(
                    base_arr=base_arr,
                    target_arr=target_arr,
                    axis_perms=axis_perms,
                )

                residual_hash = self.cas.put(compressed_delta)
                total_residual_bytes += len(compressed_delta)
                energy_ratios.append(energy_ratio)

                perm_hash = None
                if axis_perms:
                    perm_data = {
                        str(axis): p.tolist() for axis, p in axis_perms.items()
                    }
                    perm_bytes = json.dumps(perm_data).encode("utf-8")
                    perm_hash = self.cas.put(perm_bytes)

                manifest_tensors[tname] = {
                    "dtype": dtype_str,
                    "shape": shape,
                    "data_offsets": tmeta["data_offsets"],
                    "is_delta": True,
                    "base_tensor_name": tname,
                    "residual_hash": residual_hash,
                    "perm_hash": perm_hash,
                    "uncompressed_bytes": byte_len,
                    "compressed_bytes": len(compressed_delta),
                }

                del base_arr
            else:
                raw_bytes = target_arr.tobytes()
                blob_hash = self.cas.put(raw_bytes)
                total_residual_bytes += len(raw_bytes)

                manifest_tensors[tname] = {
                    "dtype": dtype_str,
                    "shape": shape,
                    "data_offsets": tmeta["data_offsets"],
                    "is_delta": False,
                    "blob_hash": blob_hash,
                    "uncompressed_bytes": byte_len,
                    "compressed_bytes": len(raw_bytes),
                }

            del target_arr
            gc.collect()

        manifest = Manifest(
            tensors=manifest_tensors,
            metadata=target_metadata,
            config_hash=None,
            base_commit_id=base_commit_id,
        )

        residual_ratio = (
            float(total_residual_bytes / total_target_bytes) if total_target_bytes > 0 else 1.0
        )
        mean_confidence = (
            float(np.mean(list(group_confidences.values()))) if group_confidences else 1.0
        )

        stats = {
            "total_target_bytes": total_target_bytes,
            "total_residual_bytes": total_residual_bytes,
            "residual_ratio": residual_ratio,
            "mean_confidence": mean_confidence,
            "is_alignable": all_alignable,
            "num_permutation_groups": len(groups),
            "energy_ratios": energy_ratios,
        }

        return manifest, stats
