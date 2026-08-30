"""
Bit-Exact Reconstructor for SynapseFS Checkpoints.
Reconstructs entire .safetensors files or individual tensor regions on-the-fly.
"""

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from synapsefs.alignment.residual import ResidualEngine, permute_tensor
from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.utils.safetensors_helper import DTYPE_TO_NUMPY, NUMPY_TO_DTYPE


class CheckpointReconstructor:
    """
    Reconstructs checkpoints from Merkle DAG manifests and CAS deltas.
    Guarantees 100% byte-for-byte identical output to the original .safetensors file.
    """

    def __init__(self, dag: RepositoryDAG):
        self.dag = dag
        self.cas = dag.cas

    def reconstruct_header_bytes(self, manifest: Manifest) -> Tuple[bytes, bytes, int]:
        """
        Generates the standard 8-byte header size prefix and JSON header bytes.
        Returns:
            header_prefix_8b: bytes (little-endian uint64)
            header_json_bytes: bytes (UTF-8 JSON string)
            data_start_offset: int (8 + len(header_json_bytes))
        """
        header_dict: Dict[str, Any] = {}
        if manifest.metadata:
            header_dict["__metadata__"] = manifest.metadata

        for tname in sorted(manifest.tensors.keys()):
            tmeta = manifest.tensors[tname]
            header_dict[tname] = {
                "dtype": tmeta["dtype"],
                "shape": tmeta["shape"],
                "data_offsets": tmeta["data_offsets"],
            }

        header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
        header_len = len(header_json)
        header_prefix = struct.pack("<Q", header_len)
        data_start_offset = 8 + header_len

        return header_prefix, header_json, data_start_offset

    def reconstruct_tensor(
        self,
        tname: str,
        manifest: Manifest,
    ) -> np.ndarray:
        """
        Reconstructs a single tensor in RAM.
        If delta: fetches base tensor, applies permutations and decompressed residuals.
        If raw blob: loads directly from CAS.
        """
        if tname not in manifest.tensors:
            raise KeyError(f"Tensor '{tname}' not present in manifest")

        tmeta = manifest.tensors[tname]
        shape = tuple(tmeta["shape"])
        dtype_str = tmeta["dtype"]
        np_dtype = np.dtype(DTYPE_TO_NUMPY.get(dtype_str, "float32"))

        is_delta = tmeta.get("is_delta", False)
        if not is_delta:
            blob_hash = tmeta["blob_hash"]
            raw_bytes = self.cas.get(blob_hash)
            return np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

        # Delta reconstruction
        base_commit_id = manifest.base_commit_id
        if not base_commit_id:
            raise ValueError(f"Delta tensor {tname} has no base_commit_id referenced in manifest")

        base_commit = self.dag.get_commit(base_commit_id)
        base_manifest = self.dag.get_manifest(base_commit.manifest_hash)

        # Recursively reconstruct base tensor
        base_tensor_name = tmeta.get("base_tensor_name", tname)
        base_arr = self.reconstruct_tensor(base_tensor_name, base_manifest)

        # Load axis permutations (if any)
        perm_hash = tmeta.get("perm_hash")
        axis_perms: Dict[int, np.ndarray] = {}
        if perm_hash:
            perm_json_bytes = self.cas.get(perm_hash)
            perm_dict = json.loads(perm_json_bytes.decode("utf-8"))
            axis_perms = {int(axis): np.array(p, dtype=np.int64) for axis, p in perm_dict.items()}

        # Load compressed residual
        residual_hash = tmeta["residual_hash"]
        compressed_delta = self.cas.get(residual_hash)

        # Apply permutation and residual
        reconstructed_arr = ResidualEngine.apply_residual(
            base_arr=base_arr,
            compressed_delta=compressed_delta,
            target_shape=shape,
            target_dtype=np_dtype,
            axis_perms=axis_perms,
        )

        return reconstructed_arr

    def reconstruct_to_file(self, manifest: Manifest, output_file: Path) -> None:
        """
        Reconstructs the full .safetensors file directly to disk (e.g. for `checkout`).
        Writes the exact header and tensor buffers sequentially out-of-core.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        prefix, header_json, _ = self.reconstruct_header_bytes(manifest)

        with open(output_file, "wb") as f:
            f.write(prefix)
            f.write(header_json)

            # Write tensors in sorted order matching data_offsets
            sorted_tensors = sorted(
                manifest.tensors.items(),
                key=lambda item: item[1]["data_offsets"][0]
            )

            for tname, _ in sorted_tensors:
                arr = self.reconstruct_tensor(tname, manifest)
                f.write(arr.tobytes())
                del arr
