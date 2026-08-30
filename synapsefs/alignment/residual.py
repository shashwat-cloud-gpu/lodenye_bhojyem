"""
Lossless Bit-Exact Residual Delta Compressor & Permutation Encoder.
Uses bitwise XOR-differential encoding + Zstandard for guaranteed 100% IEEE-754 bit-exactness.
"""

import json
import struct
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import zlib


def compress_bytes(data: bytes, level: int = 3) -> bytes:
    """Compresses raw bytes using Zstandard (or zlib as fallback)."""
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=level)
        return b"ZSTD" + cctx.compress(data)
    else:
        return b"ZLIB" + zlib.compress(data, level=level)


def decompress_bytes(data: bytes) -> bytes:
    """Decompresses byte buffer."""
    if data.startswith(b"ZSTD"):
        if HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data[4:])
        else:
            raise ImportError("zstandard is required to decompress ZSTD payload")
    elif data.startswith(b"ZLIB"):
        return zlib.decompress(data[4:])
    else:
        return data


def permute_tensor(
    tensor: np.ndarray,
    axis_permutations: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Applies permutation arrays along specified axes of a numpy array.
    axis_permutations: {axis_index: permutation_indices}
    """
    result = tensor
    for axis, perm in axis_permutations.items():
        result = np.take(result, perm, axis=axis)
    return np.ascontiguousarray(result)


class ResidualEngine:
    """
    Computes bit-exact residual deltas using bitwise XOR difference.
    Guarantees mathematically exact roundtrips across all architectures and precisions.
    """

    @staticmethod
    def compute_residual(
        base_arr: np.ndarray,
        target_arr: np.ndarray,
        axis_perms: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[bytes, float]:
        """
        Permutes base_arr, calculates bitwise XOR delta against target_arr, and compresses it.
        Returns:
            compressed_delta_bytes: bytes
            residual_energy_ratio: float
        """
        if axis_perms:
            aligned_base = permute_tensor(base_arr, axis_perms)
        else:
            aligned_base = np.ascontiguousarray(base_arr)

        target_contiguous = np.ascontiguousarray(target_arr)

        # Convert to raw byte uint8 views for bitwise XOR
        base_u8 = np.frombuffer(aligned_base.tobytes(), dtype=np.uint8)
        tgt_u8 = np.frombuffer(target_contiguous.tobytes(), dtype=np.uint8)

        # Bitwise XOR delta
        xor_delta = np.bitwise_xor(base_u8, tgt_u8)

        # Calculate residual energy ratio
        base_norm = np.linalg.norm(aligned_base.astype(np.float32)) + 1e-8
        diff_float = target_contiguous.astype(np.float32) - aligned_base.astype(np.float32)
        delta_norm = np.linalg.norm(diff_float)
        energy_ratio = float(delta_norm / base_norm)

        # Compress bitwise XOR delta
        compressed_bytes = compress_bytes(xor_delta.tobytes())

        return compressed_bytes, energy_ratio

    @staticmethod
    def apply_residual(
        base_arr: np.ndarray,
        compressed_delta: bytes,
        target_shape: Tuple[int, ...],
        target_dtype: np.dtype,
        axis_perms: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Reconstructs target tensor from base_arr, axis permutations, and compressed XOR delta.
        Guarantees 100% bit-exact reconstruction.
        """
        if axis_perms:
            aligned_base = permute_tensor(base_arr, axis_perms)
        else:
            aligned_base = np.ascontiguousarray(base_arr)

        # Decompress XOR delta
        raw_xor_bytes = decompress_bytes(compressed_delta)
        xor_u8 = np.frombuffer(raw_xor_bytes, dtype=np.uint8)
        base_u8 = np.frombuffer(aligned_base.tobytes(), dtype=np.uint8)

        # Bitwise XOR reconstruction
        reconstructed_u8 = np.bitwise_xor(base_u8, xor_u8)

        # Reinterpret as original dtype and reshape
        reconstructed_arr = np.frombuffer(reconstructed_u8.tobytes(), dtype=target_dtype).reshape(target_shape)
        return np.ascontiguousarray(reconstructed_arr)
