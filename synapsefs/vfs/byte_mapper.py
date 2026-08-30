"""
Virtual Safetensors Byte-Offset Indexer.
Translates arbitrary POSIX read(offset, size) requests to on-the-fly tensor reconstructions.
Enforces STRICT ZERO PRE-MATERIALIZATION on disk.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from synapsefs.alignment.reconstructor import CheckpointReconstructor
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.vfs.lru_cache import LRUTensorCache


class TensorByteInterval:
    def __init__(self, name: str, start: int, end: int, data_offset_start: int, data_offset_end: int):
        self.name = name
        self.start = start                          # Global file offset start
        self.end = end                              # Global file offset end
        self.data_offset_start = data_offset_start  # Relative to data buffer start
        self.data_offset_end = data_offset_end      # Relative to data buffer end


class VirtualSafetensorsMapper:
    """
    Simulates a complete .safetensors binary file in-memory.
    Translates random byte-offset reads directly to on-demand reconstructed tensor slices.
    """

    def __init__(
        self,
        dag: RepositoryDAG,
        manifest: Manifest,
        cache: Optional[LRUTensorCache] = None,
    ):
        self.dag = dag
        self.manifest = manifest
        self.reconstructor = CheckpointReconstructor(dag)
        self.cache = cache or LRUTensorCache(max_bytes=512 * 1024 * 1024)

        # 1. Build Header in memory
        prefix, header_json, self.data_start_offset = self.reconstructor.reconstruct_header_bytes(manifest)
        self.header_bytes = prefix + header_json
        self.header_size = len(self.header_bytes)

        # 2. Build Byte Interval Index for all tensors
        self.intervals: List[TensorByteInterval] = []
        self.total_file_size = self.header_size

        # Sort tensors by offset
        sorted_tensors = sorted(
            manifest.tensors.items(),
            key=lambda item: item[1]["data_offsets"][0]
        )

        for tname, tmeta in sorted_tensors:
            rel_start, rel_end = tmeta["data_offsets"]
            global_start = self.data_start_offset + rel_start
            global_end = self.data_start_offset + rel_end

            interval = TensorByteInterval(
                name=tname,
                start=global_start,
                end=global_end,
                data_offset_start=rel_start,
                data_offset_end=rel_end,
            )
            self.intervals.append(interval)
            if global_end > self.total_file_size:
                self.total_file_size = global_end

    def read(self, offset: int, size: int) -> bytes:
        """
        Serves POSIX read(offset, size) on-the-fly.
        Never writes any reconstructed data to disk.
        """
        if offset < 0 or size <= 0:
            return b""
        if offset >= self.total_file_size:
            return b""

        end_offset = min(offset + size, self.total_file_size)
        result_chunks: List[bytes] = []
        current_pos = offset

        # 1. Check if request overlaps header
        if current_pos < self.header_size:
            header_end = min(end_offset, self.header_size)
            result_chunks.append(self.header_bytes[current_pos:header_end])
            current_pos = header_end

        # 2. Process overlapping tensor intervals
        if current_pos < end_offset:
            for interval in self.intervals:
                if current_pos >= end_offset:
                    break

                # If interval is before current_pos, skip
                if interval.end <= current_pos:
                    continue

                # If interval starts after current_pos, handle gap (if any padding)
                if interval.start > current_pos:
                    gap_size = min(interval.start, end_offset) - current_pos
                    result_chunks.append(b"\x00" * gap_size)
                    current_pos += gap_size
                    if current_pos >= end_offset:
                        break

                # Overlap with interval [interval.start, interval.end)
                read_start = max(current_pos, interval.start)
                read_end = min(end_offset, interval.end)
                read_len = read_end - read_start

                if read_len > 0:
                    # Fetch / reconstruct tensor from cache or CAS
                    tensor_name = interval.name
                    tensor_arr = self.cache.get_or_compute(
                        f"tensor_{tensor_name}",
                        lambda: self.reconstructor.reconstruct_tensor(tensor_name, self.manifest)
                    )

                    # Byte offset within this tensor's contiguous buffer
                    tensor_byte_offset = read_start - interval.start
                    tensor_bytes = tensor_arr.tobytes()
                    chunk = tensor_bytes[tensor_byte_offset : tensor_byte_offset + read_len]
                    result_chunks.append(chunk)

                    current_pos += read_len

        return b"".join(result_chunks)
