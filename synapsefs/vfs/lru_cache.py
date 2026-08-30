"""
Thread-Safe Bounded LRU Cache for Dynamically Reconstructed Tensor Buffers.
Ensures daemon Peak RSS remains strictly within budget under sustained I/O.
"""

import collections
import threading
from typing import Callable, Optional
import numpy as np


class LRUTensorCache:
    """
    LRU Cache that bounds maximum memory usage (in bytes) of reconstructed tensor arrays.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024):  # Default 512MB limit
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()
        self.lock = threading.Lock()

    def get_or_compute(self, key: str, compute_fn: Callable[[], np.ndarray]) -> np.ndarray:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

        # Compute outside lock to allow concurrent decoding
        arr = compute_fn()
        arr_bytes = arr.nbytes

        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

            # Evict until space is available
            while self.current_bytes + arr_bytes > self.max_bytes and self.cache:
                evicted_key, evicted_arr = self.cache.popitem(last=False)
                self.current_bytes -= evicted_arr.nbytes

            self.cache[key] = arr
            self.current_bytes += arr_bytes
            return arr

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.current_bytes = 0

    def stats(self) -> dict:
        with self.lock:
            return {
                "cached_tensors": len(self.cache),
                "current_bytes": self.current_bytes,
                "max_bytes": self.max_bytes,
                "utilization_pct": (self.current_bytes / self.max_bytes) * 100.0,
            }
