"""
Tests for Transparent Filesystem Access Layer (VFS, Byte Mapper, LRU Cache, and Concurrency).
"""

import concurrent.futures
import shutil
import tempfile
import unittest
from pathlib import Path

from synapsefs.alignment.out_of_core import OutOfCoreAlignmentEngine
from synapsefs.alignment.reconstructor import CheckpointReconstructor
from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.utils.safetensors_helper import read_safetensors_header
from synapsefs.vfs.byte_mapper import VirtualSafetensorsMapper
from synapsefs.vfs.lru_cache import LRUTensorCache
from synapsefs.vfs.vfs_emulator import VirtualFileHandle
from tests.fixtures.generate_models import generate_fixture_files


class TestVFS(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = generate_fixture_files(Path(self.temp_dir) / "fixtures")
        self.repo_dir = Path(self.temp_dir) / "repo"
        self.repo_dir.mkdir()
        self.dag = RepositoryDAG.init(self.repo_dir)
        self.cas = self.dag.cas

        # Set up base & delta commit
        base_path = self.fixtures["mlp_base"]
        h_size, header, d_start = read_safetensors_header(base_path)
        base_manifest_tensors = {}
        with open(base_path, "rb") as f:
            for tname, tmeta in header.items():
                if tname == "__metadata__":
                    continue
                start, end = tmeta["data_offsets"]
                f.seek(d_start + start)
                raw_bytes = f.read(end - start)
                blob_hash = self.cas.put(raw_bytes)
                base_manifest_tensors[tname] = {
                    "dtype": tmeta["dtype"],
                    "shape": tmeta["shape"],
                    "data_offsets": tmeta["data_offsets"],
                    "is_delta": False,
                    "blob_hash": blob_hash,
                }
        base_manifest = Manifest(tensors=base_manifest_tensors, metadata=header.get("__metadata__", {}))
        self.base_cid = self.dag.create_commit(self.cas.put(base_manifest.serialize()), message="Base")

        target_path = self.fixtures["mlp_target"]
        engine = OutOfCoreAlignmentEngine(self.cas)
        self.delta_manifest, _ = engine.align_and_diff(
            base_checkpoint_path=base_path,
            target_checkpoint_path=target_path,
            config_path=self.fixtures["config"],
            base_commit_id=self.base_cid,
        )
        self.delta_cid = self.dag.create_commit(self.cas.put(self.delta_manifest.serialize()), message="Delta", parent_ids=[self.base_cid])

        # Write exact ground-truth file for byte comparisons
        self.ground_truth_file = Path(self.temp_dir) / "ground_truth.safetensors"
        reconstructor = CheckpointReconstructor(self.dag)
        reconstructor.reconstruct_to_file(self.delta_manifest, self.ground_truth_file)
        with open(self.ground_truth_file, "rb") as f:
            self.ground_truth_bytes = f.read()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_byte_mapper_full_stream(self):
        cache = LRUTensorCache(max_bytes=10 * 1024 * 1024)
        mapper = VirtualSafetensorsMapper(self.dag, self.delta_manifest, cache=cache)
        
        self.assertEqual(mapper.total_file_size, len(self.ground_truth_bytes))
        streamed_bytes = mapper.read(0, mapper.total_file_size)
        self.assertEqual(streamed_bytes, self.ground_truth_bytes)

    def test_byte_mapper_random_slices(self):
        cache = LRUTensorCache(max_bytes=10 * 1024 * 1024)
        mapper = VirtualSafetensorsMapper(self.dag, self.delta_manifest, cache=cache)

        # Slice 1: First 8 bytes (Header size prefix)
        self.assertEqual(mapper.read(0, 8), self.ground_truth_bytes[0:8])

        # Slice 2: Header JSON region
        self.assertEqual(mapper.read(8, 100), self.ground_truth_bytes[8:108])

        # Slice 3: Boundary spanning across header and first tensor
        h_len = mapper.header_size
        self.assertEqual(mapper.read(h_len - 10, 50), self.ground_truth_bytes[h_len - 10 : h_len + 40])

        # Slice 4: Middle of tensor
        mid = mapper.total_file_size // 2
        self.assertEqual(mapper.read(mid, 256), self.ground_truth_bytes[mid : mid + 256])

    def test_concurrent_readers(self):
        cache = LRUTensorCache(max_bytes=10 * 1024 * 1024)
        mapper = VirtualSafetensorsMapper(self.dag, self.delta_manifest, cache=cache)

        def read_slice(offset, size):
            return mapper.read(offset, size)

        offsets_sizes = [
            (0, 8),
            (8, 200),
            (500, 1024),
            (1024, 2048),
            (0, mapper.total_file_size),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(read_slice, off, sz)
                for off, sz in offsets_sizes * 10
            ]
            for idx, future in enumerate(futures):
                off, sz = (offsets_sizes * 10)[idx]
                expected = self.ground_truth_bytes[off : off + sz]
                self.assertEqual(future.result(), expected)

    def test_vfs_file_handle_seeking(self):
        mapper = VirtualSafetensorsMapper(self.dag, self.delta_manifest)
        handle = VirtualFileHandle(mapper)

        handle.seek(100)
        self.assertEqual(handle.tell(), 100)
        chunk = handle.read(50)
        self.assertEqual(chunk, self.ground_truth_bytes[100:150])
        self.assertEqual(handle.tell(), 150)


if __name__ == "__main__":
    unittest.main()
