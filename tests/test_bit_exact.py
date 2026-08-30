"""
Bit-Exact Reconstruction Verification Test Suite.
Guarantees reconstructed .safetensors files are 100% byte-for-byte identical to the original target.
"""

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from synapsefs.alignment.out_of_core import OutOfCoreAlignmentEngine
from synapsefs.alignment.reconstructor import CheckpointReconstructor
from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.utils.safetensors_helper import read_safetensors_header
from tests.fixtures.generate_models import generate_fixture_files


def file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestBitExactReconstruction(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = generate_fixture_files(Path(self.temp_dir) / "fixtures")
        self.repo_dir = Path(self.temp_dir) / "repo"
        self.repo_dir.mkdir()
        self.dag = RepositoryDAG.init(self.repo_dir)
        self.cas = self.dag.cas

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mlp_bit_exact_reconstruction(self):
        # 1. Commit Base Model
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
        base_cid = self.dag.create_commit(self.cas.put(base_manifest.serialize()), message="Base MLP")

        # 2. Align & Commit Target Model (Delta)
        target_path = self.fixtures["mlp_target"]
        engine = OutOfCoreAlignmentEngine(self.cas)
        delta_manifest, stats = engine.align_and_diff(
            base_checkpoint_path=base_path,
            target_checkpoint_path=target_path,
            config_path=self.fixtures["config"],
            base_commit_id=base_cid,
        )
        delta_cid = self.dag.create_commit(self.cas.put(delta_manifest.serialize()), message="Delta MLP", parent_ids=[base_cid])

        # 3. Reconstruct Target Checkpoint
        reconstructor = CheckpointReconstructor(self.dag)
        reconstructed_file = Path(self.temp_dir) / "reconstructed_mlp_target.safetensors"
        reconstructor.reconstruct_to_file(delta_manifest, reconstructed_file)

        # 4. Compare Exact SHA-256 Hashes
        orig_hash = file_sha256(target_path)
        recon_hash = file_sha256(reconstructed_file)

        self.assertEqual(
            orig_hash,
            recon_hash,
            f"Reconstructed file is not bit-exact!\nOriginal:      {orig_hash}\nReconstructed: {recon_hash}"
        )

    def test_cnn_bit_exact_reconstruction(self):
        # 1. Commit Base CNN
        base_path = self.fixtures["cnn_base"]
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
        base_cid = self.dag.create_commit(self.cas.put(base_manifest.serialize()), message="Base CNN")

        # 2. Align & Commit Target CNN
        target_path = self.fixtures["cnn_target"]
        engine = OutOfCoreAlignmentEngine(self.cas)
        delta_manifest, _ = engine.align_and_diff(
            base_checkpoint_path=base_path,
            target_checkpoint_path=target_path,
            config_path=self.fixtures["config"],
            base_commit_id=base_cid,
        )
        delta_cid = self.dag.create_commit(self.cas.put(delta_manifest.serialize()), message="Delta CNN", parent_ids=[base_cid])

        # 3. Reconstruct Target Checkpoint
        reconstructor = CheckpointReconstructor(self.dag)
        reconstructed_file = Path(self.temp_dir) / "reconstructed_cnn_target.safetensors"
        reconstructor.reconstruct_to_file(delta_manifest, reconstructed_file)

        # 4. Compare Exact SHA-256 Hashes
        orig_hash = file_sha256(target_path)
        recon_hash = file_sha256(reconstructed_file)

        self.assertEqual(orig_hash, recon_hash)


if __name__ == "__main__":
    unittest.main()
