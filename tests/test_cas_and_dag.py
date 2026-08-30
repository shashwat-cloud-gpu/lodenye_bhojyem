"""
Unit and Integration Tests for CAS, Merkle DAG, Branching, Lineage Verification, and 3-Way Merge.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from synapsefs.core.cas import ContentAddressableStore, compute_hash
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.core.lineage import LineageVerifier


class TestCASAndDAG(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_dir = Path(self.temp_dir) / "repo"
        self.repo_dir.mkdir()
        self.dag = RepositoryDAG.init(self.repo_dir)
        self.cas = self.dag.cas

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cas_put_get_and_immutability(self):
        data = b"synapsefs_tensor_weights_raw_bytes_12345"
        h = self.cas.put(data)
        self.assertTrue(self.cas.exists(h))
        retrieved = self.cas.get(h)
        self.assertEqual(data, retrieved)

        # Re-putting identical content should return same hash
        h2 = self.cas.put(data)
        self.assertEqual(h, h2)

    def test_tamper_detection(self):
        data = b"important_model_weights"
        h = self.cas.put(data)
        
        # Tamper the blob on disk directly
        obj_path = self.cas._get_object_path(h)
        with open(obj_path, "wb") as f:
            f.write(b"corrupted_tampered_bytes")

        # CAS get should raise ValueError
        with self.assertRaises(ValueError):
            self.cas.get(h)

    def test_commit_lineage_and_branches(self):
        # Create manifest
        manifest = Manifest(
            tensors={"fc.weight": {"dtype": "F32", "shape": [10, 10], "data_offsets": [0, 400], "blob_hash": self.cas.put(b"w1")}},
            metadata={"version": "1.0"},
        )
        m_hash = self.cas.put(manifest.serialize())
        
        # Commit 1 on main
        cid1 = self.dag.create_commit(m_hash, message="Commit 1")
        self.assertEqual(self.dag.get_branch_commit("main"), cid1)

        # Create branch 'feature'
        self.dag.update_branch("feature", cid1)
        self.assertIn("feature", self.dag.list_branches())

        # Commit 2 on feature
        self.dag.set_head_branch("feature")
        manifest2 = Manifest(
            tensors={"fc.weight": {"dtype": "F32", "shape": [10, 10], "data_offsets": [0, 400], "blob_hash": self.cas.put(b"w2")}},
            metadata={"version": "2.0"},
        )
        m2_hash = self.cas.put(manifest2.serialize())
        cid2 = self.dag.create_commit(m2_hash, message="Commit 2 on feature")
        self.assertEqual(self.dag.get_branch_commit("feature"), cid2)

        # Lineage verification
        verifier = LineageVerifier(self.dag)
        res = verifier.verify_commit_lineage(cid2)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.verified_commits, 2)

    def test_three_way_merge(self):
        # Base commit
        m0 = Manifest(tensors={"fc1": {"blob_hash": self.cas.put(b"v0"), "data_offsets": [0, 10], "dtype": "F32", "shape": [5]}})
        cid0 = self.dag.create_commit(self.cas.put(m0.serialize()), message="Base")

        # Branch A modifies fc1
        self.dag.update_branch("branchA", cid0)
        self.dag.set_head_branch("branchA")
        m_a = Manifest(tensors={"fc1": {"blob_hash": self.cas.put(b"vA"), "data_offsets": [0, 10], "dtype": "F32", "shape": [5]}})
        cid_a = self.dag.create_commit(self.cas.put(m_a.serialize()), message="Branch A edit")

        # Branch B adds fc2
        self.dag.update_branch("branchB", cid0)
        self.dag.set_head_branch("branchB")
        m_b = Manifest(tensors={
            "fc1": {"blob_hash": self.cas.put(b"v0"), "data_offsets": [0, 10], "dtype": "F32", "shape": [5]},
            "fc2": {"blob_hash": self.cas.put(b"vB_new"), "data_offsets": [10, 20], "dtype": "F32", "shape": [5]},
        })
        cid_b = self.dag.create_commit(self.cas.put(m_b.serialize()), message="Branch B edit")

        # Merge branchB into branchA
        self.dag.set_head_branch("branchA")
        status, merge_cid = self.dag.merge_branches("branchB")
        self.assertEqual(status, "Merged")

        merge_commit = self.dag.get_commit(merge_cid)
        self.assertEqual(len(merge_commit.parent_ids), 2)

        merged_manifest = self.dag.get_manifest(merge_commit.manifest_hash)
        self.assertIn("fc1", merged_manifest.tensors)
        self.assertIn("fc2", merged_manifest.tensors)
        self.assertEqual(merged_manifest.tensors["fc1"]["blob_hash"], self.cas.put(b"vA"))
        self.assertEqual(merged_manifest.tensors["fc2"]["blob_hash"], self.cas.put(b"vB_new"))


if __name__ == "__main__":
    unittest.main()
