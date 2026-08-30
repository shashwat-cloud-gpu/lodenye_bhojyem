"""
Tests for Crash-Resilience, Mid-Write Interruption, and Journal Recovery.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.crash_resilience import TransactionJournal
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.core.lineage import LineageVerifier


class TestCrashResilience(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_dir = Path(self.temp_dir) / "repo"
        self.repo_dir.mkdir()
        self.dag = RepositoryDAG.init(self.repo_dir)
        self.cas = self.dag.cas
        self.journal = TransactionJournal(self.repo_dir / ".synapsefs")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_clean_state_on_mid_write_crash(self):
        # 1. Base commit established
        m0 = Manifest(tensors={"fc1": {"blob_hash": self.cas.put(b"clean_data"), "dtype": "F32", "shape": [10], "data_offsets": [0, 40]}})
        cid0 = self.dag.create_commit(self.cas.put(m0.serialize()), message="Initial Base")

        # 2. Simulate transaction started but crashed mid-write
        tx_id = self.journal.begin_transaction("commit", {"note": "simulated interrupted commit"})
        
        # Write some temp file to .synapsefs/tmp
        temp_file = self.repo_dir / ".synapsefs" / "tmp" / "orphaned_uncommitted.tmp"
        temp_file.write_bytes(b"corrupted_half_written_bytes")
        
        # Stale transaction is left uncommitted (as if SIGKILL occurred)

        # 3. Simulate system restart & recovery
        report = self.journal.recover(self.cas)
        self.assertEqual(report["cleaned_temp_files"], 1)
        self.assertEqual(report["aborted_transactions"], 1)
        self.assertFalse(temp_file.exists())

        # 4. Verify base repository remains 100% verified and intact
        verifier = LineageVerifier(self.dag)
        res = verifier.verify_commit_lineage(cid0)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.verified_commits, 1)


if __name__ == "__main__":
    unittest.main()
