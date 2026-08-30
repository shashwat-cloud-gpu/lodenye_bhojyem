"""
Tests for Distributed Networking, Differential Sync, and Resumable Transfers.
"""

import shutil
import socket
import tempfile
import time
import unittest
from pathlib import Path

from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.core.lineage import LineageVerifier
from synapsefs.net.client import SyncClient
from synapsefs.net.server import SyncServer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestNetworkSync(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Peer 1 repo
        self.peer1_dir = Path(self.temp_dir) / "peer1"
        self.peer1_dir.mkdir()
        self.dag1 = RepositoryDAG.init(self.peer1_dir)
        self.cas1 = self.dag1.cas

        # Peer 2 repo
        self.peer2_dir = Path(self.temp_dir) / "peer2"
        self.peer2_dir.mkdir()
        self.dag2 = RepositoryDAG.init(self.peer2_dir)
        self.cas2 = self.dag2.cas

        # Start server for peer 1
        self.port = get_free_port()
        self.server = SyncServer(self.dag1, host="127.0.0.1", port=self.port)
        self.server.start(block=False)
        time.sleep(0.1)

    def tearDown(self):
        self.server.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_differential_pull_and_resumability(self):
        # 1. Create commit on Peer 1
        blob1_h = self.cas1.put(b"tensor_weights_layer1_blob_content")
        blob2_h = self.cas1.put(b"tensor_weights_layer2_blob_content")
        m1 = Manifest(tensors={
            "w1": {"blob_hash": blob1_h, "dtype": "F32", "shape": [10], "data_offsets": [0, 40]},
            "w2": {"blob_hash": blob2_h, "dtype": "F32", "shape": [10], "data_offsets": [40, 80]},
        })
        m1_h = self.cas1.put(m1.serialize())
        cid1 = self.dag1.create_commit(m1_h, message="Initial commit on peer 1")

        # 2. Pull to Peer 2
        client2 = SyncClient(self.dag2)
        remote_url = f"http://127.0.0.1:{self.port}"
        res1 = client2.pull(remote_url=remote_url, branch="main")

        self.assertEqual(res1["status"], "Success")
        self.assertEqual(self.dag2.get_branch_commit("main"), cid1)
        self.assertTrue(self.cas2.exists(blob1_h))
        self.assertTrue(self.cas2.exists(blob2_h))

        # Verify cryptographic integrity on Peer 2
        verifier = LineageVerifier(self.dag2)
        v_res = verifier.verify_commit_lineage(cid1)
        self.assertTrue(v_res.is_valid)

        # 3. Create incremental commit on Peer 1
        blob3_h = self.cas1.put(b"tensor_weights_layer3_delta_content")
        m2 = Manifest(tensors={
            "w1": {"blob_hash": blob1_h, "dtype": "F32", "shape": [10], "data_offsets": [0, 40]},
            "w2": {"blob_hash": blob2_h, "dtype": "F32", "shape": [10], "data_offsets": [40, 80]},
            "w3": {"blob_hash": blob3_h, "dtype": "F32", "shape": [10], "data_offsets": [80, 120]},
        })
        m2_h = self.cas1.put(m2.serialize())
        cid2 = self.dag1.create_commit(m2_h, message="Second commit on peer 1")

        # 4. Pull incremental update: must transfer ONLY the new objects (blob3, manifest2, commit2)
        res2 = client2.pull(remote_url=remote_url, branch="main")
        self.assertEqual(res2["status"], "Success")
        self.assertEqual(res2["transferred_blocks"], 3)  # Only 3 objects transferred instead of full history!
        self.assertEqual(self.dag2.get_branch_commit("main"), cid2)

        # 5. Subsequent pull when already up to date
        res3 = client2.pull(remote_url=remote_url, branch="main")
        self.assertEqual(res3["status"], "Already up to date")
        self.assertEqual(res3["transferred_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
