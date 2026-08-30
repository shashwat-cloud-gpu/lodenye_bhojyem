"""
Tests for Isomorphic Alignment Engine: Permutation Recovery, Residual Compression, and Out-of-Core Processing.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from synapsefs.alignment.matcher import IsomorphicMatcher, solve_bipartite_matching
from synapsefs.alignment.out_of_core import OutOfCoreAlignmentEngine
from synapsefs.core.cas import ContentAddressableStore
from tests.fixtures.generate_models import generate_fixture_files


class TestAlignment(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = generate_fixture_files(Path(self.temp_dir) / "fixtures")
        self.cas_dir = Path(self.temp_dir) / "cas"
        self.cas = ContentAddressableStore(self.cas_dir)
        self.engine = OutOfCoreAlignmentEngine(self.cas)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bipartite_matching_identity_and_permuted(self):
        # Create cost matrix where row i matches column (i+2)%5
        N = 10
        target_perm = np.roll(np.arange(N), 2)
        cost_matrix = np.ones((N, N))
        for i in range(N):
            cost_matrix[i, target_perm[i]] = 0.0

        recovered_perm = solve_bipartite_matching(cost_matrix)
        np.testing.assert_array_equal(recovered_perm, target_perm)

    def test_out_of_core_mlp_alignment_and_compression(self):
        manifest, stats = self.engine.align_and_diff(
            base_checkpoint_path=self.fixtures["mlp_base"],
            target_checkpoint_path=self.fixtures["mlp_target"],
            config_path=self.fixtures["config"],
        )

        self.assertTrue(stats["is_alignable"])
        self.assertGreater(stats["mean_confidence"], 0.7)
        # Residual delta should achieve significant compression savings
        self.assertLess(stats["residual_ratio"], 0.5)
        self.assertGreater(stats["num_permutation_groups"], 0)

    def test_out_of_core_cnn_alignment(self):
        manifest, stats = self.engine.align_and_diff(
            base_checkpoint_path=self.fixtures["cnn_base"],
            target_checkpoint_path=self.fixtures["cnn_target"],
            config_path=self.fixtures["config"],
        )

        self.assertTrue(stats["is_alignable"])
        self.assertGreater(stats["mean_confidence"], 0.7)
        self.assertLess(stats["residual_ratio"], 0.5)



if __name__ == "__main__":
    unittest.main()
