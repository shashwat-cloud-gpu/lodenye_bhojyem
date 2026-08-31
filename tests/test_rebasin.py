"""
Unit Tests for Git Re-Basin (Algorithm 1 Coordinate Ascent & Model Interpolation).
"""

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from synapsefs.alignment.rebasin import GitReBasinEngine
from synapsefs.alignment.topology import ModelTopology
from tests.fixtures.generate_models import (
    apply_permutation_and_noise_to_cnn,
    apply_permutation_and_noise_to_mlp,
    generate_synthetic_cnn,
    generate_synthetic_mlp,
)


class TestGitReBasin(unittest.TestCase):

    def test_mlp_coordinate_ascent_rebasin(self):
        base_mlp = generate_synthetic_mlp(d_in=64, d_hidden1=128, d_hidden2=128, d_out=32, dtype="float32", seed=42)
        target_mlp, known_perms = apply_permutation_and_noise_to_mlp(base_mlp, noise_std=0.0, seed=123)

        engine = GitReBasinEngine(max_iter=10)
        rebasined_b, discovered_perms = engine.rebasin(base_mlp, target_mlp)

        # Re-basined B should now be identical to base_mlp within fp16 precision
        for tname in base_mlp.keys():
            np.testing.assert_allclose(
                rebasined_b[tname].astype(np.float32),
                base_mlp[tname].astype(np.float32),
                atol=2e-3,
                err_msg=f"Tensor {tname} was not aligned correctly after Git Re-Basin"
            )

    def test_cnn_coordinate_ascent_rebasin(self):
        base_cnn = generate_synthetic_cnn(in_channels=3, c1=16, c2=32, num_classes=10, dtype="float16", seed=42)
        target_cnn, known_perms = apply_permutation_and_noise_to_cnn(base_cnn, noise_std=0.0, seed=555)

        engine = GitReBasinEngine(max_iter=10)
        rebasined_b, discovered_perms = engine.rebasin(base_cnn, target_cnn)

        for tname in base_cnn.keys():
            np.testing.assert_allclose(
                rebasined_b[tname].astype(np.float32),
                base_cnn[tname].astype(np.float32),
                atol=2e-3,
                err_msg=f"CNN Tensor {tname} was not aligned correctly after Git Re-Basin"
            )

    def test_linear_mode_interpolation(self):
        base_mlp = generate_synthetic_mlp(d_in=32, d_hidden1=64, d_hidden2=64, d_out=16, dtype="float16", seed=10)
        target_mlp, _ = apply_permutation_and_noise_to_mlp(base_mlp, noise_std=0.0001, seed=20)

        engine = GitReBasinEngine(max_iter=10)
        rebasined_b, _ = engine.rebasin(base_mlp, target_mlp)

        # 50/50 Model Blend
        interpolated = engine.interpolate(base_mlp, rebasined_b, alpha=0.5)

        for tname in base_mlp.keys():
            expected = 0.5 * base_mlp[tname].astype(np.float32) + 0.5 * rebasined_b[tname].astype(np.float32)
            np.testing.assert_allclose(interpolated[tname].astype(np.float32), expected, atol=2e-3)




if __name__ == "__main__":
    unittest.main()
