"""
Synthetic Model Checkpoint Generator for SynapseFS Testing & Benchmarking.
Generates MLP and CNN/ResNet architectures with known neuron permutations and fine-tuning perturbations.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from synapsefs.alignment.residual import permute_tensor
from synapsefs.utils.safetensors_helper import save_safetensors_file


def generate_synthetic_mlp(
    d_in: int = 128,
    d_hidden1: int = 256,
    d_hidden2: int = 256,
    d_out: int = 64,
    dtype: str = "float16",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generates base synthetic MLP weights in fp16 precision."""
    np_dtype = np.dtype(dtype)
    rng = np.random.RandomState(seed)

    tensors = {
        "fc1.weight": rng.randn(d_hidden1, d_in).astype(np_dtype),
        "fc1.bias": rng.randn(d_hidden1).astype(np_dtype),
        "fc2.weight": rng.randn(d_hidden2, d_hidden1).astype(np_dtype),
        "fc2.bias": rng.randn(d_hidden2).astype(np_dtype),
        "fc3.weight": rng.randn(d_out, d_hidden2).astype(np_dtype),
        "fc3.bias": rng.randn(d_out).astype(np_dtype),
    }
    return tensors


def apply_permutation_and_noise_to_mlp(
    base_tensors: Dict[str, np.ndarray],
    noise_std: float = 0.0005,
    seed: int = 99,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Applies known neuron permutations to hidden layers 1 and 2, plus small fine-tuning float perturbations.
    Returns:
        target_tensors: The mathematically equivalent (permuted) + fine-tuned model
        known_perms: The ground-truth permutation arrays
    """
    rng = np.random.RandomState(seed)
    d_h1 = base_tensors["fc1.weight"].shape[0]
    d_h2 = base_tensors["fc2.weight"].shape[0]

    perm_h1 = rng.permutation(d_h1)
    perm_h2 = rng.permutation(d_h2)

    target_tensors = {}

    # fc1: row permuted by perm_h1
    w1 = base_tensors["fc1.weight"][perm_h1, :]
    b1 = base_tensors["fc1.bias"][perm_h1]
    target_tensors["fc1.weight"] = (w1 + rng.randn(*w1.shape).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["fc1.bias"] = (b1 + rng.randn(*b1.shape).astype(np.float16) * noise_std).astype(np.float16)

    # fc2: row permuted by perm_h2, col permuted by perm_h1
    w2 = base_tensors["fc2.weight"][perm_h2, :][:, perm_h1]
    b2 = base_tensors["fc2.bias"][perm_h2]
    target_tensors["fc2.weight"] = (w2 + rng.randn(*w2.shape).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["fc2.bias"] = (b2 + rng.randn(*b2.shape).astype(np.float16) * noise_std).astype(np.float16)

    # fc3: col permuted by perm_h2
    w3 = base_tensors["fc3.weight"][:, perm_h2]
    b3 = base_tensors["fc3.bias"]
    target_tensors["fc3.weight"] = (w3 + rng.randn(*w3.shape).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["fc3.bias"] = (b3 + rng.randn(*b3.shape).astype(np.float16) * noise_std).astype(np.float16)

    known_perms = {
        "perm_h1": perm_h1,
        "perm_h2": perm_h2,
    }

    return target_tensors, known_perms


def generate_synthetic_cnn(
    in_channels: int = 3,
    c1: int = 32,
    c2: int = 64,
    num_classes: int = 10,
    dtype: str = "float16",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generates base synthetic CNN with Conv2D, BatchNorm2D, and Linear in fp16."""
    np_dtype = np.dtype(dtype)
    rng = np.random.RandomState(seed)

    tensors = {
        "conv1.weight": rng.randn(c1, in_channels, 3, 3).astype(np_dtype),
        "bn1.weight": rng.randn(c1).astype(np_dtype),
        "bn1.bias": rng.randn(c1).astype(np_dtype),
        "bn1.running_mean": rng.randn(c1).astype(np_dtype),
        "bn1.running_var": np.abs(rng.randn(c1)).astype(np_dtype),
        "conv2.weight": rng.randn(c2, c1, 3, 3).astype(np_dtype),
        "fc.weight": rng.randn(num_classes, c2).astype(np_dtype),
        "fc.bias": rng.randn(num_classes).astype(np_dtype),
    }
    return tensors


def apply_permutation_and_noise_to_cnn(
    base_tensors: Dict[str, np.ndarray],
    noise_std: float = 0.0005,
    seed: int = 100,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Applies filter permutation to Conv1 and associated BatchNorm, plus Conv2 input channels."""
    rng = np.random.RandomState(seed)
    c1 = base_tensors["conv1.weight"].shape[0]
    perm_c1 = rng.permutation(c1)

    target_tensors = {}
    target_tensors["conv1.weight"] = (base_tensors["conv1.weight"][perm_c1] + rng.randn(*base_tensors["conv1.weight"].shape).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["bn1.weight"] = (base_tensors["bn1.weight"][perm_c1] + rng.randn(c1).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["bn1.bias"] = (base_tensors["bn1.bias"][perm_c1] + rng.randn(c1).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["bn1.running_mean"] = (base_tensors["bn1.running_mean"][perm_c1] + rng.randn(c1).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["bn1.running_var"] = base_tensors["bn1.running_var"][perm_c1].astype(np.float16)

    # conv2: permute input channels (axis 1)
    target_tensors["conv2.weight"] = (base_tensors["conv2.weight"][:, perm_c1] + rng.randn(*base_tensors["conv2.weight"].shape).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["fc.weight"] = (base_tensors["fc.weight"] + rng.randn(*base_tensors["fc.weight"].shape).astype(np.float16) * noise_std).astype(np.float16)
    target_tensors["fc.bias"] = base_tensors["fc.bias"].copy()

    return target_tensors, {"perm_c1": perm_c1}



def generate_fixture_files(output_dir: Path) -> Dict[str, Path]:
    """Generates sample .safetensors files and config.json in output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. MLP Fixtures
    base_mlp = generate_synthetic_mlp()
    tgt_mlp, _ = apply_permutation_and_noise_to_mlp(base_mlp)

    mlp_base_path = output_dir / "mlp_base.safetensors"
    mlp_target_path = output_dir / "mlp_target.safetensors"
    save_safetensors_file(base_mlp, str(mlp_base_path), metadata={"model_type": "mlp", "version": "base"})
    save_safetensors_file(tgt_mlp, str(mlp_target_path), metadata={"model_type": "mlp", "version": "fine-tuned"})

    # 2. CNN Fixtures
    base_cnn = generate_synthetic_cnn()
    tgt_cnn, _ = apply_permutation_and_noise_to_cnn(base_cnn)

    cnn_base_path = output_dir / "cnn_base.safetensors"
    cnn_target_path = output_dir / "cnn_target.safetensors"
    save_safetensors_file(base_cnn, str(cnn_base_path), metadata={"model_type": "cnn", "version": "base"})
    save_safetensors_file(tgt_cnn, str(cnn_target_path), metadata={"model_type": "cnn", "version": "fine-tuned"})

    # 3. Config JSON
    config_data = {
        "architectures": ["MLP", "CNN"],
        "d_in": 128,
        "d_hidden": 256,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return {
        "mlp_base": mlp_base_path,
        "mlp_target": mlp_target_path,
        "cnn_base": cnn_base_path,
        "cnn_target": cnn_target_path,
        "config": config_path,
    }
