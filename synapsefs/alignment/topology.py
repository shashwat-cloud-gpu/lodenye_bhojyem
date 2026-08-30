"""
Topology Parser & Computational Graph Dependency Builder.
Analyzes model architecture and tensor relationships for MLP and CNN/ResNet models.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class PermutationGroup:
    """
    Represents a group of tensors that share a common hidden dimension / permutation.
    For example:
    - Layer l output weights (rows)
    - Layer l bias
    - Layer l BatchNorm (weight, bias, running_mean, running_var)
    - Layer l+1 input weights (columns)
    """

    def __init__(self, group_id: str, dim_size: int):
        self.group_id = group_id
        self.dim_size = dim_size
        
        # Primary output tensor whose rows/filters define the neuron order: (tensor_name, axis)
        self.output_tensors: List[Tuple[str, int]] = []
        
        # 1D Parameter vectors that permute along axis 0 (bias, norm weights, running stats)
        self.vector_tensors: List[str] = []
        
        # Subsequent input tensors whose columns/input channels permute along specified axis: (tensor_name, axis)
        self.input_tensors: List[Tuple[str, int]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "dim_size": self.dim_size,
            "output_tensors": self.output_tensors,
            "vector_tensors": self.vector_tensors,
            "input_tensors": self.input_tensors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PermutationGroup":
        grp = cls(d["group_id"], d["dim_size"])
        grp.output_tensors = [tuple(x) for x in d.get("output_tensors", [])]
        grp.vector_tensors = d.get("vector_tensors", [])
        grp.input_tensors = [tuple(x) for x in d.get("input_tensors", [])]
        return grp


class ModelTopology:
    """
    Parses config.json and tensor shapes to build the permutation dependency graph.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config_dict or {}
        self.groups: List[PermutationGroup] = []

    @classmethod
    def from_config_file(cls, config_path: Path) -> "ModelTopology":
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(data)
        return cls({})

    def build_from_tensors(self, tensor_shapes: Dict[str, List[int]]) -> List[PermutationGroup]:
        """
        Infers permutation groups across layers based on config.json or standard naming patterns.
        Supports:
        - MLPs (e.g. fc1.weight, fc1.bias, fc2.weight, ...)
        - CNNs / ResNets (e.g. conv1.weight, bn1.weight, bn1.bias, bn1.running_mean, conv2.weight, ...)
        """
        groups: List[PermutationGroup] = []

        # Check if explicit permutation groups are defined in config
        if "permutation_groups" in self.config:
            for g_def in self.config["permutation_groups"]:
                groups.append(PermutationGroup.from_dict(g_def))
            self.groups = groups
            return groups

        # Automatic graph inference based on layer sequence
        # Separate tensors by layer indices
        # Detect linear layers: name.weight with 2D shape [out_features, in_features]
        # Detect conv layers: name.weight with 4D shape [out_channels, in_channels, k_h, k_w]
        
        weight_tensors = sorted([k for k in tensor_shapes if k.endswith(".weight") and len(tensor_shapes[k]) in (2, 4)])

        for i in range(len(weight_tensors) - 1):
            curr_w = weight_tensors[i]
            next_w = weight_tensors[i + 1]

            curr_shape = tensor_shapes[curr_w]
            next_shape = tensor_shapes[next_w]

            curr_out_dim = curr_shape[0]  # out_features or out_channels
            next_in_dim = next_shape[1]   # in_features or in_channels

            # If the output dimension of curr matches input dimension of next, they form a permutation group
            if curr_out_dim == next_in_dim:
                group_id = f"perm_group_{i}_{curr_w.replace('.weight', '')}"
                grp = PermutationGroup(group_id, curr_out_dim)
                grp.output_tensors.append((curr_w, 0))  # Output rows/filters
                grp.input_tensors.append((next_w, 1))   # Next input cols/channels

                # Check for associated bias and norm parameters for curr_w
                prefix = curr_w.rsplit(".", 1)[0]
                
                # Bias
                bias_name = f"{prefix}.bias"
                if bias_name in tensor_shapes and tensor_shapes[bias_name] == [curr_out_dim]:
                    grp.vector_tensors.append(bias_name)

                # Check for adjacent BatchNorm / LayerNorm
                # Standard patterns: bn1, norm1, layer_norm
                for norm_key in [f"{prefix}_bn", f"{prefix}.bn", f"{prefix}_norm"]:
                    for suffix in [".weight", ".bias", ".running_mean", ".running_var"]:
                        norm_tensor = norm_key + suffix
                        if norm_tensor in tensor_shapes and tensor_shapes[norm_tensor] == [curr_out_dim]:
                            grp.vector_tensors.append(norm_tensor)

                # Also search for any 1D tensors matching curr_out_dim in the immediate neighborhood
                layer_base = prefix.split(".")[0]
                for tname, shape in tensor_shapes.items():
                    if tname.startswith(layer_base) and shape == [curr_out_dim]:
                        if tname not in grp.vector_tensors and tname != bias_name and not tname.endswith(".weight"):
                            grp.vector_tensors.append(tname)

                groups.append(grp)

        self.groups = groups
        return groups
