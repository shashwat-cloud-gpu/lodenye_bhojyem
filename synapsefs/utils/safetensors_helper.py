"""
Safetensors format parser, serializer, and out-of-core memory-mapped reader.
Fully compliant with Hugging Face Safetensors specification.
"""

import json
import mmap
import os
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Safetensors dtype mapping to numpy dtype string
DTYPE_TO_NUMPY = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U64": "uint64",
    "U32": "uint32",
    "U16": "uint16",
    "U8": "uint8",
    "BOOL": "bool",
}

NUMPY_TO_DTYPE = {v: k for k, v in DTYPE_TO_NUMPY.items()}
# Additional aliases
NUMPY_TO_DTYPE["float64"] = "F64"
NUMPY_TO_DTYPE["float32"] = "F32"
NUMPY_TO_DTYPE["float16"] = "F16"
NUMPY_TO_DTYPE["int64"] = "I64"
NUMPY_TO_DTYPE["int32"] = "I32"
NUMPY_TO_DTYPE["int16"] = "I16"
NUMPY_TO_DTYPE["int8"] = "I8"
NUMPY_TO_DTYPE["uint64"] = "U64"
NUMPY_TO_DTYPE["uint32"] = "U32"
NUMPY_TO_DTYPE["uint16"] = "U16"
NUMPY_TO_DTYPE["uint8"] = "U8"
NUMPY_TO_DTYPE["bool"] = "BOOL"

DTYPE_BYTE_SIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


def read_safetensors_header(filepath_or_data: Union[str, bytes]) -> Tuple[int, Dict[str, Any], int]:
    """
    Reads the header of a .safetensors file without loading tensor buffers.
    
    Returns:
        header_size: int (N bytes of JSON)
        header: dict (parsed JSON metadata & tensor descriptors)
        data_start_offset: int (8 + header_size)
    """
    if isinstance(filepath_or_data, (str, os.PathLike)):
        with open(filepath_or_data, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                raise ValueError("File too small to be a valid safetensors file")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_json_bytes = f.read(header_size)
            header = json.loads(header_json_bytes.decode("utf-8"))
            return header_size, header, 8 + header_size
    elif isinstance(filepath_or_data, bytes):
        if len(filepath_or_data) < 8:
            raise ValueError("Buffer too small to be a valid safetensors buffer")
        header_size = struct.unpack("<Q", filepath_or_data[:8])[0]
        header_json_bytes = filepath_or_data[8 : 8 + header_size]
        header = json.loads(header_json_bytes.decode("utf-8"))
        return header_size, header, 8 + header_size
    else:
        raise TypeError("Expected filepath or bytes")


def load_tensor_lazy(filepath: str, tensor_name: str, header_info: Optional[Tuple[int, Dict[str, Any], int]] = None) -> np.ndarray:
    """
    Memory-mapped out-of-core loader for a single tensor from a safetensors file.
    Does NOT load the rest of the model into RAM.
    """
    if header_info is None:
        header_size, header, data_start = read_safetensors_header(filepath)
    else:
        header_size, header, data_start = header_info

    if tensor_name not in header:
        raise KeyError(f"Tensor '{tensor_name}' not found in safetensors header")

    meta = header[tensor_name]
    dtype_str = meta["dtype"]
    shape = meta["shape"]
    offsets = meta["data_offsets"]

    start = data_start + offsets[0]
    end = data_start + offsets[1]
    byte_count = end - start

    with open(filepath, "rb") as f:
        # Using memory mapping
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            raw_bytes = mm[start:end]
            
    np_dtype = DTYPE_TO_NUMPY.get(dtype_str, "float32")
    if dtype_str == "BF16":
        # Handle bfloat16 representation via uint16 view
        arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape)
    else:
        arr = np.frombuffer(raw_bytes, dtype=np.dtype(np_dtype)).reshape(shape)
    return arr.copy()


def save_safetensors_file(
    tensors: Dict[str, np.ndarray],
    filepath: str,
    metadata: Optional[Dict[str, str]] = None
) -> None:
    """
    Serializes a dictionary of numpy arrays to a .safetensors file.
    Preserves exact standard layout.
    """
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    current_offset = 0
    tensor_bytes_list: List[bytes] = []

    # Sort keys for deterministic reproducible serialization
    tensor_names = sorted(tensors.keys())
    for name in tensor_names:
        arr = tensors[name]
        raw_bytes = arr.tobytes()
        byte_len = len(raw_bytes)
        
        dtype_str = NUMPY_TO_DTYPE.get(str(arr.dtype), "F32")
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + byte_len],
        }
        current_offset += byte_len
        tensor_bytes_list.append(raw_bytes)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)
    header_len_bytes = struct.pack("<Q", header_len)

    with open(filepath, "wb") as f:
        f.write(header_len_bytes)
        f.write(header_json)
        for chunk in tensor_bytes_list:
            f.write(chunk)


def serialize_safetensors_to_bytes(
    tensors: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Serializes dictionary of tensors to safetensors byte buffer in memory.
    """
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    current_offset = 0
    tensor_bytes_list: List[bytes] = []

    tensor_names = sorted(tensors.keys())
    for name in tensor_names:
        arr = tensors[name]
        raw_bytes = arr.tobytes()
        byte_len = len(raw_bytes)
        dtype_str = NUMPY_TO_DTYPE.get(str(arr.dtype), "F32")
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + byte_len],
        }
        current_offset += byte_len
        tensor_bytes_list.append(raw_bytes)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)
    header_len_bytes = struct.pack("<Q", header_len)

    return header_len_bytes + header_json + b"".join(tensor_bytes_list)
