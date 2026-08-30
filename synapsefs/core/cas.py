"""
Content-Addressable Storage (CAS) with Atomic Writes and Tamper Detection.
"""

import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple, Union

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


def compute_hash(data: Union[bytes, bytearray, memoryview]) -> str:
    """Computes cryptographic hash of byte buffer using SHA-256 (or BLAKE3 if available)."""
    if HAS_BLAKE3:
        return blake3.blake3(data).hexdigest()
    return hashlib.sha256(data).hexdigest()


class ContentAddressableStore:
    """
    Content-Addressable Storage engine.
    Stores immutable objects by their content hash in a sharded 2-character hex directory structure:
    .synapsefs/objects/ab/cdef123456...
    """

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.objects_dir = self.root_dir / "objects"
        self.tmp_dir = self.root_dir / "tmp"
        self.wal_dir = self.root_dir / "wal"
        
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(parents=True, exist_ok=True)

    def _get_object_path(self, object_hash: str) -> Path:
        """Returns the canonical filesystem path for a given object hash."""
        prefix = object_hash[:2]
        rest = object_hash[2:]
        return self.objects_dir / prefix / rest

    def exists(self, object_hash: str) -> bool:
        """Checks if an object with the specified hash exists in the CAS."""
        return self._get_object_path(object_hash).is_file()

    def put(self, data: Union[bytes, bytearray, memoryview]) -> str:
        """
        Atomically stores data in the CAS.
        Returns the content hash.
        Guarantees crash-consistency via temp file write, fsync, and atomic rename.
        """
        object_hash = compute_hash(data)
        dest_path = self._get_object_path(object_hash)

        if dest_path.is_file():
            # Object already exists and is immutable
            return object_hash

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first in tmp_dir
        temp_file = self.tmp_dir / f"blob_{uuid.uuid4().hex}.tmp"
        try:
            with open(temp_file, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename / replace
            os.replace(temp_file, dest_path)
        except Exception:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise

        return object_hash

    def get(self, object_hash: str) -> bytes:
        """
        Retrieves data bytes for an object hash.
        Raises FileNotFoundError if object does not exist.
        Raises ValueError if object is corrupted / tampered.
        """
        obj_path = self._get_object_path(object_hash)
        if not obj_path.is_file():
            raise FileNotFoundError(f"Object {object_hash} not found in CAS")

        with open(obj_path, "rb") as f:
            data = f.read()

        actual_hash = compute_hash(data)
        if actual_hash != object_hash:
            raise ValueError(f"Integrity check failed for {object_hash}: actual hash is {actual_hash}")

        return data

    def get_stream(self, object_hash: str) -> Path:
        """Returns the file path for direct streaming/mmap access without reading into RAM."""
        obj_path = self._get_object_path(object_hash)
        if not obj_path.is_file():
            raise FileNotFoundError(f"Object {object_hash} not found in CAS")
        return obj_path

    def verify_object(self, object_hash: str) -> bool:
        """Verifies integrity of a single stored object against its content hash."""
        obj_path = self._get_object_path(object_hash)
        if not obj_path.is_file():
            return False
        try:
            with open(obj_path, "rb") as f:
                data = f.read()
            return compute_hash(data) == object_hash
        except Exception:
            return False

    def list_all_objects(self) -> Set[str]:
        """Returns a set of all object hashes present in the CAS."""
        hashes: Set[str] = set()
        for prefix_dir in self.objects_dir.iterdir():
            if prefix_dir.is_dir() and len(prefix_dir.name) == 2:
                for obj_file in prefix_dir.iterdir():
                    if obj_file.is_file():
                        hashes.add(prefix_dir.name + obj_file.name)
        return hashes
