"""
Linux POSIX FUSE Filesystem Daemon for SynapseFS.
Exposes dynamically reconstructed checkpoints via read()/mmap() with zero disk pre-materialization.
"""

import errno
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from synapsefs.alignment.reconstructor import CheckpointReconstructor
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.vfs.byte_mapper import VirtualSafetensorsMapper
from synapsefs.vfs.lru_cache import LRUTensorCache


def run_fuse_mount(
    dag: RepositoryDAG,
    commit_id: str,
    mount_point: Path,
    foreground: bool = True,
    max_cache_mb: int = 512,
) -> None:
    """
    Mounts the SynapseFS virtual filesystem at mount_point.
    """
    mount_point = Path(mount_point).resolve()
    if not mount_point.exists():
        mount_point.mkdir(parents=True, exist_ok=True)

    commit = dag.get_commit(commit_id)
    manifest = dag.get_manifest(commit.manifest_hash)
    cache = LRUTensorCache(max_bytes=max_cache_mb * 1024 * 1024)
    mapper = VirtualSafetensorsMapper(dag=dag, manifest=manifest, cache=cache)

    try:
        from fuse import FUSE, FuseOSError, Operations
    except ImportError:
        print("[ERROR] 'fusepy' is not installed or libfuse is missing.", file=sys.stderr)
        print("Install via: pip install fusepy and ensure libfuse3/libfuse2 is installed on Linux.", file=sys.stderr)
        sys.exit(1)

    class SynapseFSOperations(Operations):
        def __init__(self):
            self.model_filename = "model.safetensors"
            self.manifest_filename = "manifest.json"
            self.commit_filename = "commit_info.json"

            self.manifest_bytes = manifest.serialize()
            self.commit_bytes = commit.serialize()
            self.model_size = mapper.total_file_size
            self.now = time.time()

        def getattr(self, path: str, fh=None) -> Dict[str, Any]:
            if path == "/":
                return {
                    "st_mode": stat.S_IFDIR | 0o555,
                    "st_nlink": 2,
                    "st_size": 4096,
                    "st_ctime": self.now,
                    "st_mtime": self.now,
                    "st_atime": self.now,
                }
            elif path == f"/{self.model_filename}":
                return {
                    "st_mode": stat.S_IFREG | 0o444,
                    "st_nlink": 1,
                    "st_size": self.model_size,
                    "st_ctime": self.now,
                    "st_mtime": self.now,
                    "st_atime": self.now,
                }
            elif path == f"/{self.manifest_filename}":
                return {
                    "st_mode": stat.S_IFREG | 0o444,
                    "st_nlink": 1,
                    "st_size": len(self.manifest_bytes),
                    "st_ctime": self.now,
                    "st_mtime": self.now,
                    "st_atime": self.now,
                }
            elif path == f"/{self.commit_filename}":
                return {
                    "st_mode": stat.S_IFREG | 0o444,
                    "st_nlink": 1,
                    "st_size": len(self.commit_bytes),
                    "st_ctime": self.now,
                    "st_mtime": self.now,
                    "st_atime": self.now,
                }
            else:
                raise FuseOSError(errno.ENOENT)

        def readdir(self, path: str, fh):
            if path == "/":
                return [".", "..", self.model_filename, self.manifest_filename, self.commit_filename]
            raise FuseOSError(errno.ENOENT)

        def open(self, path: str, flags: int):
            # Enforce read-only
            if (flags & os.O_WRONLY) or (flags & os.O_RDWR):
                raise FuseOSError(errno.EROFS)
            return 0

        def read(self, path: str, size: int, offset: int, fh) -> bytes:
            if path == f"/{self.model_filename}":
                # Serve on-the-fly from mapper
                return mapper.read(offset, size)
            elif path == f"/{self.manifest_filename}":
                return self.manifest_bytes[offset : offset + size]
            elif path == f"/{self.commit_filename}":
                return self.commit_bytes[offset : offset + size]
            else:
                raise FuseOSError(errno.ENOENT)

        # Disallow write operations
        def write(self, path, data, offset, fh):
            raise FuseOSError(errno.EROFS)

        def create(self, path, mode, fi=None):
            raise FuseOSError(errno.EROFS)

        def unlink(self, path):
            raise FuseOSError(errno.EROFS)

        def rmdir(self, path):
            raise FuseOSError(errno.EROFS)

    print(f"[*] Mounting SynapseFS commit {commit_id[:10]} at {mount_point}")
    print(f"[*] Virtual file available: {mount_point / 'model.safetensors'} (Size: {mapper.total_file_size / (1024*1024):.2f} MB)")
    print(f"[*] Strict No Pre-Materialization enforced: Serving dynamically via RAM/CAS on demand.")
    FUSE(SynapseFSOperations(), str(mount_point), nothreads=False, foreground=foreground, ro=True)
