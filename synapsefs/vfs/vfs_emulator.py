"""
Virtual File System (VFS) Emulator for Cross-Platform POSIX I/O Testing.
Enables full read()/mmap() compatibility testing on Windows, macOS, and Linux without native FUSE kernel drivers.
"""

import io
import os
from typing import Optional
from synapsefs.vfs.byte_mapper import VirtualSafetensorsMapper


class VirtualFileHandle(io.RawIOBase):
    """
    A POSIX-compliant seekable/readable stream wrapping VirtualSafetensorsMapper.
    """

    def __init__(self, mapper: VirtualSafetensorsMapper):
        self.mapper = mapper
        self.position = 0
        self._is_closed = False

    def readable(self) -> bool:
        return not self._is_closed

    def seekable(self) -> bool:
        return not self._is_closed

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._is_closed

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self.position = max(0, offset)
        elif whence == os.SEEK_CUR:
            self.position = max(0, self.position + offset)
        elif whence == os.SEEK_END:
            self.position = max(0, self.mapper.total_file_size + offset)
        else:
            raise ValueError(f"Invalid whence argument: {whence}")
        return self.position

    def readinto(self, b) -> int:
        size = len(b)
        data = self.mapper.read(self.position, size)
        n = len(data)
        b[:n] = data
        self.position += n
        return n

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.mapper.total_file_size - self.position
        data = self.mapper.read(self.position, size)
        self.position += len(data)
        return data

    def close(self) -> None:
        self._is_closed = True

