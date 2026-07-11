from __future__ import annotations

import hashlib
from pathlib import Path


def stable_int_hash(value: str, modulo: int = 2**31 - 1) -> int:
    """Return a deterministic non-negative integer hash across Python processes."""
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % modulo


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a stable SHA-256 digest without loading the whole file into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
