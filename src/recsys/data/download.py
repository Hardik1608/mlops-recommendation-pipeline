"""Download + verify + extract a MovieLens archive into ``data/raw/<dataset>/``."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

from recsys.data.datasets import get_spec

logger = logging.getLogger(__name__)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(name: str, raw_dir: Path, *, force: bool = False, timeout: int = 60) -> Path:
    """Return the extracted dataset directory, downloading it if needed.

    The archive checksum is verified for variants whose SHA-256 is pinned in ``datasets.py``.
    """
    spec = get_spec(name)
    if spec.url is None:
        raise ValueError(f"{name!r} is generated locally - use `recsys synth` instead")

    target_dir = raw_dir / name
    if (target_dir / spec.ratings_file).exists() and not force:
        logger.info("Dataset %s already present at %s", name, target_dir)
        return target_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{name}.zip"
    logger.info("Downloading %s -> %s", spec.url, archive)
    tmp = archive.with_suffix(".zip.part")
    with urllib.request.urlopen(spec.url, timeout=timeout) as resp, tmp.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)
    tmp.replace(archive)

    digest = sha256_file(archive)
    if spec.sha256 and digest != spec.sha256:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {name}: expected {spec.sha256}, got {digest}")
    if not spec.sha256:
        logger.warning("No pinned checksum for %s (sha256=%s)", name, digest)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)  # archives contain a top-level <name>/ folder
    if not (target_dir / spec.ratings_file).exists():
        raise RuntimeError(f"Archive for {name} did not contain {spec.ratings_file}")
    logger.info("Extracted %s to %s", name, target_dir)
    return target_dir
