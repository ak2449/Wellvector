"""
PDF downloader with local caching.

Downloads PDFs from SODIR factpages and caches them locally to avoid
re-downloading on subsequent runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from tqdm import tqdm

from config import CACHE_DIR
from triage import WellDocument

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60  # seconds
CHUNK_SIZE = 8192


def _cache_path(doc: WellDocument) -> Path:
    """Deterministic local path for a document."""
    # Use wellbore + filename to avoid collisions
    safe_wellbore = doc.wellbore.replace("/", "_")
    return CACHE_DIR / safe_wellbore / doc.filename


def download_pdf(doc: WellDocument, force: bool = False) -> Path:
    """
    Download a single PDF, caching locally.

    Parameters
    ----------
    doc : WellDocument
        The document to download.
    force : bool
        If True, re-download even if cached.

    Returns
    -------
    Path
        Local path to the downloaded PDF.
    """
    dest = _cache_path(doc)
    if dest.exists() and not force:
        logger.debug("Cache hit: %s", dest)
        doc.local_path = dest
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (%d KB) → %s", doc.url, doc.size_kb, dest)

    try:
        resp = requests.get(doc.url, timeout=DEFAULT_TIMEOUT, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        doc.local_path = dest
        return dest

    except requests.RequestException as e:
        logger.error("Download failed for %s: %s", doc.url, e)
        raise


def download_batch(
    docs: list[WellDocument],
    force: bool = False,
    show_progress: bool = True,
) -> list[WellDocument]:
    """
    Download all documents in a list, returning those that succeeded.
    """
    succeeded = []
    iterator = tqdm(docs, desc="Downloading PDFs") if show_progress else docs

    for doc in iterator:
        try:
            download_pdf(doc, force=force)
            succeeded.append(doc)
        except Exception:
            logger.warning("Skipping %s (download failed)", doc.doc_name)

    logger.info("Downloaded %d / %d documents", len(succeeded), len(docs))
    return succeeded


def download_single_url(url: str, dest_dir: Path | None = None) -> Path:
    """
    Download a single PDF by URL (for the live demo use case).

    Returns the local path.
    """
    dest_dir = dest_dir or CACHE_DIR / "live_demo"
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    dest = dest_dir / filename

    if dest.exists():
        logger.info("Already cached: %s", dest)
        return dest

    logger.info("Downloading %s", url)
    resp = requests.get(url, timeout=DEFAULT_TIMEOUT, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
    return dest
