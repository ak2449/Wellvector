"""
PDF text/table extraction with scanned-document detection.

Strategy:
1. Try pdfplumber for text + tables (works well for digital PDFs).
2. If text yield is low → flag as scanned and prepare page images for
   vision-model extraction.
3. Score each page for casing-relevance keywords to select only the
   pages worth sending to the LLM.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from config import CASING_PAGE_KEYWORDS, MAX_PAGES_PER_LLM_CALL, MAX_VISION_PAGES

logger = logging.getLogger(__name__)

# If average chars per page < this, treat as scanned
SCANNED_THRESHOLD = 80


@dataclass
class PageContent:
    """Extracted content for a single PDF page."""
    page_num: int                   # 0-indexed
    text: str = ""
    tables: list[list[list[str]]] = field(default_factory=list)
    relevance_score: float = 0.0
    is_scanned: bool = False

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip()) or bool(self.tables)

    @property
    def combined_text(self) -> str:
        """Text + serialised tables for keyword matching / LLM input."""
        parts = [self.text]
        for table in self.tables:
            for row in table:
                parts.append(" | ".join(cell or "" for cell in row))
        return "\n".join(parts)


@dataclass
class DocumentContent:
    """Full extraction result for one PDF."""
    path: Path
    pages: list[PageContent]
    total_pages: int
    is_scanned: bool = False

    @property
    def relevant_pages(self) -> list[PageContent]:
        """Pages with relevance_score > 0, sorted descending."""
        return sorted(
            [p for p in self.pages if p.relevance_score > 0],
            key=lambda p: -p.relevance_score,
        )

    def top_pages_text(self, max_pages: int = MAX_PAGES_PER_LLM_CALL) -> str:
        """Concatenated text of the most relevant pages for LLM input."""
        selected = self.relevant_pages[:max_pages]
        if not selected:
            # Fallback: use first few pages (often contain summary data)
            selected = [p for p in self.pages if p.has_content][:max_pages]
        parts = []
        for p in selected:
            parts.append(f"--- PAGE {p.page_num + 1} ---")
            parts.append(p.combined_text)
        return "\n\n".join(parts)


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> DocumentContent:
    """
    Extract text and tables from a PDF using pdfplumber.
    Detects scanned documents and scores pages for relevance.
    """
    pdf_path = Path(pdf_path)
    pages: list[PageContent] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            total_chars = 0

            for i, page in enumerate(pdf.pages):
                pc = PageContent(page_num=i)

                # Text
                try:
                    raw_text = page.extract_text() or ""
                    pc.text = raw_text
                    total_chars += len(raw_text)
                except Exception as e:
                    logger.debug("Text extraction failed on page %d: %s", i, e)

                # Tables
                try:
                    raw_tables = page.extract_tables()
                    if raw_tables:
                        pc.tables = raw_tables
                except Exception as e:
                    logger.debug("Table extraction failed on page %d: %s", i, e)

                pages.append(pc)

        # Detect scanned
        avg_chars = total_chars / max(total_pages, 1)
        is_scanned = avg_chars < SCANNED_THRESHOLD

        if is_scanned:
            logger.info(
                "%s appears scanned (avg %.0f chars/page). Will use vision model.",
                pdf_path.name, avg_chars,
            )
        else:
            logger.info(
                "%s: extracted %d pages, avg %.0f chars/page",
                pdf_path.name, total_pages, avg_chars,
            )

        # Score relevance
        for pc in pages:
            pc.relevance_score = _score_relevance(pc.combined_text)
            pc.is_scanned = is_scanned

        doc = DocumentContent(
            path=pdf_path,
            pages=pages,
            total_pages=total_pages,
            is_scanned=is_scanned,
        )
        return doc

    except Exception as e:
        logger.error("Failed to open PDF %s: %s", pdf_path, e)
        raise


def _score_relevance(text: str) -> float:
    """Score a page's text for casing-data relevance (0–1 scale)."""
    if not text.strip():
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in CASING_PAGE_KEYWORDS if kw.lower() in text_lower)
    # Normalise: 5+ keyword hits = max score
    return min(hits / 5.0, 1.0)


# ── Image extraction (for scanned PDFs) ─────────────────────────────────────

def extract_page_images(
    pdf_path: Path,
    page_indices: list[int] | None = None,
    dpi: int = 200,
    max_pages: int = MAX_VISION_PAGES,
) -> list[tuple[int, bytes]]:
    """
    Render PDF pages as PNG images for vision-model input.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF.
    page_indices : list[int] | None
        Specific pages to render (0-indexed). If None, renders first `max_pages`.
    dpi : int
        Resolution for rendering.
    max_pages : int
        Maximum number of pages to render.

    Returns
    -------
    list[tuple[int, bytes]]
        List of (page_index, png_bytes) tuples.
    """
    doc = fitz.open(str(pdf_path))
    total = len(doc)

    if page_indices is None:
        page_indices = list(range(min(total, max_pages)))
    else:
        page_indices = page_indices[:max_pages]

    results = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for idx in page_indices:
        if idx >= total:
            continue
        page = doc[idx]
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        results.append((idx, buf.getvalue()))
        logger.debug("Rendered page %d of %s (%dx%d)", idx, pdf_path.name, pix.width, pix.height)

    doc.close()
    return results


def find_relevant_scanned_pages(
    pdf_path: Path,
    sample_pages: int = 10,
    dpi: int = 72,
) -> list[int]:
    """
    For scanned PDFs: render low-res thumbnails of the first N pages and
    use a quick heuristic (page position) to guess which pages matter.

    In practice, casing data is usually in the first 5-10 pages of well
    completion reports (the summary section).
    """
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    doc.close()

    # Heuristic: for well completion reports, casing summary is usually
    # in pages 1-8 (after cover page). For very long docs, also check
    # around page 15-20 where detailed casing programmes often sit.
    candidates = list(range(min(8, total)))
    if total > 15:
        candidates.extend(range(14, min(22, total)))

    return sorted(set(candidates))
