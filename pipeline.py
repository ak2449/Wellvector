"""
Main processing loop.

Workflow:
1. Triage: classify documents from CSV
2. Download: fetch priority PDFs
3. Extract: pull text/tables from PDFs
4. LLM: structured extraction via OpenAI
5. Standardise: normalise, validate, deduplicate
6. Output: produce final CSV/DataFrame
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from tenacity import RetryError

from config import OUTPUT_DIR
from downloader import download_batch, download_single_url
from extractor import (
    DocumentContent,
    extract_page_images,
    extract_text,
    find_relevant_scanned_pages,
)
from llm_client import (
    check_relevance,
    extract_casing_data,
    extract_casing_data_vision,
    tracker,
)
from standardiser import (
    deduplicate,
    standardise_record,
    to_dataframe,
    validate_record,
)
from triage import (
    WellDocument,
    filter_by_tier,
    get_wellbore_names,
    group_by_wellbore,
    load_and_triage,
)

logger = logging.getLogger(__name__)


def _format_exception(exc: BaseException) -> str:
    """
    Format exceptions for logging, unwrapping Tenacity RetryError so the
    underlying API error text is visible in the logs.
    """
    root: BaseException = exc
    if isinstance(exc, RetryError):
        last_exc = exc.last_attempt.exception()
        if last_exc is not None:
            root = last_exc

    parts = [f"{type(root).__name__}: {root}"]

    status_code = getattr(root, "status_code", None)
    if status_code is not None:
        parts.append(f"status={status_code}")

    body = getattr(root, "body", None)
    if body:
        parts.append(f"body={body}")

    return " | ".join(parts)


# ── Single document processing ───────────────────────────────────────────────

def process_document(doc: WellDocument) -> list[dict]:
    """
    Process a single document end-to-end: extract text → check relevance →
    LLM extraction → return raw records.
    """
    if doc.local_path is None:
        logger.warning("No local path for %s — skipping", doc.doc_name)
        return []

    logger.info("Processing: [%s] %s (Tier %d)", doc.wellbore, doc.doc_name, doc.tier)

    # Step 1: Extract text
    try:
        content: DocumentContent = extract_text(doc.local_path)
    except Exception as e:
        logger.error("Extraction failed for %s: %s", doc.doc_name, _format_exception(e))
        return []

    # Step 2: Handle scanned vs digital
    if content.is_scanned:
        return _process_scanned(doc, content)
    else:
        return _process_digital(doc, content)


def _process_digital(doc: WellDocument, content: DocumentContent) -> list[dict]:
    """Process a digital (text-extractable) PDF."""
    # Get relevant page text
    relevant_text = content.top_pages_text()

    if not relevant_text.strip():
        logger.info("No text extracted from %s — skipping", doc.doc_name)
        return []

    # For Tier 2 docs: do a quick relevance check first to save tokens
    if doc.tier == 2:
        sample = relevant_text[:2000]
        if not check_relevance(sample):
            logger.info("Tier 2 doc %s not relevant — skipping", doc.doc_name)
            return []

    # Extract casing data
    records = extract_casing_data(
        document_text=relevant_text,
        wellbore=doc.wellbore,
    )
    doc.extraction_result = records
    return records


def _process_scanned(doc: WellDocument, content: DocumentContent) -> list[dict]:
    """Process a scanned (image-based) PDF using vision model."""
    logger.info("Using vision model for scanned doc: %s", doc.doc_name)

    # Find candidate pages
    page_indices = find_relevant_scanned_pages(doc.local_path)

    # Render as images
    page_images = extract_page_images(
        doc.local_path,
        page_indices=page_indices,
    )

    if not page_images:
        logger.warning("No pages rendered for %s", doc.doc_name)
        return []

    # Vision extraction
    records = extract_casing_data_vision(
        page_images=page_images,
        wellbore=doc.wellbore,
    )
    doc.extraction_result = records
    return records


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(
    csv_path: str | Path,
    max_tier: int = 2,
    output_filename: str = "casing_data_cod_field.csv",
) -> pd.DataFrame:
    """
    Run the complete pipeline on a CSV dataset.

    Parameters
    ----------
    csv_path : str | Path
        Path to the wellbore_document_7_11.csv file.
    max_tier : int
        Maximum document tier to process (1 = only completion reports,
        2 = also supplementary docs).
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    pd.DataFrame
        The final standardised casing data table.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SODIR Casing Extraction Pipeline")
    logger.info("=" * 60)

    # ── 1. Triage ────────────────────────────────────────────────────────
    logger.info("Stage 1: Triage")
    all_docs = load_and_triage(csv_path)
    priority_docs = filter_by_tier(all_docs, max_tier=max_tier)
    wellbores = get_wellbore_names(all_docs)
    logger.info("Wellbores: %s", wellbores)
    logger.info("Processing %d / %d documents (tier ≤ %d)", len(priority_docs), len(all_docs), max_tier)

    # ── 2. Download ──────────────────────────────────────────────────────
    logger.info("Stage 2: Download")
    downloaded = download_batch(priority_docs)

    # ── 3 & 4. Extract + LLM ────────────────────────────────────────────
    logger.info("Stage 3-4: Extract & LLM")
    all_raw_records: list[dict] = []

    for doc in downloaded:
        try:
            records = process_document(doc)
            all_raw_records.extend(records)
        except Exception as e:
            logger.error("Failed processing %s: %s", doc.doc_name, _format_exception(e))

    logger.info("Raw extraction: %d records total", len(all_raw_records))

    # ── 5. Standardise ───────────────────────────────────────────────────
    logger.info("Stage 5: Standardise & Validate")
    standardised = []
    all_warnings = []

    for raw in all_raw_records:
        record = standardise_record(raw)
        record, warnings = validate_record(record)
        standardised.append(record)
        all_warnings.extend(warnings)

    if all_warnings:
        logger.warning("Validation produced %d warnings", len(all_warnings))

    # ── 6. Deduplicate ───────────────────────────────────────────────────
    logger.info("Stage 6: Deduplicate")
    deduped = deduplicate(standardised)
    logger.info("After deduplication: %d records", len(deduped))

    # ── 7. Output ────────────────────────────────────────────────────────
    df = to_dataframe(deduped)
    output_path = OUTPUT_DIR / output_filename
    df.to_csv(output_path, index=False)
    logger.info("Output saved to %s", output_path)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("Token usage: %s", tracker.summary())
    logger.info("Output: %d rows × %d columns", len(df), len(df.columns))
    for wb in wellbores:
        wb_rows = df[df["Wellbore"] == wb]
        logger.info("  %s: %d casing strings", wb, len(wb_rows))
    logger.info("=" * 60)

    return df


# ── Live demo: single URL ───────────────────────────────────────────────────

def run_single_url(
    url: str,
    wellbore_name: str | None = None,
) -> pd.DataFrame:
    """
    Run the pipeline on a single PDF URL (for the live demo).

    Parameters
    ----------
    url : str
        Direct URL to a SODIR PDF.
    wellbore_name : str | None
        Wellbore name. If None, will attempt to infer from the document.

    Returns
    -------
    pd.DataFrame
        Extracted casing data.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Live Demo: Single URL Pipeline")
    logger.info("URL: %s", url)
    logger.info("=" * 60)

    # Download
    local_path = download_single_url(url)

    # Create a synthetic WellDocument
    doc = WellDocument(
        wellbore=wellbore_name or "UNKNOWN",
        doc_type="LIVE_DEMO",
        doc_name=local_path.stem,
        url=url,
        fmt="pdf",
        size_kb=local_path.stat().st_size // 1024,
        npd_id=0,
        tier=1,
    )
    doc.local_path = local_path

    # Extract text
    content = extract_text(local_path)

    # If wellbore not given, try to infer from document text
    if wellbore_name is None:
        inferred = _infer_wellbore(content)
        if inferred:
            doc.wellbore = inferred
            logger.info("Inferred wellbore: %s", inferred)
        else:
            logger.warning("Could not infer wellbore name — using 'UNKNOWN'")

    # Process
    raw_records = process_document(doc)

    # Standardise
    standardised = []
    for raw in raw_records:
        record = standardise_record(raw)
        record, _ = validate_record(record)
        standardised.append(record)

    deduped = deduplicate(standardised)
    df = to_dataframe(deduped)

    # Save
    output_path = OUTPUT_DIR / f"live_demo_{doc.wellbore.replace('/', '_')}.csv"
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info("Live demo complete in %.1fs", elapsed)
    logger.info("Token usage: %s", tracker.summary())
    logger.info("Output: %d rows", len(df))
    print(df.to_string(index=False))

    return df


def _infer_wellbore(content: DocumentContent) -> str | None:
    """Try to find wellbore name (e.g., 7/11-1) in the document text."""
    import re
    # Match patterns like 7/11-1, 7/11-2, 15/9-19 A, etc.
    pattern = r"\b(\d{1,2}/\d{1,2}\s*-\s*\d{1,3}\s*[A-Z]?)\b"
    for page in content.pages[:5]:
        match = re.search(pattern, page.combined_text)
        if match:
            return match.group(1).replace(" ", "")
    return None
