"""
Document triage — classify documents into priority tiers using CSV metadata.
No PDFs are downloaded and no LLM calls are made at this stage.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from config import TIER_1_KEYWORDS, TIER_2_KEYWORDS

logger = logging.getLogger(__name__)

Tier = Literal[1, 2, 3]


@dataclass
class WellDocument:
    """A single well document record from the CSV."""
    wellbore: str
    doc_type: str
    doc_name: str
    url: str
    fmt: str
    size_kb: int
    npd_id: int
    tier: Tier = 3
    local_path: Path | None = None

    # Populated after extraction
    extraction_result: list[dict] = field(default_factory=list)
    tokens_used: int = 0

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]

    def __repr__(self) -> str:
        return f"WellDocument({self.wellbore!r}, tier={self.tier}, {self.doc_name!r})"


def _classify(doc_name: str) -> Tier:
    """Assign a priority tier based on document name keywords."""
    name_lower = doc_name.lower()
    for kw in TIER_1_KEYWORDS:
        if kw in name_lower:
            return 1
    for kw in TIER_2_KEYWORDS:
        if kw in name_lower:
            return 2
    return 3


def load_and_triage(csv_path: str | Path) -> list[WellDocument]:
    """
    Read the CSV and return documents sorted by processing priority.

    Returns
    -------
    list[WellDocument]
        Sorted: Tier 1 first, then Tier 2, then Tier 3.
        Within each tier, smaller files come first (cheaper to process).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    docs: list[WellDocument] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = WellDocument(
                wellbore=row["wlbName"].strip(),
                doc_type=row["wlbDocumentType"].strip(),
                doc_name=row["wlbDocumentName"].strip(),
                url=row["wlbDocumentUrl"].strip(),
                fmt=row.get("wlbDocumentFormat", "pdf").strip().lower(),
                size_kb=int(row.get("wlbDocumentSize", 0)),
                npd_id=int(row.get("wlbNpdidWellbore", 0)),
            )
            doc.tier = _classify(doc.doc_name)
            docs.append(doc)

    # Sort: tier ascending, then size ascending (small first)
    docs.sort(key=lambda d: (d.tier, d.size_kb))

    # Summary logging
    tier_counts = {1: 0, 2: 0, 3: 0}
    for d in docs:
        tier_counts[d.tier] += 1
    logger.info(
        "Triage complete: %d docs → Tier 1: %d, Tier 2: %d, Tier 3 (skip): %d",
        len(docs), tier_counts[1], tier_counts[2], tier_counts[3],
    )
    return docs


def get_wellbore_names(docs: list[WellDocument]) -> list[str]:
    """Return sorted unique wellbore names."""
    return sorted({d.wellbore for d in docs})


def filter_by_tier(docs: list[WellDocument], max_tier: int = 2) -> list[WellDocument]:
    """Return only documents up to the given tier."""
    return [d for d in docs if d.tier <= max_tier]


def group_by_wellbore(docs: list[WellDocument]) -> dict[str, list[WellDocument]]:
    """Group documents by wellbore name."""
    groups: dict[str, list[WellDocument]] = {}
    for d in docs:
        groups.setdefault(d.wellbore, []).append(d)
    return groups
