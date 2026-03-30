"""
Standardisation and validation of extracted casing data.

Handles:
- Casing type normalisation
- Diameter fraction parsing (9 5/8 → 9.625)
- Unit validation (depths in metres, diameters in inches)
- Sanity checks (reasonable value ranges)
- Deduplication across multiple source documents
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from config import CASING_TYPE_MAP, COMMON_FRACTIONS, TYPICAL_CASING_HOLE_PAIRS

logger = logging.getLogger(__name__)

# Expected output columns in order
OUTPUT_COLUMNS = [
    "Wellbore",
    "Casing type",
    "Casing diameter [in]",
    "Casing depth [m]",
    "Hole diameter [in]",
    "Hole depth [m]",
    "LOT/FIT mud eqv. [g/cm3]",
    "Formation test type",
]

# Reasonable value ranges for sanity checking
VALID_RANGES = {
    "casing_diameter_in":   (4.0, 36.0),
    "hole_diameter_in":     (5.0, 42.0),
    "casing_depth_m":       (10.0, 8000.0),
    "hole_depth_m":         (10.0, 8000.0),
    "lot_fit_mud_eqv_gcm3": (0.8, 2.5),
}


# ── Fraction parsing ─────────────────────────────────────────────────────────

def parse_diameter(value: Any) -> float | None:
    """
    Parse a diameter value that may contain fractions.

    Examples:
        9.625 → 9.625
        "9 5/8" → 9.625
        "13-3/8" → 13.375
        "17 1/2" → 17.5
        "9⅝" → 9.625
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().replace('"', "").replace("'", "").replace("″", "")

    # Try direct float parse
    try:
        return float(s)
    except ValueError:
        pass

    # Pattern: "13-3/8" or "13 3/8" or "9 5/8"
    match = re.match(r"(\d+)[\s\-](\d+/\d+)", s)
    if match:
        whole = int(match.group(1))
        frac_str = match.group(2)
        frac_val = COMMON_FRACTIONS.get(frac_str)
        if frac_val is None:
            num, den = frac_str.split("/")
            frac_val = int(num) / int(den)
        return whole + frac_val

    # Pattern: just a fraction "5/8"
    match = re.match(r"^(\d+/\d+)$", s)
    if match:
        frac_val = COMMON_FRACTIONS.get(match.group(1))
        if frac_val is not None:
            return frac_val
        num, den = match.group(1).split("/")
        return int(num) / int(den)

    logger.debug("Could not parse diameter: %r", value)
    return None


# ── Casing type normalisation ────────────────────────────────────────────────

def normalise_casing_type(raw: Any) -> str | None:
    """Map raw casing type string to standard name."""
    if raw is None:
        return None

    raw_lower = str(raw).strip().lower()

    # Direct lookup
    if raw_lower in CASING_TYPE_MAP:
        return CASING_TYPE_MAP[raw_lower]

    # Partial match — check if any key is contained in raw
    for key, standard in CASING_TYPE_MAP.items():
        if key in raw_lower:
            return standard

    # Already in standard form?
    standard_values = set(CASING_TYPE_MAP.values())
    raw_title = str(raw).strip().title()
    if raw_title in standard_values:
        return raw_title

    logger.warning("Unknown casing type: %r — keeping as-is", raw)
    return str(raw).strip()


# ── Formation test type normalisation ────────────────────────────────────────

def normalise_test_type(raw: Any) -> str | None:
    """Normalise formation test type to LOT or FIT."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if s in ("LOT", "LEAK-OFF TEST", "LEAK OFF TEST", "LEAKOFF TEST", "LEAK-OFF"):
        return "LOT"
    if s in ("FIT", "FORMATION INTEGRITY TEST", "FORMATION INTEGRITY"):
        return "FIT"
    if s in ("NULL", "NONE", "N/A", ""):
        return None
    logger.debug("Unknown test type: %r", raw)
    return s


# ── Record standardisation ──────────────────────────────────────────────────

def standardise_record(record: dict) -> dict:
    """
    Standardise a single raw extraction record.
    """
    return {
        "Wellbore":                  str(record.get("wellbore", "")).strip(),
        "Casing type":               normalise_casing_type(record.get("casing_type")),
        "Casing diameter [in]":      parse_diameter(record.get("casing_diameter_in")),
        "Casing depth [m]":          _to_float(record.get("casing_depth_m")),
        "Hole diameter [in]":        parse_diameter(record.get("hole_diameter_in")),
        "Hole depth [m]":            _to_float(record.get("hole_depth_m")),
        "LOT/FIT mud eqv. [g/cm3]":  _to_float(record.get("lot_fit_mud_eqv_gcm3")),
        "Formation test type":       normalise_test_type(record.get("formation_test_type")),
    }


def _to_float(value: Any) -> float | None:
    """Safe float conversion."""
    if value is None:
        return None
    try:
        v = float(value)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


# ── Validation ───────────────────────────────────────────────────────────────

def validate_record(record: dict) -> tuple[dict, list[str]]:
    """
    Validate a standardised record. Returns the record and a list of warnings.
    """
    warnings = []

    for field, (lo, hi) in VALID_RANGES.items():
        col_name = {
            "casing_diameter_in":   "Casing diameter [in]",
            "hole_diameter_in":     "Hole diameter [in]",
            "casing_depth_m":       "Casing depth [m]",
            "hole_depth_m":         "Hole depth [m]",
            "lot_fit_mud_eqv_gcm3": "LOT/FIT mud eqv. [g/cm3]",
        }[field]
        val = record.get(col_name)
        if val is not None and not (lo <= val <= hi):
            warnings.append(
                f"{col_name}={val} outside expected range [{lo}, {hi}]"
            )

    # Casing depth should not exceed hole depth
    cd = record.get("Casing depth [m]")
    hd = record.get("Hole depth [m]")
    if cd is not None and hd is not None and cd > hd:
        warnings.append(
            f"Casing depth ({cd}m) > hole depth ({hd}m)"
        )

    # Casing diameter should be smaller than hole diameter
    c_dia = record.get("Casing diameter [in]")
    h_dia = record.get("Hole diameter [in]")
    if c_dia is not None and h_dia is not None and c_dia >= h_dia:
        warnings.append(
            f"Casing diameter ({c_dia}\") >= hole diameter ({h_dia}\")"
        )

    # LOT/FIT value without test type or vice versa
    lot = record.get("LOT/FIT mud eqv. [g/cm3]")
    test = record.get("Formation test type")
    if lot is not None and test is None:
        warnings.append("LOT/FIT value present but no test type specified")
    if test is not None and lot is None:
        warnings.append("Formation test type specified but no LOT/FIT value")

    if warnings:
        logger.debug(
            "Validation warnings for %s %s: %s",
            record.get("Wellbore"),
            record.get("Casing type"),
            "; ".join(warnings),
        )

    return record, warnings


# ── Deduplication ────────────────────────────────────────────────────────────

def deduplicate(records: list[dict]) -> list[dict]:
    """
    Deduplicate casing records for a wellbore.

    Strategy: group by (wellbore, casing_type, casing_diameter). If duplicates
    exist, keep the record with the most non-null fields.
    """
    if not records:
        return []

    def _dedup_key(r: dict) -> tuple:
        return (
            r.get("Wellbore", ""),
            r.get("Casing type", ""),
            r.get("Casing diameter [in]"),
        )

    def _completeness(r: dict) -> int:
        return sum(1 for v in r.values() if v is not None)

    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = _dedup_key(r)
        groups.setdefault(key, []).append(r)

    deduped = []
    for key, group in groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Merge: take the most complete record as base, fill gaps from others
            group.sort(key=_completeness, reverse=True)
            best = dict(group[0])
            for other in group[1:]:
                for k, v in other.items():
                    if best.get(k) is None and v is not None:
                        best[k] = v
            deduped.append(best)
            logger.debug("Merged %d duplicate records for %s", len(group), key)

    # Sort by wellbore, then by depth (shallowest first)
    deduped.sort(
        key=lambda r: (
            r.get("Wellbore", ""),
            r.get("Casing depth [m]") or 0,
        )
    )
    return deduped


# ── DataFrame output ─────────────────────────────────────────────────────────

def to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert standardised records to a clean DataFrame."""
    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(records)

    # Ensure all output columns exist
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[OUTPUT_COLUMNS]

    # Round numeric columns
    for col in ["Casing diameter [in]", "Hole diameter [in]"]:
        df[col] = df[col].apply(lambda x: round(x, 3) if pd.notna(x) else None)
    for col in ["Casing depth [m]", "Hole depth [m]"]:
        df[col] = df[col].apply(lambda x: round(x, 1) if pd.notna(x) else None)
    df["LOT/FIT mud eqv. [g/cm3]"] = df["LOT/FIT mud eqv. [g/cm3]"].apply(
        lambda x: round(x, 2) if pd.notna(x) else None
    )

    return df
