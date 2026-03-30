"""
Configuration: triage rules, normalisation maps, LLM prompts, and constants.
"""

from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"       # downloaded PDFs
OUTPUT_DIR = PROJECT_ROOT / "output"     # final results
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── LLM Settings ─────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-5.4"
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.0          # deterministic extraction
MAX_PAGES_PER_LLM_CALL = 9     # cap pages sent to avoid bloated context
MAX_VISION_PAGES = 9            # cap for image-based (scanned) extraction

# ── Document Triage ──────────────────────────────────────────────────────────
# Tier 1: almost certainly contains casing design tables
TIER_1_KEYWORDS = [
    "well_completion_report",
    "completion_report",
    "completion_log",
    "individual_well_record",
]

# Tier 2: may contain partial casing/hole/LOT data
TIER_2_KEYWORDS = [
    "wdss",
    "drilling_fluid_summary",
    "formation_test",
    "drilling_program",
    "exploratory_test_drilling",
    "aaodc_report",
    "boring_programme",
    "final_well_report",
    "well_summary",
]

# Tier 3 (skip): everything not matching Tier 1 or 2

# ── Page Relevance Keywords ──────────────────────────────────────────────────
CASING_PAGE_KEYWORDS = [
    "casing", "conductor", "surface casing", "intermediate casing",
    "production casing", "liner", "hole section", "hole size",
    "bit size", "shoe depth", "shoe @ ", "setting depth",
    "LOT", "FIT", "leak-off", "leak off", "formation integrity",
    "mud weight", "mud eqv", "cement", "wellbore diagram",
    "well schematic", "tubular", "csg", "casing programme",
    "casing program", "casing string", "casing summary",
    "well data", "well design",
]

# ── Casing Type Normalisation ────────────────────────────────────────────────
CASING_TYPE_MAP = {
    # conductors
    "conductor":            "Conductor",
    "conductor casing":     "Conductor",
    "conductor pipe":       "Conductor",
    "structural casing":    "Conductor",
    "drive pipe":           "Conductor",
    # surface
    "surface":              "Surface Casing",
    "surface casing":       "Surface Casing",
    "surface csg":          "Surface Casing",
    # intermediate
    "intermediate":         "Intermediate Casing",
    "intermediate casing":  "Intermediate Casing",
    "intermediate csg":     "Intermediate Casing",
    # production
    "production":           "Production Casing",
    "production casing":    "Production Casing",
    "production csg":       "Production Casing",
    # liner
    "liner":                "Liner",
    "production liner":     "Liner",
    "drilling liner":       "Liner",
}

# Common casing OD → expected hole size (sanity check)
TYPICAL_CASING_HOLE_PAIRS = {
    30.0:   36.0,
    20.0:   26.0,
    18.625: 24.0,
    13.375: 17.5,
    9.625:  12.25,
    7.0:    8.5,
    5.5:    8.5,
    5.0:    6.125,
}

# ── Unit Conversions ─────────────────────────────────────────────────────────
FEET_TO_METRES = 0.3048
PSI_PER_FT_TO_GCM3 = 2.3067   # (psi/ft) → (g/cm³) via 0.052 factor

# ── Fraction Parsing ─────────────────────────────────────────────────────────
COMMON_FRACTIONS = {
    "1/8": 0.125, "1/4": 0.25, "3/8": 0.375, "1/2": 0.5,
    "5/8": 0.625, "3/4": 0.75, "7/8": 0.875,
    "1/3": 0.333, "2/3": 0.667, "1/16": 0.0625,
    "3/16": 0.1875, "5/16": 0.3125, "7/16": 0.4375,
    "9/16": 0.5625, "11/16": 0.6875, "13/16": 0.8125,
    "15/16": 0.9375,
}

# ── LLM Prompts ──────────────────────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """\
You are a petroleum engineering data-extraction assistant. You extract casing \
design information from Norwegian Continental Shelf well documents with high \
accuracy. You return ONLY valid JSON — no commentary, no markdown fences."""

EXTRACTION_USER_PROMPT = """\
Extract ALL casing string / hole section data from the following well document \
pages for wellbore {wellbore}.

Return a JSON array. Each element represents one casing string or open-hole \
section, ordered from shallowest (largest diameter) to deepest (smallest):

[
  {{
    "wellbore": "{wellbore}",
    "casing_type": "<Conductor | Surface Casing | Intermediate Casing | Production Casing | Liner>",
    "casing_diameter_in": <number or null>,
    "casing_depth_m": <number or null>,
    "hole_diameter_in": <number or null>,
    "hole_depth_m": <number or null>,
    "lot_fit_mud_eqv_gcm3": <number or null>,
    "formation_test_type": "<LOT | FIT | null>"
  }}
]

Rules:
1. Depths must be in metres (MD). If the source uses feet, convert (× 0.3048).
2. Diameters in inches. Convert fractions to decimals (9 5/8" → 9.625).
3. "Casing depth" = shoe depth / setting depth of the casing.
4. "Hole depth" = total depth drilled for that hole section.
5. LOT/FIT values in g/cm³ (equivalent mud weight / EMW).
   - If given as psi or pressure, convert using TVD if available.
   - If given as specific gravity (SG) — SG and g/cm³ are identical, keep as-is.
6. If data for a field is not present, use null.
7. If you find NO casing data at all, return an empty array [].

DOCUMENT TEXT:
---
{document_text}
---

Return ONLY the JSON array."""

RELEVANCE_CHECK_PROMPT = """\
Does the following document page contain information about well casing design, \
hole sections, casing setting depths, or leak-off / formation integrity tests?

Answer with ONLY "yes" or "no".

TEXT:
---
{text}
---"""
