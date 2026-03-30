# SODIR Well Document Pipeline

Extract casing design data from Norwegian Continental Shelf (NCS) well documents sourced from [SODIR Factpages](https://factpages.sodir.no/).

## Output Schema

| Column | Description |
|---|---|
| Wellbore | Wellbore name (e.g., 7/11-1) |
| Casing type | Conductor, Surface Casing, Intermediate Casing, Production Casing, Liner |
| Casing diameter [in] | Outer diameter of casing in inches |
| Casing depth [m] | Shoe / setting depth in metres MD |
| Hole diameter [in] | Bit / hole size in inches |
| Hole depth [m] | Total depth of hole section in metres MD |
| LOT/FIT mud eqv. [g/cm3] | Leak-off or formation integrity test result as equivalent mud weight |
| Formation test type | LOT or FIT |

## Architecture

```
CSV metadata
     │
     ▼
┌──────────┐     ┌────────────┐     ┌──────────────┐     ┌──────────────┐
│  Triage  │────▶│  Download   │────▶│  PDF Extract  │────▶│  LLM Extract │
│ (no LLM) │     │  (cached)   │     │  text/tables  │     │   (OpenAI)   │
└──────────┘     └────────────┘     └──────────────┘     └──────────────┘
  118 docs          ~18 docs          page scoring            JSON out
  → 18 docs                           scanned detect
                                                                  │
                                                                  ▼
                                                        ┌──────────────┐
                                                        │ Standardise  │
                                                        │  & Validate  │
                                                        └──────────────┘
                                                                  │
                                                                  ▼
                                                            output CSV
```

**Key efficiency features:**
- Static triage cuts 118 documents → ~18 using filename keywords (zero tokens)
- Tier 2 docs get a cheap relevance check before full extraction
- Pages are scored for casing keywords; only top pages are sent to the LLM
- Scanned PDFs are detected and routed to OpenAI vision
- Results are cached locally to avoid re-downloading

## Setup

```bash
# 1. Clone and install
git clone <repo-url> && cd sodir-pipeline
pip install -r requirements.txt

# 2. Set your API key
export OPENAI_API_KEY="sk-...."
```

## Usage

### Batch mode (full dataset)

```bash
# Process Tier 1 + 2 documents (recommended)
python main.py --csv wellbore_document_7_11.csv

# Process only completion reports (faster, less complete)
python main.py --csv wellbore_document_7_11.csv --max-tier 1

# Verbose logging
python main.py --csv wellbore_document_7_11.csv -v
```

### Live demo (single URL)

```bash
# Auto-infer wellbore name from the document
python main.py --url https://factpages.sodir.no/pbl/wellbore_documents/XXXXX.pdf

# Specify the wellbore explicitly
python main.py --url https://factpages.sodir.no/pbl/wellbore_documents/XXXXX.pdf \
    --wellbore "15/9-19 A"
```

## Project Structure

```
sodir-pipeline/
├── main.py                          # CLI entry point
├── requirements.txt
├── sodir_pipeline/
│   ├── __init__.py
│   ├── config.py                    # Constants, prompts, keyword lists
│   ├── triage.py                    # CSV loading, document classification
│   ├── downloader.py                # PDF download with caching
│   ├── extractor.py                 # Text/table extraction, scanned detection
│   ├── llm_client.py                # OpenAI API: relevance check + extraction
│   ├── standardiser.py              # Normalisation, validation, deduplication
│   └── pipeline.py                  # Main orchestration
├── cache/                           # Downloaded PDFs (git-ignored)
└── output/                          # Final CSV results
```

## Document Triage Strategy

| Tier | Criteria | Action | Typical Count |
|------|----------|--------|---------------|
| 1 | Completion reports, individual well records, completion logs | Process immediately | ~10 |
| 2 | WDSS summaries, drilling fluid summaries, formation tests, AAODC reports | Relevance-check then extract | ~8 |
| 3 | Geochemistry, core analysis, biostratigraphy, lithology, seismic, DSTs | Skip | ~100 |

## Token Budget Estimate

| Document type | Docs | Tokens/doc | Total |
|---|---|---|---|
| WDSS summaries | 5 | ~2,500 | ~12,500 |
| Completion reports (digital) | 8 | ~10,000 | ~80,000 |
| Large scanned docs (vision) | 2 | ~15,000 | ~30,000 |
| Tier 2 supplementary | 5 | ~5,000 | ~25,000 |
| Relevance checks (Tier 2) | 8 | ~500 | ~4,000 |
| **Total** | | | **~150K input + ~10K output** |
