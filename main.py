#!/usr/bin/env python3
"""
Well Document Pipeline.

Usage:
    # Full pipeline on the Cod field dataset
    python main.py --csv wellbore_document_7_11.csv

    # Process only Tier 1 docs (completion reports)
    python main.py --csv wellbore_document_7_11.csv --max-tier 1

    # Live demo: single URL
    python main.py --url https://factpages.sodir.no/pbl/wellbore_documents/XXXX.pdf

    # Live demo with known wellbore
    python main.py --url https://factpages.sodir.no/pbl/... --wellbore "15/9-19 A"
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import run_full_pipeline, run_single_url


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Extract casing design data from SODIR well PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode: batch (CSV) or single URL
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--csv",
        type=str,
        help="Path to the SODIR CSV file (batch mode)",
    )
    mode.add_argument(
        "--url",
        type=str,
        help="Single PDF URL (live demo mode)",
    )

    # Options
    parser.add_argument(
        "--max-tier",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Maximum document tier to process (default: 2)",
    )
    parser.add_argument(
        "--wellbore",
        type=str,
        default=None,
        help="Wellbore name for single-URL mode (auto-inferred if omitted)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="casing_data_cod_field.csv",
        help="Output filename (default: casing_data_cod_field.csv)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.csv:
        # ── Batch mode ───────────────────────────────────────────────
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)

        df = run_full_pipeline(
            csv_path=csv_path,
            max_tier=args.max_tier,
            output_filename=args.output,
        )

        print("\n" + "=" * 80)
        print("FINAL OUTPUT")
        print("=" * 80)
        print(df.to_string(index=False))
        print(f"\nSaved to: output/{args.output}")

    else:
        # ── Live demo mode ───────────────────────────────────────────
        df = run_single_url(
            url=args.url,
            wellbore_name=args.wellbore,
        )


if __name__ == "__main__":
    main()
