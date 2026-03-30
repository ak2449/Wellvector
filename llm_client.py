"""
LLM client — handles all OpenAI API interactions.

Two main operations:
1. Relevance check: quick yes/no — does this document have casing data?
2. Structured extraction: pull casing design JSON from relevant pages.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RELEVANCE_CHECK_PROMPT,
)

logger = logging.getLogger(__name__)


_CASING_RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "wellbore": {"type": "string"},
                    "casing_type": {
                        "enum": [
                            "Conductor",
                            "Surface Casing",
                            "Intermediate Casing",
                            "Production Casing",
                            "Liner",
                            None,
                        ]
                    },
                    "casing_diameter_in": {"type": ["number", "null"]},
                    "casing_depth_m": {"type": ["number", "null"]},
                    "hole_diameter_in": {"type": ["number", "null"]},
                    "hole_depth_m": {"type": ["number", "null"]},
                    "lot_fit_mud_eqv_gcm3": {"type": ["number", "null"]},
                    "formation_test_type": {"enum": ["LOT", "FIT", None]},
                },
                "required": [
                    "wellbore",
                    "casing_type",
                    "casing_diameter_in",
                    "casing_depth_m",
                    "hole_diameter_in",
                    "hole_depth_m",
                    "lot_fit_mud_eqv_gcm3",
                    "formation_test_type",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["records"],
    "additionalProperties": False,
}


def _get_client() -> OpenAI:
    """Initialise OpenAI client (reads OPENAI_API_KEY from env)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set OPENAI_API_KEY environment variable before running the pipeline."
        )
    return OpenAI(api_key=api_key)


# ── Token tracking ───────────────────────────────────────────────────────────

class TokenTracker:
    """Simple token usage accumulator."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_calls = 0

    def record(self, usage: Any):
        self.input_tokens += _usage_value(usage, "input_tokens")
        self.output_tokens += _usage_value(usage, "output_tokens")
        self.api_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        return (
            f"API calls: {self.api_calls} | "
            f"Input tokens: {self.input_tokens:,} | "
            f"Output tokens: {self.output_tokens:,} | "
            f"Total: {self.total_tokens:,}"
        )


# Global tracker
tracker = TokenTracker()


# ── API calls ────────────────────────────────────────────────────────────────

def _usage_value(usage: Any, field: str) -> int:
    """Read usage fields from SDK objects or plain dicts."""
    value = getattr(usage, field, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(field)
    return int(value or 0)


def _response_text(response: Any) -> str:
    """Extract concatenated text from a Responses API object."""
    text = getattr(response, "output_text", None)
    if text:
        return text

    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def _call_api(
    instructions: str,
    input_items: list[dict],
    max_tokens: int = LLM_MAX_TOKENS,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Make a single OpenAI Responses API call with retry logic."""
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": LLM_MODEL,
        "instructions": instructions,
        "input": input_items,
        "max_output_tokens": max_tokens,
        "temperature": LLM_TEMPERATURE,
    }
    if response_format is not None:
        kwargs["text"] = {"format": response_format}

    response = client.responses.create(**kwargs)
    tracker.record(response.usage)
    return _response_text(response)


# ── Relevance check ─────────────────────────────────────────────────────────

def check_relevance(text_sample: str) -> bool:
    """
    Quick check: does this text contain casing design information?
    Uses a very short prompt to minimise tokens.

    Parameters
    ----------
    text_sample : str
        First ~2000 chars of the document.

    Returns
    -------
    bool
        True if the document likely contains casing data.
    """
    # Truncate to save tokens — we only need a flavour of the document
    text_sample = text_sample[:2000]

    prompt = RELEVANCE_CHECK_PROMPT.format(text=text_sample)
    response = _call_api(
        instructions="Answer with only 'yes' or 'no'.",
        input_items=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
        max_tokens=8,
    )
    answer = response.strip().lower()
    return answer.startswith("yes")


# ── Structured extraction ───────────────────────────────────────────────────

def _parse_json_response(raw: str) -> list[dict]:
    """Parse model response into a list of casing records."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict) and isinstance(result.get("records"), list):
            return result["records"]
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        return []
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the response
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from LLM response: %s...", raw[:200])
    return []


def extract_casing_data(
    document_text: str,
    wellbore: str,
) -> list[dict]:
    """
    Extract casing design data from document text using OpenAI.

    Parameters
    ----------
    document_text : str
        The relevant page text to analyse.
    wellbore : str
        Wellbore name (e.g., "7/11-1").

    Returns
    -------
    list[dict]
        Extracted casing records.
    """
    if not document_text.strip():
        return []

    prompt = EXTRACTION_USER_PROMPT.format(
        wellbore=wellbore,
        document_text=document_text,
    )

    response = _call_api(
        instructions=EXTRACTION_SYSTEM_PROMPT,
        input_items=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
        response_format={
            "type": "json_schema",
            "name": "casing_records",
            "strict": True,
            "schema": _CASING_RECORD_SCHEMA,
        },
    )

    records = _parse_json_response(response)
    logger.info("Extracted %d casing records for %s", len(records), wellbore)
    return records


def extract_casing_data_vision(
    page_images: list[tuple[int, bytes]],
    wellbore: str,
) -> list[dict]:
    """
    Extract casing data from scanned PDF page images using OpenAI vision.

    Parameters
    ----------
    page_images : list[tuple[int, bytes]]
        List of (page_index, png_bytes) tuples.
    wellbore : str
        Wellbore name.

    Returns
    -------
    list[dict]
        Extracted casing records.
    """
    if not page_images:
        return []

    # Build multimodal message content
    content: list[dict] = []
    for page_idx, png_bytes in page_images:
        b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
        content.append({
            "type": "input_text",
            "text": f"Page {page_idx + 1}:",
        })
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{b64}",
            "detail": "auto",
        })

    # Add the extraction prompt at the end
    prompt_text = EXTRACTION_USER_PROMPT.format(
        wellbore=wellbore,
        document_text="[See page images above]",
    )
    content.append({"type": "input_text", "text": prompt_text})

    response = _call_api(
        instructions=EXTRACTION_SYSTEM_PROMPT,
        input_items=[{"role": "user", "content": content}],
        response_format={
            "type": "json_schema",
            "name": "casing_records",
            "strict": True,
            "schema": _CASING_RECORD_SCHEMA,
        },
    )

    records = _parse_json_response(response)
    logger.info(
        "Vision extraction: %d records for %s from %d pages",
        len(records), wellbore, len(page_images),
    )
    return records
