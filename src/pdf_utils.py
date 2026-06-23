"""
pdf_utils.py
────────────
Reusable helpers for FPDF text sanitization (Helvetica is Latin-1 only).
"""

from __future__ import annotations

import re
from typing import Any, Union

# Common Unicode symbols used in UI/reports → ASCII-safe replacements
_UNICODE_REPLACEMENTS: dict[str, str] = {
    '⚠': 'WARNING',
    '⚠️': 'WARNING',
    '🟢': 'LOW',
    '🟡': 'MODERATE',
    '🟠': 'HIGH',
    '🔴': 'SEVERE',
    '📊': 'REPORT',
    '📄': 'REPORT',
    '📱': 'GADGET',
    '📥': 'DOWNLOAD',
    '🎯': 'SCORE',
    '💡': 'TIP',
    '🔍': 'INFO',
    '✓': 'PASS',
    '✅': 'PASS',
    '🆘': 'CRITICAL',
    '•': '*',
    '—': '-',
    '–': '-',
    '…': '...',
    ''': "'",
    ''': "'",
    '"': '"',
    '"': '"',
}


def safe_pdf_text(text: Any) -> str:
    """
    Sanitize text for fpdf2 Helvetica output.

    Replaces known emoji/symbols with ASCII words, then strips any
    remaining non-ASCII characters so PDF generation never fails on Unicode.

    Parameters
    ----------
    text : Any
        Value to write into a PDF cell/multi_cell.

    Returns
    -------
    str
        ASCII-safe string suitable for FPDF Helvetica fonts.
    """
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return re.sub(r'[^\x00-\x7F]', '', text)
