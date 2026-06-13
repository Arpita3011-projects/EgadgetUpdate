"""
test_pdf_utils.py
─────────────────
Tests for safe_pdf_text Unicode sanitization.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pdf_utils import safe_pdf_text


def test_safe_pdf_text_replaces_emojis():
    text = '⚠ High risk 🟠 — student • name'
    result = safe_pdf_text(text)
    assert 'WARNING' in result
    assert 'HIGH' in result
    assert '—' not in result
    assert '•' not in result
    assert all(ord(c) < 128 for c in result)


def test_safe_pdf_text_handles_none():
    assert safe_pdf_text(None) == ''


def test_safe_pdf_text_strips_unicode_names():
    result = safe_pdf_text('Arpita Müller 日本')
    assert 'Arpita' in result
    assert all(ord(c) < 128 for c in result)
