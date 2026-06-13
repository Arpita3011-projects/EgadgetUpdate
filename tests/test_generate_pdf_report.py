import sys
from pathlib import Path

import pytest


# Ensure src/ is importable (matches how app modules load src)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from report_generator import generate_pdf_report


def test_generate_pdf_report_creates_valid_pdf(tmp_path):
    """Generate a PDF and verify: exists, size>5000, readable by pypdf, contains student name."""
    save_dir = tmp_path / "reports"
    save_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = generate_pdf_report(
        student_name="TestStudent",
        risk_label="High",
        score=74.0,
        tips=["Tip 1", "Tip 2", "Tip 3"],
        shap_impact={"daily_screen_time_hours": 0.4, "gpa": -0.2},
        save_dir=str(save_dir),
    )

    p = Path(pdf_path)
    assert p.exists(), f"PDF was not created at {pdf_path}"
    assert p.stat().st_size > 5000, f"PDF too small ({p.stat().st_size} bytes)"

    # Verify pypdf can open it and first page contains the student name
    try:
        from pypdf import PdfReader
    except Exception as e:
        pytest.skip(f"pypdf not available: {e}")

    reader = PdfReader(str(p))
    assert len(reader.pages) >= 1
    first_page = reader.pages[0]
    text = first_page.extract_text() or ""
    assert "TestStudent" in text, "Student name not found on first page"
