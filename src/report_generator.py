"""
report_generator.py
───────────────────
Generates a PDF risk report for a student using fpdf2.

Usage:
    from report_generator import generate_pdf_report
    path = generate_pdf_report("Arpita", "High", 74.2, tips, shap_dict)
"""

from __future__ import annotations

import datetime
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF, XPos, YPos

from logging_config import get_logger
from model_evaluation import load_model_metadata
from pdf_utils import safe_pdf_text

logger = get_logger('report')

CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']


class ReportPDF(FPDF):
    """Single-student report with branded header/footer."""

    def header(self) -> None:
        self.set_font('Helvetica', 'B', 13)
        self.set_fill_color(30, 90, 160)
        self.set_text_color(255, 255, 255)
        self.cell(
            0, 12,
            safe_pdf_text('  E-Gadget Addiction Risk Report'),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True,
        )
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 8,
            safe_pdf_text(
                f'KLE College of Engineering · Confidential · Page {self.page_no()}'
            ),
            align='C',
        )


def _risk_gauge_image(score: float, save_path: str) -> str:
    """Render addiction risk score gauge as PNG."""
    fig, ax = plt.subplots(figsize=(8, 1.2))
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 100, 0, 1])
    ax.axvline(score, color='black', linewidth=3)
    ax.text(score, 0.5, f' {score}', va='center', fontsize=11, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlabel('Addiction Risk Score (0 = No Risk, 100 = Severe)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def generate_pdf_report(
    student_name: str,
    risk_label: str,
    score: float,
    tips: List[str],
    shap_impact: Optional[Dict[str, float]] = None,
    waterfall_img: Optional[str] = None,
    save_dir: Optional[str] = None,
    confidence_pct: Optional[float] = None,
) -> str:
    """
    Build and save a single-student PDF assessment report.

    Parameters
    ----------
    student_name : str
        Display name.
    risk_label : str
        One of Low, Moderate, High, Severe.
    score : float
        Addiction risk score 0-100.
    tips : list[str]
        Personalised recommendation strings.
    shap_impact : dict, optional
        {feature: shap_value} mapping.
    waterfall_img : str, optional
        Path to SHAP waterfall PNG.
    save_dir : str, optional
        Output directory (default: reports/).
    confidence_pct : float, optional
        Model prediction confidence percentage.

    Returns
    -------
    str
        Absolute path to the generated PDF.
    """
    logger.info(
        'Generating student PDF for %s (risk=%s, score=%.1f)',
        student_name or 'Anonymous', risk_label, score,
    )

    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'reports',
        )
    os.makedirs(save_dir, exist_ok=True)

    pdf = ReportPDF()
    pdf.add_page()

    # Student info
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(
        0, 8,
        safe_pdf_text(f'Student : {student_name or "Anonymous"}'),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.cell(
        0, 8,
        safe_pdf_text(f'Date    : {datetime.date.today().strftime("%B %d, %Y")}'),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(3)

    # Risk badge
    color_map = {
        'Low': (39, 174, 96),
        'Moderate': (243, 156, 18),
        'High': (230, 126, 34),
        'Severe': (192, 57, 43),
    }
    r, g, b = color_map.get(risk_label, (0, 0, 0))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(
        60, 12,
        safe_pdf_text(f'  Risk Level: {risk_label}'),
        fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 12)
    conf_text = f'   Addiction Risk Score: {score} / 100'
    if confidence_pct is not None:
        conf_text += f'   |   Confidence: {confidence_pct:.1f}%'
    pdf.cell(0, 12, safe_pdf_text(conf_text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Risk gauge chart
    gauge_path = None
    try:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='_gauge.png')
        gauge_path = _risk_gauge_image(score, tmpf.name)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, safe_pdf_text('Risk Score Gauge:'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(gauge_path, x=10, w=185)
        pdf.ln(4)
    except Exception as exc:
        logger.warning('Risk gauge chart failed: %s', exc)

    # SHAP feature impacts
    tmp_chart = None
    if shap_impact:
        if not waterfall_img:
            try:
                feats = list(shap_impact.items())[:8]
                names = [n for n, _ in feats]
                vals = [v for _, v in feats]
                fig, ax = plt.subplots(figsize=(6, 2))
                colors = ['#27ae60' if v <= 0 else '#c0392b' for v in vals]
                ax.barh(names, [abs(v) for v in vals], color=colors)
                ax.set_xlabel('SHAP impact (abs)')
                ax.invert_yaxis()
                plt.tight_layout()
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                fig.savefig(tmpf.name, dpi=150)
                plt.close(fig)
                waterfall_img = tmpf.name
                tmp_chart = tmpf.name
            except Exception as exc:
                logger.warning('Fallback SHAP bar chart failed: %s', exc)
                waterfall_img = None

        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(
            0, 8,
            safe_pdf_text('Key Contributing Factors (SHAP):'),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.set_font('Helvetica', '', 10)
        for feat, val in list(shap_impact.items())[:6]:
            arrow = 'increases' if val > 0 else 'decreases'
            pdf.cell(
                0, 7,
                safe_pdf_text(f'  * {feat}: {val:+.3f}  ({arrow} risk)'),
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        pdf.ln(2)

    # Waterfall image
    if waterfall_img and os.path.isfile(waterfall_img):
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(
            0, 8,
            safe_pdf_text('SHAP Waterfall Explanation:'),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.image(waterfall_img, x=10, w=185)
        pdf.ln(4)

    # Recommendations
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(
        0, 8,
        safe_pdf_text('Personalised Recommendations:'),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_font('Helvetica', '', 10)
    for tip in tips:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, safe_pdf_text(f'  {tip}'))
    pdf.ln(3)

    # Model metadata snippet
    metadata = load_model_metadata()
    if metadata:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(
            0, 7,
            safe_pdf_text('Model Information:'),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.set_font('Helvetica', '', 9)
        meta_lines = [
            f"Algorithm: {metadata.get('algorithm', 'Random Forest')}",
            f"Model Accuracy: {metadata.get('accuracy', 'N/A')}",
            f"Trained: {metadata.get('date_trained', 'N/A')}",
        ]
        for line in meta_lines:
            pdf.cell(0, 6, safe_pdf_text(f'  {line}'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # Disclaimer
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 6,
        safe_pdf_text(
            'This report is generated by an AI-based prediction system and is intended '
            'as a supportive tool only. It does not constitute medical or psychological '
            'advice. Please consult a qualified professional for clinical assessment.'
        ),
    )

    safe_name = safe_pdf_text(student_name or 'student').replace(' ', '_')
    safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', safe_name)
    filename = f"{safe_name}_{datetime.date.today()}_report.pdf"
    pdf_path = os.path.join(save_dir, filename)
    pdf.output(pdf_path)

    for tmp in (tmp_chart, gauge_path):
        try:
            if tmp and os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass

    logger.info('Student PDF saved -> %s', pdf_path)
    return pdf_path


if __name__ == '__main__':
    sample_tips = [
        'Set daily screen time limits.',
        'Sleep by 10 PM.',
        'Increase physical activity.',
    ]
    generate_pdf_report(
        student_name='Demo Student',
        risk_label='High',
        score=72.5,
        tips=sample_tips,
        confidence_pct=78.5,
    )
