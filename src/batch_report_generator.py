"""
batch_report_generator.py
─────────────────────────
Generates two PDF reports:
    1. CLASS REPORT  - overall analytics for the teacher
    2. STUDENT CARDS - one page per student with their risk + recommendations

Also generates an Excel export with colour-coded results.
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logging_config import get_logger
from model_evaluation import load_model_metadata
from pdf_utils import safe_pdf_text
from recommendation import get_recommendations

logger = get_logger('batch_report')

CLASS_NAMES = ['Low', 'Moderate', 'High', 'Severe']
RISK_COLORS_HEX = {
    'Low': ('#27ae60', (39, 174, 96)),
    'Moderate': ('#f39c12', (243, 156, 18)),
    'High': ('#e67e22', (230, 126, 34)),
    'Severe': ('#c0392b', (192, 57, 43)),
}


# ── Matplotlib helpers ────────────────────────────────────────────────────────

def _pie_chart(distribution: dict, title: str, save_path: str) -> None:
    labels = list(distribution.keys())
    sizes = list(distribution.values())
    colors = [RISK_COLORS_HEX[l][0] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    )
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _bar_chart(data: dict, xlabel: str, title: str, save_path: str) -> None:
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(labels[::-1], values[::-1], color='#3498db', edgecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{val}', va='center', fontsize=9,
        )
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _score_histogram(scores: list, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    _, bins, patches = ax.hist(scores, bins=20, color='#3498db', edgecolor='white')
    for patch, left in zip(patches, bins[:-1]):
        if left < 33:
            patch.set_facecolor('#27ae60')
        elif left < 55:
            patch.set_facecolor('#f39c12')
        elif left < 75:
            patch.set_facecolor('#e67e22')
        else:
            patch.set_facecolor('#c0392b')
    ax.set_xlabel('Addiction Risk Score')
    ax.set_ylabel('Number of Students')
    ax.set_title('Class Addiction Score Distribution', fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _dept_risk_heatmap(results_df: pd.DataFrame, save_path: str) -> bool:
    """Department vs risk level heatmap. Returns False if insufficient data."""
    if 'department' not in results_df.columns:
        return False
    pivot = pd.crosstab(
        results_df['department'],
        results_df['predicted_risk'],
    ).reindex(columns=CLASS_NAMES, fill_value=0)

    if pivot.empty:
        return False

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    ax.set_title('Department Risk Heatmap', fontsize=11, fontweight='bold')
    ax.set_xlabel('Risk Level')
    ax.set_ylabel('Department')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def _semester_risk_chart(results_df: pd.DataFrame, save_path: str) -> bool:
    """Stacked bar: semester vs risk distribution."""
    if 'semester' not in results_df.columns:
        return False

    pivot = pd.crosstab(
        results_df['semester'].astype(str),
        results_df['predicted_risk'],
    ).reindex(columns=CLASS_NAMES, fill_value=0)

    if pivot.empty:
        return False

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(pivot))
    for level in CLASS_NAMES:
        if level in pivot.columns:
            vals = pivot[level].values
            ax.bar(pivot.index, vals, bottom=bottom, label=level,
                   color=RISK_COLORS_HEX[level][0], edgecolor='white')
            bottom += vals

    ax.set_xlabel('Semester')
    ax.set_ylabel('Students')
    ax.set_title('Semester-wise Risk Distribution', fontsize=11, fontweight='bold')
    ax.legend(title='Risk', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def _risk_gauge_image(score: float, save_path: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 1.0))
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 100, 0, 1])
    ax.axvline(score, color='black', linewidth=2)
    ax.text(score, 0.5, f' {score}', va='center', fontsize=9, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    return save_path


# ── Class Report PDF ──────────────────────────────────────────────────────────

class ClassReportPDF(FPDF):
    def header(self) -> None:
        self.set_fill_color(30, 90, 160)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 13)
        self.cell(
            0, 12,
            safe_pdf_text('  Class Gadget Addiction Risk Report - Teacher Dashboard'),
            ln=True, fill=True,
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
                f'E-Gadget Addiction Predictor · KLE College · '
                f'{datetime.date.today()} · Page {self.page_no()}'
            ),
            align='C',
        )


def generate_class_report(
    summary: dict,
    results_df: pd.DataFrame,
    alerts: list,
    reports_dir: str,
) -> str:
    """Generate a comprehensive PDF class report for the teacher."""
    logger.info('Generating class report PDF (%d students)', summary.get('total_students', 0))
    os.makedirs(reports_dir, exist_ok=True)

    pie_path = os.path.join(reports_dir, '_tmp_pie.png')
    hist_path = os.path.join(reports_dir, '_tmp_hist.png')
    bar_path = os.path.join(reports_dir, '_tmp_bar.png')
    heatmap_path = os.path.join(reports_dir, '_tmp_heatmap.png')
    sem_path = os.path.join(reports_dir, '_tmp_semester.png')
    temp_files = [pie_path, hist_path, bar_path, heatmap_path, sem_path]

    _pie_chart(summary['risk_distribution'], 'Risk Level Distribution', pie_path)
    _score_histogram(results_df['addiction_score'].tolist(), hist_path)

    if summary.get('department_avg_score'):
        _bar_chart(
            summary['department_avg_score'],
            'Avg Addiction Score', 'Avg Score by Department', bar_path,
        )

    has_heatmap = _dept_risk_heatmap(results_df, heatmap_path)
    has_semester = _semester_risk_chart(results_df, sem_path)

    pdf = ClassReportPDF()
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(
        0, 7,
        safe_pdf_text(f"Report Date   : {datetime.date.today().strftime('%B %d, %Y')}"),
        ln=True,
    )
    pdf.cell(0, 7, safe_pdf_text(f"Total Students: {summary['total_students']}"), ln=True)
    pdf.set_font('Helvetica', '', 11)
    pdf.ln(3)

    # Key stats
    pdf.set_fill_color(240, 242, 246)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, safe_pdf_text('Key Statistics'), ln=True, fill=True)
    pdf.set_font('Helvetica', '', 10)

    stats = [
        ('Average Addiction Risk Score', f"{summary['avg_addiction_score']} / 100"),
        ('Most Common Risk Level', summary['most_common_risk']),
        ('Students Needing Attention', f"{summary['high_severe_count']} (High + Severe)"),
        ('Average Screen Time', f"{summary.get('avg_screen_time', '-')} hrs/day"),
        ('Average Sleep Hours', f"{summary.get('avg_sleep_hours', '-')} hrs/night"),
        ('Average GPA', f"{summary.get('avg_gpa', '-')}"),
        ('Average Stress Level', f"{summary.get('avg_stress', '-')} / 10"),
    ]
    for label, value in stats:
        pdf.cell(90, 7, safe_pdf_text(f'  {label}'), border=0)
        pdf.cell(0, 7, safe_pdf_text(f': {value}'), ln=True)
    pdf.ln(4)

    # Alert summary
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, safe_pdf_text('Alert Summary'), ln=True, fill=True)
    pdf.set_font('Helvetica', '', 10)
    high_count = sum(1 for a in alerts if a.get('risk') == 'High')
    severe_count = sum(1 for a in alerts if a.get('risk') == 'Severe')
    pdf.cell(0, 7, safe_pdf_text(f'  Total Alerts: {len(alerts)}'), ln=True)
    pdf.cell(0, 7, safe_pdf_text(f'  High Risk: {high_count}  |  Severe Risk: {severe_count}'), ln=True)
    pdf.ln(3)

    # Risk distribution table
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, safe_pdf_text('Risk Level Distribution'), ln=True, fill=True)
    pdf.set_font('Helvetica', '', 10)
    for level in CLASS_NAMES:
        count = summary['risk_distribution'].get(level, 0)
        pct = summary['risk_percentages'].get(level, 0)
        r, g, b = RISK_COLORS_HEX[level][1]
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(35, 7, safe_pdf_text(f'  {level}'), fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        pdf.cell(0, 7, safe_pdf_text(f'  {count} students  ({pct}%)'), ln=True)
    pdf.ln(4)

    # Charts
    pdf.image(pie_path, x=10, w=85)
    if os.path.isfile(hist_path):
        pdf.image(hist_path, x=105, y=pdf.get_y() - 60, w=95)
    pdf.ln(5)

    if os.path.isfile(bar_path) and summary.get('department_avg_score'):
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, safe_pdf_text('Department-wise Average Score'), ln=True)
        pdf.image(bar_path, x=10, w=180)
        pdf.ln(4)

    if has_heatmap:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, safe_pdf_text('Department Risk Heatmap'), ln=True)
        pdf.image(heatmap_path, x=10, w=180)
        pdf.ln(4)

    if has_semester:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, safe_pdf_text('Semester-wise Risk Distribution'), ln=True)
        pdf.image(sem_path, x=10, w=180)
        pdf.ln(4)

    # Model metadata
    metadata = load_model_metadata()
    if metadata:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, safe_pdf_text('Model Metadata'), ln=True, fill=True)
        pdf.set_font('Helvetica', '', 9)
        for key in ('algorithm', 'accuracy', 'precision', 'recall', 'f1_score', 'date_trained'):
            if metadata.get(key):
                pdf.cell(
                    0, 6,
                    safe_pdf_text(f"  {key.replace('_', ' ').title()}: {metadata[key]}"),
                    ln=True,
                )
        pdf.ln(3)

    # Alerts detail
    if alerts:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_fill_color(192, 57, 43)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(
            0, 9,
            safe_pdf_text(
                f'  WARNING  Students Requiring Immediate Attention ({len(alerts)})'
            ),
            ln=True, fill=True,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        for idx, alert in enumerate(alerts, 1):
            risk = alert['risk']
            r, g, b = RISK_COLORS_HEX[risk][1]
            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Helvetica', 'B', 10)
            alert_text = (
                f"  {idx}. {alert['name']}  |  ID: {alert['student_id']}"
                f"  |  Dept: {alert['department']}"
                f"  |  Risk: {alert['risk']}  |  Score: {alert['score']}/100"
            )
            pdf.cell(0, 7, safe_pdf_text(alert_text), ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', '', 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0, 6,
                safe_pdf_text(f"     Action: {alert.get('alert_message', '')}"),
            )
            tips = get_recommendations(CLASS_NAMES.index(alert['risk']), verbose=False)[:2]
            for tip in tips:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 6, safe_pdf_text(f"     {tip}"))
            pdf.ln(2)

    # Top 10 at-risk table
    if len(results_df) > 0:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, safe_pdf_text('Top 10 Highest Risk Students'), ln=True)
        pdf.set_font('Helvetica', '', 9)

        top10 = results_df.nlargest(10, 'addiction_score')
        headers = ['Name', 'ID', 'Dept', 'Risk', 'Score']
        widths = [50, 25, 35, 25, 20]

        pdf.set_fill_color(30, 90, 160)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(headers, widths):
            pdf.cell(w, 7, safe_pdf_text(f' {h}'), fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        for _, row in top10.iterrows():
            risk = row.get('predicted_risk', '-')
            r, g, b = RISK_COLORS_HEX.get(risk, ('', (255, 255, 255)))[1]
            pdf.set_fill_color(r, g, b)
            fill = risk in ('High', 'Severe')
            if fill:
                pdf.set_text_color(255, 255, 255)
            row_data = [
                str(row.get('student_name', '-'))[:20],
                str(row.get('student_id', '-'))[:12],
                str(row.get('department', '-'))[:14],
                risk,
                str(row.get('addiction_score', '-')),
            ]
            for val, w in zip(row_data, widths):
                pdf.cell(w, 6, safe_pdf_text(f' {val}'), fill=fill)
            pdf.set_text_color(0, 0, 0)
            pdf.ln()

    for p in temp_files:
        if os.path.isfile(p):
            os.remove(p)

    out_path = os.path.join(reports_dir, f'class_report_{datetime.date.today()}.pdf')
    pdf.output(out_path)
    logger.info('Class PDF saved -> %s', out_path)
    return out_path


# ── Individual Student Cards PDF ──────────────────────────────────────────────

class StudentCardPDF(FPDF):
    def header(self) -> None:
        self.set_fill_color(44, 62, 80)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 11)
        self.cell(
            0, 10,
            safe_pdf_text('  E-Gadget Addiction · Individual Student Report'),
            ln=True, fill=True,
        )
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 6,
            safe_pdf_text(
                f'KLE College of Engineering · Confidential · Page {self.page_no()}'
            ),
            align='C',
        )


def generate_student_cards(
    results_df: pd.DataFrame,
    reports_dir: str,
    max_students: int = 200,
) -> str:
    """
    One page per student with risk, score, gauge, and recommendations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Batch prediction results.
    reports_dir : str
        Output directory.
    max_students : int
        Cap to prevent huge PDF files.
    """
    logger.info('Generating student cards PDF (max %d)', max_students)
    os.makedirs(reports_dir, exist_ok=True)
    pdf = StudentCardPDF()
    subset = results_df.head(max_students)
    gauge_tmp: Optional[str] = None

    for idx, row in subset.iterrows():
        pdf.add_page()

        name = str(row.get('student_name', f'Student {idx + 1}'))
        sid = str(row.get('student_id', '-'))
        dept = str(row.get('department', '-'))
        sem = str(row.get('semester', '-'))
        risk = str(row.get('predicted_risk', '-'))
        score = row.get('addiction_score', 0)
        conf = row.get('confidence_pct', 0)
        ridx = int(row.get('risk_index', 0))

        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 8, safe_pdf_text(name), ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(60, 6, safe_pdf_text(f'Student ID : {sid}'))
        pdf.cell(0, 6, safe_pdf_text(f'Department : {dept}  |  Semester : {sem}'), ln=True)
        pdf.ln(3)

        r, g, b = RISK_COLORS_HEX.get(risk, ('', (200, 200, 200)))[1]
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(
            0, 10,
            safe_pdf_text(
                f'  Risk Level: {risk}    |    Addiction Score: {score}/100'
                f'    |    Confidence: {conf}%'
            ),
            ln=True, fill=True,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        # Risk gauge
        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='_gauge.png')
            gauge_tmp = _risk_gauge_image(float(score), tmpf.name)
            pdf.image(gauge_tmp, x=10, w=180)
            pdf.ln(3)
        except Exception as exc:
            logger.warning('Student card gauge failed: %s', exc)

        feature_labels = {
            'daily_screen_time_hours': 'Screen Time (hrs/day)',
            'num_social_media_platforms': 'Social Media Platforms',
            'late_night_usage': 'Late Night Usage',
            'gpa': 'GPA',
            'missed_classes_per_month': 'Missed Classes/Month',
            'sleep_hours': 'Sleep Hours/Night',
            'sleep_disturbances': 'Sleep Disturbances',
            'physical_activity_hours': 'Physical Activity (hrs/week)',
            'stress_level': 'Stress Level (1-10)',
            'social_interaction_quality': 'Social Quality (1-5)',
        }
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, safe_pdf_text('Student Profile:'), ln=True)
        pdf.set_font('Helvetica', '', 9)

        col_w = 90
        items = [(v, row.get(k, '-')) for k, v in feature_labels.items() if k in row]
        for i in range(0, len(items), 2):
            label1, val1 = items[i]
            val1_str = (
                'Yes' if val1 == 1 else 'No'
                if label1 in ('Late Night Usage', 'Sleep Disturbances')
                else str(val1)
            )
            pdf.cell(col_w, 6, safe_pdf_text(f'  {label1}: {val1_str}'))
            if i + 1 < len(items):
                label2, val2 = items[i + 1]
                val2_str = (
                    'Yes' if val2 == 1 else 'No'
                    if label2 in ('Late Night Usage', 'Sleep Disturbances')
                    else str(val2)
                )
                pdf.cell(col_w, 6, safe_pdf_text(f'  {label2}: {val2_str}'))
            pdf.ln()
        pdf.ln(3)

        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, safe_pdf_text('Personalised Recommendations:'), ln=True)
        pdf.set_font('Helvetica', '', 9)
        tips = get_recommendations(ridx, verbose=False)
        for tip in tips:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 6, safe_pdf_text(f'  {tip}'))

        pdf.ln(3)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(
            0, 5,
            safe_pdf_text(
                'This report is AI-generated and for informational purposes only. '
                'Please consult a professional for clinical assessment.'
            ),
        )
        pdf.set_text_color(0, 0, 0)

        if gauge_tmp and os.path.isfile(gauge_tmp):
            try:
                os.remove(gauge_tmp)
            except OSError:
                pass
            gauge_tmp = None

    out_path = os.path.join(reports_dir, f'student_cards_{datetime.date.today()}.pdf')
    pdf.output(out_path)
    logger.info('Student cards PDF saved -> %s', out_path)
    return out_path
