"""
Export docs/PROJECT_DESCRIPTION.md to docs/PROJECT_DESCRIPTION.pdf
using ReportLab with clean styling and layout.
"""
import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def markdown_to_pdf():
    md_path = Path("docs/PROJECT_DESCRIPTION.md")
    pdf_path = Path("docs/PROJECT_DESCRIPTION.pdf")
    
    if not md_path.exists():
        print(f"[!] {md_path} not found.")
        return

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#0f172a") # Slate 900
    ACCENT = colors.HexColor("#0284c7")  # Sky 600
    TEXT_COLOR = colors.HexColor("#334155") # Slate 700
    BG_LIGHT = colors.HexColor("#f8fafc") # Slate 50
    BORDER_COLOR = colors.HexColor("#cbd5e1") # Slate 300

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=ACCENT,
        spaceAfter=10,
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_COLOR,
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_COLOR,
        leftIndent=12,
        spaceAfter=3,
    )

    table_cell_style = ParagraphStyle(
        'DocTableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=TEXT_COLOR,
    )

    table_header_style = ParagraphStyle(
        'DocTableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=PRIMARY,
    )

    story = []

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table and table_rows:
                t = Table(table_rows, colWidths=[120, 180, 204])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(t)
                story.append(Spacer(1, 6))
                in_table = False
                table_rows = []
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(set(c).issubset({'-', ':', ' '}) for c in cells):
                continue
            in_table = True
            is_header = (len(table_rows) == 0)
            row_paras = []
            for cell in cells:
                clean_cell = cell.replace("**", "").replace("*", "")
                st = table_header_style if is_header else table_cell_style
                row_paras.append(Paragraph(clean_cell, st))
            table_rows.append(row_paras)
            continue
        elif in_table and table_rows:
            t = Table(table_rows, colWidths=[120, 180, 204])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))
            in_table = False
            table_rows = []

        if stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], title_style))
        elif stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], subtitle_style))
            story.append(HRFlowable(width="100%", thickness=1, color=BORDER_COLOR, spaceAfter=6, spaceBefore=2))
        elif stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], h2_style))
        elif stripped.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceAfter=6, spaceBefore=3))
        elif stripped.startswith("```"):
            continue
        elif stripped.startswith("- ") or stripped.startswith("* "):
            clean_text = stripped[2:].replace("**", "")
            story.append(Paragraph(f"&bull; {clean_text}", bullet_style))
        elif stripped[0].isdigit() and len(stripped) > 2 and stripped[1:3] in [". ", ") "]:
            clean_text = stripped[3:].replace("**", "")
            story.append(Paragraph(f"{stripped[:2]} {clean_text}", bullet_style))
        else:
            clean_text = stripped.replace("**", "")
            story.append(Paragraph(clean_text, body_style))

    if in_table and table_rows:
        t = Table(table_rows, colWidths=[120, 180, 204])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t)

    doc.build(story)
    print(f"[+] PDF successfully created at {pdf_path} ({pdf_path.stat().st_size} bytes)")

if __name__ == "__main__":
    markdown_to_pdf()
