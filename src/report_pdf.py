from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


@dataclass(frozen=True)
class ReportPaths:
    pdf_path: Path


def _money(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _kpi_cards(total_revenue: float, orders: int, aov: float) -> Table:
    data = [
        [
            Paragraph("<b>Total Revenue</b><br/>" + _money(total_revenue), _STYLES["Card"]),
            Paragraph("<b>Orders</b><br/>" + _safe_str(orders), _STYLES["Card"]),
            Paragraph("<b>Avg Order Value</b><br/>" + _money(aov), _STYLES["Card"]),
        ]
    ]
    t = Table(data, colWidths=[6.2 * cm, 6.2 * cm, 6.2 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return t


def _cleaning_summary_table(summary: Dict[str, Any]) -> Table:
    rows = [
        ["Rows before", summary.get("rows_before")],
        ["Rows after", summary.get("rows_after")],
        ["Empty rows dropped", summary.get("empty_rows_dropped")],
        ["Duplicates removed", summary.get("duplicates_removed")],
        ["Invalid dates", summary.get("invalid_dates")],
        ["Invalid amounts", summary.get("invalid_amounts")],
    ]
    t = Table(rows, colWidths=[7.5 * cm, 3.5 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _image_or_note(path: Path, width_cm: float, height_cm: float, note: str) -> Any:
    path = Path(path)
    if path.exists():
        img = Image(str(path), width=width_cm * cm, height=height_cm * cm)
        return img
    return Paragraph(f"<i>{note}</i>", _STYLES["BodyGrey"])


def _df_to_table(df: pd.DataFrame, max_rows: int = 10) -> Table:
    if df is None or df.empty:
        return Table([["No data available."]], colWidths=[17.9 * cm])

    df_show = df.copy().head(max_rows)

    # Make it look nice
    for col in df_show.columns:
        df_show[col] = df_show[col].apply(lambda v: _safe_str(v))

    data = [list(df_show.columns)] + df_show.values.tolist()

    # Set widths nicely for typical columns
    col_widths = []
    for col in df_show.columns:
        if col.lower() in {"product"}:
            col_widths.append(9.0 * cm)
        elif col.lower() in {"revenue"}:
            col_widths.append(4.0 * cm)
        else:
            col_widths.append(3.5 * cm)

    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


_STYLES = {}
def _init_styles():
    global _STYLES
    styles = getSampleStyleSheet()
    _STYLES = {
        "Title": ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, spaceAfter=10),
        "Sub": ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, textColor=colors.grey, spaceAfter=8),
        "H2": ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceAfter=6),
        "Body": ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14),
        "BodyGrey": ParagraphStyle("BodyGrey", parent=styles["Normal"], fontSize=10, textColor=colors.grey, leading=14),
        "Card": ParagraphStyle("Card", parent=styles["Normal"], fontSize=10, leading=14),
    }


def generate_weekly_pdf_report(
    pdf_path: Path,
    kpi,  # KPIResult from src/kpi.py
    cleaning_summary: dict,
    chart_paths,  # ChartPaths from src/charts.py
) -> ReportPaths:
    _init_styles()

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Weekly Sales Report",
    )

    story = []

    # Title area
    story.append(Paragraph("Weekly Sales Report", _STYLES["Title"]))
    story.append(Paragraph("Generated by Auto Report Generator", _STYLES["Sub"]))

    # Date range (based on daily trend, if available)
    date_range = "Date range: N/A"
    try:
        if kpi.daily_revenue is not None and len(kpi.daily_revenue) > 0 and "day" in kpi.daily_revenue.columns:
            days = pd.to_datetime(kpi.daily_revenue["day"], errors="coerce").dropna()
            if len(days) > 0:
                date_range = f"Date range: {days.min().date()} → {days.max().date()}"
    except Exception:
        pass

    story.append(Paragraph(date_range, _STYLES["BodyGrey"]))
    story.append(Spacer(1, 10))

    # KPI cards
    story.append(_kpi_cards(kpi.total_revenue, kpi.orders_count, kpi.average_order_value))
    story.append(Spacer(1, 14))

    # Cleaning summary
    story.append(Paragraph("Cleaning Summary", _STYLES["H2"]))
    story.append(_cleaning_summary_table(cleaning_summary))
    story.append(Spacer(1, 10))

    # Notes
    if getattr(kpi, "notes", None):
        story.append(Paragraph("Notes", _STYLES["H2"]))
        notes_text = "<br/>".join([f"• {_safe_str(n)}" for n in kpi.notes])
        story.append(Paragraph(notes_text, _STYLES["Body"]))
        story.append(Spacer(1, 10))

    # Chart: Daily trend
    story.append(Paragraph("Daily Revenue Trend", _STYLES["H2"]))
    story.append(_image_or_note(Path(chart_paths.daily_revenue), 17.9, 8.5, "Daily revenue chart not found."))
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Details", _STYLES["Title"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Revenue by Category", _STYLES["H2"]))
    story.append(_image_or_note(Path(chart_paths.revenue_by_category), 17.9, 8.5, "Category chart not found."))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Top Products", _STYLES["H2"]))
    top_df = getattr(kpi, "top_products", pd.DataFrame())
    if isinstance(top_df, pd.DataFrame) and not top_df.empty and "revenue" in top_df.columns:
        df_show = top_df.copy()
        df_show["revenue"] = df_show["revenue"].apply(lambda v: _money(v))
    else:
        df_show = top_df
    story.append(_df_to_table(df_show, max_rows=10))

    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return ReportPaths(pdf_path=pdf_path)
