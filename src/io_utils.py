from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_cleaned_excel(df_clean: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path)
    df_clean.to_excel(out_path, index=False)
    return out_path


def write_json(data: Dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def save_all_outputs(
    df_clean: pd.DataFrame,
    cleaning_summary: Dict[str, Any],
    kpi_result,  # KPIResult from src/kpi.py
    pdf_path: Path,
    excel_path: Path = Path("output/cleaned_data.xlsx"),
    json_path: Path = Path("output/kpi_summary.json"),
) -> Dict[str, Path]:
    """
    Centralized output writer (enterprise-style).
    """
    excel_path = write_cleaned_excel(df_clean, excel_path)

    kpi_payload = {
        "kpis": kpi_result.to_dict() if hasattr(kpi_result, "to_dict") else {},
        "cleaning_summary": cleaning_summary,
        "pdf_path": str(pdf_path),
        "excel_path": str(excel_path),
    }
    json_path = write_json(kpi_payload, json_path)

    return {
        "excel": excel_path,
        "pdf": Path(pdf_path),
        "json": json_path,
    }
