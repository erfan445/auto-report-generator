from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import warnings



@dataclass
class GenericAnalysisResult:
    rows: int
    cols: int
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    date_col: Optional[str]
    primary_numeric_col: Optional[str]
    numeric_summary: pd.DataFrame
    top_categories: Dict[str, pd.DataFrame]
    daily_trend: pd.DataFrame
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "missing_cells": self.missing_cells,
            "missing_pct": self.missing_pct,
            "duplicate_rows": self.duplicate_rows,
            "date_col": self.date_col,
            "primary_numeric_col": self.primary_numeric_col,
            "notes": self.notes,
        }


def _normalize_columns(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c2 = str(c).strip().lower()
        c2 = c2.replace("\n", " ").replace("\t", " ")
        c2 = "_".join(c2.split())  # collapse whitespace -> underscore
        c2 = "".join(ch for ch in c2 if ch.isalnum() or ch == "_")
        out.append(c2)
    # make unique
    seen = {}
    unique = []
    for c in out:
        if c not in seen:
            seen[c] = 0
            unique.append(c)
        else:
            seen[c] += 1
            unique.append(f"{c}_{seen[c]}")
    return unique


def basic_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = _normalize_columns([str(c) for c in out.columns])

    # drop fully empty rows
    out = out.dropna(how="all")

    # strip object cells
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("string").str.strip()

    return out


def infer_date_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None

    # Prefer columns containing 'date'/'time'
    candidates = [c for c in df.columns if any(k in c for k in ["date", "time", "day"])]
    candidates = candidates + [c for c in df.columns if c not in candidates]

    best_col = None
    best_success = 0.0

    for c in candidates[:15]:  # limit scanning
        s = df[c]
        if s.dtype.kind in "mM":
            return c

        # Silence pandas warnings for mixed date formats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                parsed = pd.to_datetime(s, errors="coerce", utc=False, format="mixed")
            except TypeError:
                parsed = pd.to_datetime(s, errors="coerce", utc=False)

        success = float(parsed.notna().mean())

        # Pick the best date-like column even if only a portion parses
        if success > best_success:
            best_success = success
            best_col = c

    # Only accept if at least 20% of rows parse as dates (messy data friendly)
    if best_success >= 0.20:
        return best_col
    return None


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []

    numeric_cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
            continue

        # try coercion (handles "1,234", "$5.20", "(2.9)", etc.)
        s2 = (
            s.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("₺", "", regex=False)
            .str.replace("tl", "", case=False, regex=False)
            .str.replace("usd", "", case=False, regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
        )
        coerced = pd.to_numeric(s2, errors="coerce")
        if float(coerced.notna().mean()) >= 0.7:  # 70% numeric-like
            df[c] = coerced
            numeric_cols.append(c)

    return numeric_cols


def pick_primary_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
    if not numeric_cols:
        return None

    # Prefer names that look like key metrics
    priority = ["amount", "revenue", "total", "price", "views", "watch_time", "impressions", "ctr"]
    for p in priority:
        for c in numeric_cols:
            if p in c:
                return c

    # Otherwise choose the one with most non-null values
    best = max(numeric_cols, key=lambda c: df[c].notna().sum())
    return best


def compute_generic_analysis(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, GenericAnalysisResult]:
    df = basic_clean_dataframe(df_raw)

    if df.empty:
        res = GenericAnalysisResult(
            rows=0,
            cols=0,
            missing_cells=0,
            missing_pct=0.0,
            duplicate_rows=0,
            date_col=None,
            primary_numeric_col=None,
            numeric_summary=pd.DataFrame(),
            top_categories={},
            daily_trend=pd.DataFrame(),
            notes=["Empty file after cleaning (no rows)."],
        )
        return df, res

    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()

    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    missing_pct = float(missing_cells / max(rows * cols, 1))

    # infer types
    date_col = infer_date_column(df)
    numeric_cols = infer_numeric_columns(df)
    primary_num = pick_primary_numeric(df, numeric_cols)

    notes: List[str] = []
    if duplicate_rows > 0:
        notes.append(f"Removed {duplicate_rows} duplicate rows.")
    if missing_pct > 0.1:
        notes.append(f"High missingness detected: {missing_pct:.1%} of cells are missing.")

    # numeric summary
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().T
    else:
        numeric_summary = pd.DataFrame()
        notes.append("No numeric columns detected (or numeric-like).")

    # top categories for up to 3 text columns
    top_categories: Dict[str, pd.DataFrame] = {}
    text_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("string")]
    for c in text_cols[:3]:
        vc = df[c].fillna("Unknown").astype("string").value_counts().head(10).reset_index()
        vc.columns = [c, "count"]
        top_categories[c] = vc

    # daily trend
    daily_trend = pd.DataFrame()
    if date_col and primary_num:
        d = pd.to_datetime(df[date_col], errors="coerce")
        tmp = df.copy()
        tmp["_date"] = d.dt.date
        tmp = tmp.dropna(subset=["_date"])
        if not tmp.empty:
            daily_trend = tmp.groupby("_date", as_index=False)[primary_num].sum()
            daily_trend.columns = ["day", "value"]
        else:
            notes.append("Date column found, but no valid parsed dates for trend chart.")
    else:
        notes.append("Trend chart skipped (missing date column or numeric metric).")

    res = GenericAnalysisResult(
        rows=rows,
        cols=cols,
        missing_cells=missing_cells,
        missing_pct=missing_pct,
        duplicate_rows=duplicate_rows,
        date_col=date_col,
        primary_numeric_col=primary_num,
        numeric_summary=numeric_summary,
        top_categories=top_categories,
        daily_trend=daily_trend,
        notes=notes,
    )

    return df, res
