from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


class CleaningError(ValueError):
    """Raised when input data cannot be cleaned safely (missing required columns, empty file, etc.)."""


@dataclass(frozen=True)
class CleaningConfig:
    # Policies
    invalid_date_policy: str = "drop"      # "drop" | "keep"
    invalid_amount_policy: str = "keep_nan" # "drop" | "zero" | "keep_nan"

    # Defaults for missing text fields
    default_customer: str = "Anonymous"
    default_product: str = "Unknown"
    default_category: str = "Unknown"
    default_payment_status: str = "Unknown"
    default_city: str = "Unknown"
    default_country: str = "Unknown"


@dataclass
class CleaningSummary:
    rows_before: int
    rows_after: int
    empty_rows_dropped: int
    duplicates_removed: int
    invalid_dates: int
    invalid_amounts: int
    filled_customer: int
    filled_product: int
    filled_category: int
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CANONICAL_COLS = [
    "order_date",
    "customer_name",
    "product",
    "category",
    "amount",
    "payment_status",
    "city",
    "country",
]

# Normalized synonyms → canonical name
_SYNONYMS: Dict[str, str] = {
    # order_date
    "order_date": "order_date",
    "orderdate": "order_date",
    "date": "order_date",
    "created_at": "order_date",
    "createdat": "order_date",
    "order_date_": "order_date",
    "orderdate_": "order_date",
    "order_date__": "order_date",
    "order_date__ ": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    "order_date__": "order_date",
    # common variants used in our sample
    "order_date": "order_date",
    "order_date__": "order_date",
    "order_date_": "order_date",
    "order_date__ ": "order_date",
    "order_date__  ": "order_date",
    "order_date___": "order_date",
    "order_date___ ": "order_date",
    "order_date___  ": "order_date",
    "order_date____": "order_date",
    "order date": "order_date",
    "orderdate": "order_date",
    "order_date": "order_date",

    # customer_name
    "customer_name": "customer_name",
    "customer": "customer_name",
    "client": "customer_name",

    # product
    "product": "product",
    "product_name": "product",
    "item": "product",

    # category
    "category": "category",
    "category_name": "category",
    "cat": "category",

    # amount
    "amount": "amount",
    "total": "amount",
    "price": "amount",

    # payment_status
    "payment_status": "payment_status",
    "payment": "payment_status",
    "status": "payment_status",

    # city
    "city": "city",
    "city_name": "city",
    "town": "city",

    # country
    "country": "country",
    "country_name": "country",
}


def _normalize_col(name: Any) -> str:
    s = str(name).strip().lower()
    # Replace non-alphanumeric with underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _coalesce_columns(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Coalesce multiple columns (take first non-null across them)."""
    if len(cols) == 1:
        return df[cols[0]]
    tmp = df[cols].copy()
    # Prefer leftmost non-null
    return tmp.bfill(axis=1).iloc[:, 0]


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Robust-ish date parsing for messy formats.
    Strategy:
      1) parse with dayfirst=False
      2) re-parse remaining NaT with dayfirst=True
    """
    s = series.copy()
    parsed_1 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    missing_mask = parsed_1.isna()
    if missing_mask.any():
        parsed_2 = pd.to_datetime(s[missing_mask], errors="coerce", dayfirst=True)
        parsed_1.loc[missing_mask] = parsed_2
    return parsed_1


_BAD_AMOUNT_TOKENS = {"", "n/a", "na", "none", "nan", "—", "-", "free"}


def _parse_amount_value(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    if isinstance(x, float) and pd.isna(x):
        return None

    s = str(x).strip()
    if not s:
        return None

    low = s.lower().strip()
    if low in _BAD_AMOUNT_TOKENS:
        return None

    negative = False
    if "(" in s and ")" in s:
        negative = True
        s = s.replace("(", "").replace(")", "")
    s = s.strip()

    # Keep only digits, dot, comma, minus
    s = re.sub(r"[^\d\.,\-]+", "", s)

    # Handle European format like 1.234,56
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        # Only commas: could be thousands or decimal comma
        if "," in s and "." not in s:
            parts = s.split(",")
            if len(parts[-1]) == 2:  # likely decimal comma
                s = "".join(parts[:-1]) + "." + parts[-1]
            else:
                s = s.replace(",", "")

    # Clean extra minus signs
    s = s.replace("--", "-")
    if s.count("-") > 1:
        s = s.replace("-", "")
    if negative and not s.startswith("-"):
        s = "-" + s

    try:
        return float(s)
    except Exception:
        return None


def _clean_text_series(series: pd.Series, default: str) -> Tuple[pd.Series, int]:
    s = series.copy()
    s = s.astype("string")
    s = s.str.strip()
    s = s.replace({"": pd.NA})
    missing = int(s.isna().sum())
    s = s.fillna(default)
    return s, missing


def _normalize_payment_status(series: pd.Series, default: str) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    s = s.replace({"": pd.NA})

    paid = {"paid", "yes", "y", "true", "1"}
    unpaid = {"unpaid", "no", "n", "false", "0", "pending"}

    def norm(v: Any) -> str:
        if v is None or pd.isna(v):
            return default
        vv = str(v).strip().lower()
        if vv in paid:
            return "Paid"
        if vv in unpaid:
            return "Unpaid"
        return default

    return s.apply(norm)


def clean_sales_dataframe(df_raw: pd.DataFrame, cfg: CleaningConfig = CleaningConfig()) -> Tuple[pd.DataFrame, CleaningSummary]:
    if df_raw is None or not isinstance(df_raw, pd.DataFrame):
        raise CleaningError("Input is not a valid table (DataFrame).")

    rows_before = int(len(df_raw))
    if rows_before == 0:
        raise CleaningError("The uploaded file has no rows.")

    warnings: List[str] = []

    # Drop fully empty rows early (all NaN)
    df = df_raw.copy()
    empty_before = int(df.isna().all(axis=1).sum())
    df = df.dropna(how="all")
    empty_rows_dropped = empty_before

    # Normalize and map columns to canonical
    normalized = [_normalize_col(c) for c in df.columns]
    norm_to_original: Dict[str, List[str]] = {}
    for orig, norm in zip(df.columns, normalized):
        norm_to_original.setdefault(norm, []).append(orig)

    # Build canonical -> list of original columns that map to it
    canonical_sources: Dict[str, List[str]] = {c: [] for c in _CANONICAL_COLS}
    passthrough_cols: List[str] = []

    for norm, originals in norm_to_original.items():
        canonical = _SYNONYMS.get(norm)
        if canonical is None:
            passthrough_cols.extend(originals)
        else:
            canonical_sources[canonical].extend(originals)

    # Required fields
    if not canonical_sources["order_date"]:
        raise CleaningError("Missing required column: Order Date (e.g., 'Order Date' / 'date' / 'orderDate').")
    if not canonical_sources["amount"]:
        raise CleaningError("Missing required column: Amount (e.g., 'Amount' / 'total' / 'price').")

    # Build canonical dataframe (coalesce duplicates)
    out = pd.DataFrame()

    for canonical in _CANONICAL_COLS:
        sources = canonical_sources[canonical]
        if not sources:
            out[canonical] = pd.NA
            continue
        out[canonical] = _coalesce_columns(df, sources)

        if len(sources) > 1:
            warnings.append(f"Merged {len(sources)} columns into '{canonical}': {sources}")

    # Keep extra columns (optional)
    for c in passthrough_cols:
        if c not in out.columns:
            out[c] = df[c]

    # Clean dates
    out["order_date"] = _parse_dates(out["order_date"])
    invalid_dates = int(out["order_date"].isna().sum())
    if invalid_dates > 0 and cfg.invalid_date_policy == "drop":
        out = out.dropna(subset=["order_date"])

    # Clean amounts
    parsed_amount = out["amount"].apply(_parse_amount_value)
    out["amount"] = pd.to_numeric(parsed_amount, errors="coerce")
    invalid_amounts = int(out["amount"].isna().sum())
    if invalid_amounts > 0:
        if cfg.invalid_amount_policy == "drop":
            out = out.dropna(subset=["amount"])
        elif cfg.invalid_amount_policy == "zero":
            out["amount"] = out["amount"].fillna(0.0)
        elif cfg.invalid_amount_policy == "keep_nan":
            pass
        else:
            raise CleaningError(f"Invalid invalid_amount_policy: {cfg.invalid_amount_policy}")

    # Fill/clean text fields
    out["customer_name"], filled_customer = _clean_text_series(out["customer_name"], cfg.default_customer)
    out["product"], filled_product = _clean_text_series(out["product"], cfg.default_product)
    out["category"], filled_category = _clean_text_series(out["category"], cfg.default_category)

    out["city"], _ = _clean_text_series(out["city"], cfg.default_city)
    out["country"], _ = _clean_text_series(out["country"], cfg.default_country)

    out["payment_status"] = _normalize_payment_status(out["payment_status"], cfg.default_payment_status)

    # Drop duplicates (after normalization)
    before_dupes = int(len(out))
    out = out.drop_duplicates()
    duplicates_removed = before_dupes - int(len(out))

    rows_after = int(len(out))

    summary = CleaningSummary(
        rows_before=rows_before,
        rows_after=rows_after,
        empty_rows_dropped=empty_rows_dropped,
        duplicates_removed=duplicates_removed,
        invalid_dates=invalid_dates,
        invalid_amounts=invalid_amounts,
        filled_customer=filled_customer,
        filled_product=filled_product,
        filled_category=filled_category,
        warnings=warnings,
    )

    # Order columns nicely: canonical first, then any extras
    extras = [c for c in out.columns if c not in _CANONICAL_COLS]
    out = out[_CANONICAL_COLS + extras]

    return out.reset_index(drop=True), summary
