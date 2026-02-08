from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Config:
    rows: int = 80
    duplicates: int = 5
    empty_rows: int = 3
    seed: int = 7
    out_path: Path = Path("data/sample_raw_sales.xlsx")


DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%b-%Y",
]

AMOUNT_FORMATTERS = [
    lambda v: f"${v:,.2f}",
    lambda v: f"{v:,.0f} TL",
    lambda v: f"€{v:,.2f}",
    lambda v: f"{v:,}",
    lambda v: str(v),
]


def ensure_dependencies() -> None:
    """Fail fast with a clear message if Excel engine is missing."""
    try:
        import openpyxl  # noqa: F401
    except Exception:
        print("❌ Missing dependency: openpyxl")
        print("Run: pip install openpyxl")
        sys.exit(1)


def ensure_output_dir(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)


def make_canonical_rows(cfg: Config) -> pd.DataFrame:
    """Create a stable dataset with consistent columns (source of truth)."""
    random.seed(cfg.seed)

    customers = ["Alice", "Bob", "Carmen", "Deniz", "Eren", "Fatima", "George", None, ""]
    products = ["Shampoo", "Laptop Stand", "Face Cream", "Protein Bar", "Headphones", None, ""]
    categories = ["Beauty", "Electronics", "Food", "Health", None, ""]
    statuses = ["Paid", "Unpaid", "paid", "UNPAID", None, ""]
    cities = ["Nicosia", "Famagusta", "Iskele", "Istanbul", "Tehran", None, ""]
    countries = ["Cyprus", "Turkey", "Iran", None, ""]

    base = datetime(2026, 1, 1)

    rows: list[dict[str, Any]] = []
    for _ in range(cfg.rows):
        d = base + timedelta(days=random.randint(0, 35))
        rows.append(
            {
                "order_date": d,
                "customer_name": random.choice(customers),
                "product": random.choice(products),
                "category": random.choice(categories),
                "amount": random.choice([120, 89.5, 1450, 19.99, 250, 999.9, 0, None]),
                "payment_status": random.choice(statuses),
                "city": random.choice(cities),
                "country": random.choice(countries),
            }
        )

    return pd.DataFrame(rows)


def mess_up_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Turn canonical data into realistic messy Excel/CSV style data."""
    random.seed(cfg.seed)

    messy = df.copy()

    # 1) Mess up dates (mixed formats + some invalids)
    def messy_date(d: Any) -> Any:
        if pd.isna(d):
            return None
        if random.random() < 0.04:
            return random.choice(["not-a-date", "32/13/2026", ""])  # invalid on purpose
        fmt = random.choice(DATE_FORMATS)
        return pd.to_datetime(d).to_pydatetime().strftime(fmt)

    messy["order_date"] = messy["order_date"].apply(messy_date)

    # 2) Mess up amounts (currency symbols, commas, text junk)
    def messy_amount(v: Any) -> Any:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return random.choice([None, "", "N/A"])
        if random.random() < 0.05:
            return random.choice(["N/A", "—", "free", ""])
        v_float = float(v)
        return random.choice(AMOUNT_FORMATTERS)(v_float)

    messy["amount"] = messy["amount"].apply(messy_amount)

    # 3) Inconsistent column naming (controlled mapping)
    col_variants = {
        "order_date": ["Order Date", "orderDate", "DATE", "Order date "],
        "customer_name": ["Customer Name", "customer", "Client", "Customer  "],
        "product": ["Product", "product_name", "Item"],
        "category": ["Category", "category_name", "CAT"],
        "amount": ["Amount", "price", "Total", "amount "],
        "payment_status": ["Payment Status", "payment", "Status"],
        "city": ["City", "city_name", "Town"],
        "country": ["Country", "country_name"],
    }

    rename_map: dict[str, str] = {}
    used_names: set[str] = set()
    for canonical_col, options in col_variants.items():
        # pick a variant that isn't already used (avoid duplicate column names)
        random.shuffle(options)
        picked = None
        for opt in options:
            if opt not in used_names:
                picked = opt
                used_names.add(opt)
                break
        if picked is None:
            picked = canonical_col  # fallback
        rename_map[canonical_col] = picked

    messy = messy.rename(columns=rename_map)

    # 4) Add duplicates (for de-dup cleaning)
    if cfg.duplicates > 0 and len(messy) > 0:
        dup_n = min(cfg.duplicates, len(messy))
        messy = pd.concat([messy, messy.iloc[:dup_n]], ignore_index=True)

    # 5) Add empty rows
    if cfg.empty_rows > 0:
        messy = pd.concat([messy, pd.DataFrame([{} for _ in range(cfg.empty_rows)])], ignore_index=True)

    # 6) Shuffle rows (more realistic)
    messy = messy.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    return messy


def main() -> None:
    ensure_dependencies()
    cfg = Config()

    try:
        ensure_output_dir(cfg.out_path)

        canonical = make_canonical_rows(cfg)
        messy = mess_up_dataframe(canonical, cfg)

        messy.to_excel(cfg.out_path, index=False)

        print("✅ Sample data generated successfully")
        print(f"   File: {cfg.out_path}")
        print(f"   Rows: {len(messy)} (includes duplicates + empty rows)")
        print("   Notes: mixed date formats, messy currency strings, missing values, invalid entries")
    except PermissionError:
        print(f"❌ Permission error: cannot write to {cfg.out_path}")
        print("Close the Excel file if it’s open, then run again.")
        sys.exit(1)
    except Exception as e:
        print("❌ Unexpected error while generating sample data:")
        print(f"   {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()