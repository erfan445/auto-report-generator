from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import pandas as pd


@dataclass
class KPIResult:
    total_revenue: float
    orders_count: int
    average_order_value: float
    paid_count: int
    unpaid_count: int
    unknown_payment_count: int
    revenue_by_category: pd.DataFrame
    top_products: pd.DataFrame
    daily_revenue: pd.DataFrame
    notes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # DataFrames aren't JSON-serializable by default; keep them separate in app/report
        d["revenue_by_category"] = self.revenue_by_category.to_dict(orient="records")
        d["top_products"] = self.top_products.to_dict(orient="records")
        d["daily_revenue"] = self.daily_revenue.to_dict(orient="records")
        return d


def _safe_amount_series(df: pd.DataFrame) -> pd.Series:
    if "amount" not in df.columns:
        raise ValueError("Missing required column: amount")
    return pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)


def compute_kpis(df_clean: pd.DataFrame) -> KPIResult:
    """
    Compute business KPIs safely.

    Rules:
    - Revenue uses amount with NaN treated as 0 (for KPI aggregation only).
    - Date-based trend uses only rows with valid order_date.
    - payment_status is expected to be 'Paid'/'Unpaid'/'Unknown' from cleaner.
    """
    notes: list[str] = []

    if df_clean is None or len(df_clean) == 0:
        raise ValueError("No data available after cleaning.")

    amount = _safe_amount_series(df_clean)

    orders_count = int(len(df_clean))
    total_revenue = float(amount.sum())
    average_order_value = float(total_revenue / orders_count) if orders_count > 0 else 0.0

    # Payment counts
    status = df_clean.get("payment_status", pd.Series(["Unknown"] * len(df_clean))).astype("string")
    paid_count = int((status == "Paid").sum())
    unpaid_count = int((status == "Unpaid").sum())
    unknown_payment_count = orders_count - paid_count - unpaid_count

    # Revenue by category
    if "category" in df_clean.columns:
        rev_cat = (
            df_clean.assign(amount_kpi=amount)
            .groupby("category", dropna=False)["amount_kpi"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"amount_kpi": "revenue"})
        )
    else:
        rev_cat = pd.DataFrame(columns=["category", "revenue"])
        notes.append("Category column missing; revenue_by_category is empty.")

    # Top products
    if "product" in df_clean.columns:
        top_products = (
            df_clean.assign(amount_kpi=amount)
            .groupby("product", dropna=False)
            .agg(revenue=("amount_kpi", "sum"), orders=("product", "count"))
            .sort_values(by=["revenue", "orders"], ascending=False)
            .head(5)
            .reset_index()
        )
    else:
        top_products = pd.DataFrame(columns=["product", "revenue", "orders"])
        notes.append("Product column missing; top_products is empty.")

    # Daily revenue trend (only valid dates)
    if "order_date" in df_clean.columns:
        valid_dates = df_clean["order_date"].notna()
        valid_n = int(valid_dates.sum())
        invalid_n = orders_count - valid_n
        if invalid_n > 0:
            notes.append(f"{invalid_n} rows ignored for daily trend due to invalid order_date.")

        if valid_n > 0:
            daily = (
                df_clean.loc[valid_dates]
                .assign(amount_kpi=amount.loc[valid_dates])
                .assign(day=lambda x: pd.to_datetime(x["order_date"]).dt.date)
                .groupby("day")["amount_kpi"]
                .sum()
                .reset_index()
                .rename(columns={"amount_kpi": "revenue"})
                .sort_values("day")
            )
        else:
            daily = pd.DataFrame(columns=["day", "revenue"])
            notes.append("No valid dates available for daily trend.")
    else:
        daily = pd.DataFrame(columns=["day", "revenue"])
        notes.append("order_date column missing; daily trend is empty.")

    return KPIResult(
        total_revenue=total_revenue,
        orders_count=orders_count,
        average_order_value=average_order_value,
        paid_count=paid_count,
        unpaid_count=unpaid_count,
        unknown_payment_count=unknown_payment_count,
        revenue_by_category=rev_cat,
        top_products=top_products,
        daily_revenue=daily,
        notes=notes,
    )
