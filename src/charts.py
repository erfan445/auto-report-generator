from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ChartPaths:
    daily_revenue: Path
    revenue_by_category: Path
    payment_status: Path


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _placeholder_png(path: Path, title: str, message: str) -> None:
    _ensure_dir(path)
    fig = plt.figure()
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_daily_revenue(daily_revenue: pd.DataFrame, out_path: Path) -> Path:
    """
    daily_revenue columns: day, revenue
    """
    out_path = Path(out_path)
    if daily_revenue is None or daily_revenue.empty:
        _placeholder_png(out_path, "Daily Revenue", "No valid dates available for daily revenue trend.")
        return out_path

    if "day" not in daily_revenue.columns or "revenue" not in daily_revenue.columns:
        _placeholder_png(out_path, "Daily Revenue", "Missing required columns: day, revenue")
        return out_path

    df = daily_revenue.copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["day"]).sort_values("day")

    if df.empty:
        _placeholder_png(out_path, "Daily Revenue", "Daily trend data is empty after parsing.")
        return out_path

    _ensure_dir(out_path)
    fig = plt.figure()
    plt.plot(df["day"], df["revenue"])
    plt.title("Daily Revenue")
    plt.xlabel("Day")
    plt.ylabel("Revenue")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_revenue_by_category(rev_by_cat: pd.DataFrame, out_path: Path, top_n: int = 10) -> Path:
    """
    rev_by_cat columns: category, revenue
    """
    out_path = Path(out_path)
    if rev_by_cat is None or rev_by_cat.empty:
        _placeholder_png(out_path, "Revenue by Category", "No category data available.")
        return out_path

    if "category" not in rev_by_cat.columns or "revenue" not in rev_by_cat.columns:
        _placeholder_png(out_path, "Revenue by Category", "Missing required columns: category, revenue")
        return out_path

    df = rev_by_cat.copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    df["category"] = df["category"].astype("string").fillna("Unknown")

    df = df.sort_values("revenue", ascending=False).head(top_n)

    if df.empty:
        _placeholder_png(out_path, "Revenue by Category", "Category chart data is empty after filtering.")
        return out_path

    _ensure_dir(out_path)
    fig = plt.figure()
    plt.bar(df["category"], df["revenue"])
    plt.title(f"Revenue by Category (Top {min(top_n, len(df))})")
    plt.xlabel("Category")
    plt.ylabel("Revenue")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_payment_status(paid: int, unpaid: int, unknown: int, out_path: Path) -> Path:
    out_path = Path(out_path)
    total = (paid or 0) + (unpaid or 0) + (unknown or 0)

    if total == 0:
        _placeholder_png(out_path, "Payment Status", "No payment status data available.")
        return out_path

    labels = []
    sizes = []
    if paid > 0:
        labels.append("Paid")
        sizes.append(paid)
    if unpaid > 0:
        labels.append("Unpaid")
        sizes.append(unpaid)
    if unknown > 0:
        labels.append("Unknown")
        sizes.append(unknown)

    _ensure_dir(out_path)
    fig = plt.figure()
    plt.pie(sizes, labels=labels, autopct="%1.0f%%")
    plt.title("Payment Status")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_charts(
    daily_revenue: pd.DataFrame,
    revenue_by_category: pd.DataFrame,
    paid_count: int,
    unpaid_count: int,
    unknown_payment_count: int,
    out_dir: Path = Path("output/charts"),
) -> ChartPaths:
    out_dir = Path(out_dir)
    daily_path = out_dir / "daily_revenue.png"
    cat_path = out_dir / "revenue_by_category.png"
    pay_path = out_dir / "payment_status.png"

    plot_daily_revenue(daily_revenue, daily_path)
    plot_revenue_by_category(revenue_by_category, cat_path)
    plot_payment_status(paid_count, unpaid_count, unknown_payment_count, pay_path)

    return ChartPaths(daily_revenue=daily_path, revenue_by_category=cat_path, payment_status=pay_path)
