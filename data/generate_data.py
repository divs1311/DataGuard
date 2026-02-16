"""
Generate a realistic sample dataset for DataGuard testing.
Includes intentional quality issues: missing values, duplicates, outliers.
"""

import pandas as pd
import numpy as np
import os


def generate_sample_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset mimicking loan application / customer records."""
    rng = np.random.default_rng(seed)

    customer_ids = np.arange(1, n_rows + 1)

    ages = rng.normal(loc=40, scale=12, size=n_rows).clip(18, 85).astype(int)

    income = rng.lognormal(mean=10.8, sigma=0.5, size=n_rows).round(2)

    credit_scores = (ages * 5 + rng.normal(0, 50, n_rows) + 450).clip(300, 850).astype(int)

    loan_amounts = rng.lognormal(mean=10, sigma=0.8, size=n_rows).round(2)

    employment_years = (ages - 18 - rng.exponential(5, n_rows)).clip(0, 50).round(1)

    transaction_amounts = rng.exponential(500, n_rows).round(2)

    categories = rng.choice(["A", "B", "C", "D", "E"], size=n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.1])

    regions = rng.choice(["North", "South", "East", "West", "Central"], size=n_rows)

    approved = ((credit_scores > 600) & (income > 30000)).astype(int)
    noise_mask = rng.random(n_rows) < 0.05
    approved[noise_mask] = 1 - approved[noise_mask]

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "income": income,
        "credit_score": credit_scores,
        "loan_amount": loan_amounts,
        "employment_years": employment_years,
        "transaction_amount": transaction_amounts,
        "category": categories,
        "region": regions,
        "approved": approved,
    })

    # --- Inject quality issues ---

    # Missing values: ~12% in income, ~15% in employment_years, ~8% in credit_score
    income_missing = rng.choice(n_rows, size=int(n_rows * 0.12), replace=False)
    df.loc[income_missing, "income"] = np.nan

    emp_missing = rng.choice(n_rows, size=int(n_rows * 0.15), replace=False)
    df.loc[emp_missing, "employment_years"] = np.nan

    cs_missing = rng.choice(n_rows, size=int(n_rows * 0.08), replace=False)
    df.loc[cs_missing, "credit_score"] = np.nan

    # Outliers: inject extreme values in transaction_amount
    outlier_idx = rng.choice(n_rows, size=int(n_rows * 0.03), replace=False)
    df.loc[outlier_idx, "transaction_amount"] = rng.uniform(50000, 200000, size=len(outlier_idx))

    # Outliers in age (impossible values)
    age_outlier_idx = rng.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    df.loc[age_outlier_idx, "age"] = rng.choice([0, 1, 2, 120, 130, 150], size=len(age_outlier_idx))

    # Duplicates: duplicate ~2% of rows
    dup_idx = rng.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_sample_data()
    df.to_csv("data/sample_data.csv", index=False)
    print(f"Generated sample data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved to data/sample_data.csv")
