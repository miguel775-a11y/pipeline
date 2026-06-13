from pathlib import Path
from typing import List, Tuple

import pandas as pd

from config import REQUIRED_COLUMNS


def load_master_merged(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name="Master_merged")


def validate_required_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    checks = []
    errors = []

    for col in REQUIRED_COLUMNS:
        present = col in df.columns
        non_null_count = int(df[col].notna().sum()) if present else 0
        status = "PASS" if present and non_null_count > 0 else "FAIL"

        checks.append({
            "column_name": col,
            "present": present,
            "non_null_count": non_null_count,
            "status": status,
        })

        if status == "FAIL":
            errors.append(f"{col} (present={present}, non_null_count={non_null_count})")

    return pd.DataFrame(checks), errors


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    text_cols = {
        "stock_name",
        "nse_code",
        "isin",
        "sector_name",
        "industry_name",
        "record_status",
        "merge_status",
        "best_stock_key",
    }

    for col in out.columns:
        if col in text_cols:
            continue

        if pd.api.types.is_numeric_dtype(out[col]):
            continue

        s = out[col].astype("string").str.strip()
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace("%", "", regex=False)
        s = s.replace({
            "<NA>": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "": pd.NA,
            "-": pd.NA,
            "N/A": pd.NA,
            "NA": pd.NA,
        })

        converted = pd.to_numeric(s, errors="coerce")
        out[col] = converted if converted.notna().sum() > 0 else out[col]

    return out