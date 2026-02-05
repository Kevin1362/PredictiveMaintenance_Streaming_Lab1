import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.config import RAW_TABLE, TIME_COL, AXIS_COLS
from src.db import ensure_tables, insert_raw_rows, get_conn

TRAINING_CSV = "data/training/RMBR4-2_export_test.csv"

def _to_number_series(s: pd.Series) -> pd.Series:
    """
    Convert a column to numeric safely:
    - strips commas
    - strips non-numeric characters (keeps digits, dot, minus)
    - coerces errors to NaN
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def _parse_time_to_seconds(time_series: pd.Series) -> pd.Series:
    """
    Handles:
    - numeric time (already seconds)
    - hh:mm:ss / mm:ss / etc. via to_timedelta
    - datetime strings via to_datetime then convert to seconds from start
    """
    # First try numeric
    t_num = pd.to_numeric(time_series, errors="coerce")
    if t_num.notna().sum() > 0 and t_num.notna().sum() >= len(time_series) * 0.8:
        return t_num

    # Try timedelta (00:00:01)
    t_td = pd.to_timedelta(time_series, errors="coerce")
    if t_td.notna().sum() > 0 and t_td.notna().sum() >= len(time_series) * 0.8:
        return t_td.dt.total_seconds()

    # Try datetime strings
    t_dt = pd.to_datetime(time_series, errors="coerce")
    if t_dt.notna().sum() > 0 and t_dt.notna().sum() >= len(time_series) * 0.8:
        # convert to seconds from the first timestamp
        base = t_dt.dropna().iloc[0]
        return (t_dt - base).dt.total_seconds()

    # If none works well, return numeric coercion (will mostly be NaN)
    return t_num

def main():
    if not os.path.exists(TRAINING_CSV):
        raise FileNotFoundError(
            f"Training CSV not found: {TRAINING_CSV}\n"
            f"Put your file here: data/training/RMBR4-2_export_test.csv"
        )

    print(f"Reading: {TRAINING_CSV}")
    df = pd.read_csv(TRAINING_CSV)
    print("Original columns:", df.columns.tolist())
    print("Original rows:", len(df))

    # Rename columns from your CSV -> expected project names
    rename_map = {
        "Time": "time_s",
        "Axis #1": "axis_1",
        "Axis #2": "axis_2",
        "Axis #3": "axis_3",
        "Axis #4": "axis_4",
        "Axis #5": "axis_5",
        "Axis #6": "axis_6",
        "Axis #7": "axis_7",
        "Axis #8": "axis_8",
    }
    df = df.rename(columns=rename_map)

    needed = ["time_s"] + [f"axis_{i}" for i in range(1, 9)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"CSV missing required columns after rename: {missing}\n"
            f"CSV has columns: {df.columns.tolist()}"
        )

    # Convert Time to seconds (robust)
    df["time_s"] = _parse_time_to_seconds(df["time_s"])

    # Convert axes to numeric (robust)
    for c in [f"axis_{i}" for i in range(1, 9)]:
        df[c] = _to_number_series(df[c])

    # Keep only required columns
    df = df[needed]

    # Print non-null counts before dropna
    print("\nNon-null counts before dropna:")
    print(df.notna().sum())

    # Drop rows with any missing required values
    df_clean = df.dropna().sort_values("time_s").reset_index(drop=True)

    print("\nRows after cleaning:", len(df_clean))
    if len(df_clean) == 0:
        raise RuntimeError(
            "After converting columns, 0 rows remain.\n"
            "This usually means your Time column is not numeric/time-like or axis values are not parseable.\n"
            "Run: python -c \"import pandas as pd; df=pd.read_csv('data/training/RMBR4-2_export_test.csv'); print(df.head(10))\""
        )

    # Create tables
    ensure_tables(TIME_COL, AXIS_COLS)

    # Upload in chunks
    chunk_size = 500
    total = len(df_clean)
    for i in range(0, total, chunk_size):
        insert_raw_rows(df_clean.iloc[i:i + chunk_size], TIME_COL, AXIS_COLS)
        print(f"Uploaded rows {i} to {min(i + chunk_size, total)} / {total}")

    print("✅ Training data upload complete.")

    # Verify count in Neon
    with get_conn() as conn:
        c = pd.read_sql(f"SELECT COUNT(*) AS cnt FROM {RAW_TABLE}", conn)
    print("\nNeon row count now:")
    print(c)

if __name__ == "__main__":
    main()
