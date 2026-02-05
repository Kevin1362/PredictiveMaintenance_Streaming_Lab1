import time
import pandas as pd
from .db import insert_raw_rows

def stream_dataframe_to_db(df: pd.DataFrame, time_col: str, axis_cols: list[str], chunk_size: int = 50, sleep_s: float = 0.1):
    """
    Simulate streaming: write small chunks to DB with delays.
    """
    n = len(df)
    for i in range(0, n, chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        insert_raw_rows(chunk, time_col, axis_cols)
        time.sleep(sleep_s)
