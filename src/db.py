import psycopg2
import pandas as pd
from psycopg2.extras import execute_values

from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, RAW_TABLE, EVENTS_TABLE

def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
    )

def ensure_tables(time_col: str, axis_cols: list[str]):
    """
    Creates:
    - RAW_TABLE to store streamed current readings
    - EVENTS_TABLE to store detected ALERT/ERROR events
    """
    with get_conn() as conn, conn.cursor() as cur:
        cols_sql = ", ".join([f"{c} DOUBLE PRECISION" for c in axis_cols])
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                {time_col} DOUBLE PRECISION,
                {cols_sql},
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {EVENTS_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                axis_name TEXT NOT NULL,
                event_type TEXT NOT NULL,          -- ALERT or ERROR
                start_time DOUBLE PRECISION NOT NULL,
                end_time DOUBLE PRECISION NOT NULL,
                duration_s DOUBLE PRECISION NOT NULL,
                threshold DOUBLE PRECISION NOT NULL,
                max_deviation DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

def insert_raw_rows(df: pd.DataFrame, time_col: str, axis_cols: list[str]):
    cols = [time_col] + axis_cols

    # ✅ Convert numpy types to plain Python floats (and handle NaN)
    records = []
    for row in df[cols].itertuples(index=False, name=None):
        cleaned = []
        for v in row:
            if v is None:
                cleaned.append(None)
            else:
                # convert numpy scalar -> python float
                try:
                    cleaned.append(float(v))
                except Exception:
                    cleaned.append(None)
        records.append(tuple(cleaned))

    with get_conn() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {RAW_TABLE} ({', '.join(cols)}) VALUES %s",
            records
        )
        conn.commit()


def read_training_data(limit: int | None = None) -> pd.DataFrame:
    q = f"SELECT * FROM {RAW_TABLE} ORDER BY id"
    if limit:
        q += f" LIMIT {limit}"
    with get_conn() as conn:
        return pd.read_sql(q, conn)

def insert_events(events: list[dict]):
    if not events:
        return
    cols = ["axis_name","event_type","start_time","end_time","duration_s","threshold","max_deviation"]
    values = [tuple(e[c] for c in cols) for e in events]
    with get_conn() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {EVENTS_TABLE} ({', '.join(cols)}) VALUES %s",
            values
        )
