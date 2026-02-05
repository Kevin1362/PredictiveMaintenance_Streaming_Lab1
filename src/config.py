import os

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Expected columns (adjust if your DB uses different names)
TIME_COL = "time_s"
AXIS_COLS = [f"axis_{i}" for i in range(1, 9)]  # axis_1 ... axis_8

RAW_TABLE = os.getenv("RAW_TABLE", "robot_currents_raw")
EVENTS_TABLE = os.getenv("EVENTS_TABLE", "pm_events")
