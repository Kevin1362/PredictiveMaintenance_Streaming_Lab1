import os
import json
import pandas as pd

from dotenv import load_dotenv
load_dotenv()


from .config import TIME_COL, AXIS_COLS
from .db import ensure_tables, read_training_data, insert_events
from .preprocessing import fit_train_scalers, transform_zscore
from .regression import fit_models, residuals
from .synthetic_generator import generate_synthetic, inject_anomalies
from .detector import RuleConfig, detect_events_for_axis

def main():
    ensure_tables(TIME_COL, AXIS_COLS)

    # 1) Pull training from DB
    train = read_training_data()
    if train.empty:
        raise RuntimeError("No training data found in DB. Stream/insert training first into RAW_TABLE.")

    # 2) Fit scalers on training
    scalers = fit_train_scalers(train, AXIS_COLS)
    os.makedirs("outputs/models", exist_ok=True)
    with open("outputs/models/scalers.json", "w", encoding="utf-8") as f:
        json.dump(scalers, f, indent=2)

    # 3) Standardize training before fitting LR (explain this choice in README/notebook)
    train_z = transform_zscore(train, AXIS_COLS, scalers)

    # 4) Fit regression models (Time -> axes)
    models = fit_models(train_z, TIME_COL, AXIS_COLS)
    with open("outputs/models/linreg_models.json", "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2)

    # 5) Generate synthetic test from training distribution
    test = generate_synthetic(train, TIME_COL, AXIS_COLS, n_rows=3000, seed=7)

    # Inject a couple sustained anomalies so alerts/errors appear (adjust bump/time)
    test = inject_anomalies(test, TIME_COL, "axis_2", start_time=float(test[TIME_COL].iloc[500]), duration_s=20, bump=2.0)
    test = inject_anomalies(test, TIME_COL, "axis_5", start_time=float(test[TIME_COL].iloc[1200]), duration_s=30, bump=4.0)

    os.makedirs("data/synthetic_test", exist_ok=True)
    test.to_csv("data/synthetic_test/synthetic_test.csv", index=False)

    # 6) Standardize test using TRAIN scalers (rubric requirement)
    test_z = transform_zscore(test, AXIS_COLS, scalers)

    # 7) Example thresholds (YOU MUST replace with discovered values from notebook)
    cfg = RuleConfig(minC=0.8, maxC=1.4, T=5.0)

    all_events = []
    for ax in AXIS_COLS:
        r, _ = residuals(test_z, TIME_COL, ax, models[ax])
        dev = [max(0.0, float(v)) for v in r]  # positive only
        t = test_z[TIME_COL].to_numpy()

        ev = detect_events_for_axis(t, dev, ax, cfg)
        all_events.extend(ev)

    # 8) Log events
    os.makedirs("outputs/logs", exist_ok=True)
    events_df = pd.DataFrame(all_events)
    events_df.to_csv("outputs/logs/events.csv", index=False)

    # 9) Store events in DB table
    insert_events(all_events)

    print("Done.")
    print(f"Events found: {len(all_events)}")
    if len(all_events):
        print(events_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
