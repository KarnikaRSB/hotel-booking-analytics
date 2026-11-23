"""
TEMPORARY stand-in for the course's BatchSimulator (replace once we receive it).

Creates a CSV of "new" bookings in Data/batches/incoming/ by sampling rows from the raw
dataset and slightly changing them, so they are not exact duplicates of existing bookings.

Run:  python Scripts/simulate_batch.py          (default 500 rows)
      python Scripts/simulate_batch.py 1000
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ("Data" if (ROOT / "Data").exists() else "data")
RAW = DATA / "raw" / "hotel_bookings.csv"
INCOMING = DATA / "batches" / "incoming"


def main(n: int = 500):
    rng = np.random.default_rng()
    batch = pd.read_csv(RAW).sample(n, random_state=int(rng.integers(1_000_000)))
    batch["lead_time"] = (batch["lead_time"] + rng.integers(-5, 6, n)).clip(lower=0)
    batch["adr"] = (batch["adr"] * rng.uniform(0.95, 1.05, n)).round(2)

    INCOMING.mkdir(parents=True, exist_ok=True)
    out = INCOMING / f"batch_{datetime.now():%Y%m%d_%H%M%S}.csv"
    batch.to_csv(out, index=False)
    print(f"Created {out.relative_to(ROOT)} with {n} rows")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
