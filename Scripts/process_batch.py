"""
Member 4 - Automation: process new batches of bookings.

Every CSV placed in Data/batches/incoming/ is:
  1. cleaned with the same clean() function as the original data (Karnika's data exam script),
  2. appended to Data/clean/hotel_bookings_clean.csv (duplicates removed),
  3. moved to Data/batches/processed/ so it is never processed twice.

After this, ml_model.py re-scores everything and generate_report.py rebuilds the report.
If there is no new batch, the script says so and exits without changing anything.

Run:  python Scripts/process_batch.py
"""
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ("Data" if (ROOT / "Data").exists() else "data")
CLEAN = DATA / "clean" / "hotel_bookings_clean.csv"
INCOMING = DATA / "batches" / "incoming"
PROCESSED = DATA / "batches" / "processed"


def load_clean_function():
    """Import clean() from the data exam script, whichever name it has in the repo."""
    for name in ["data_exam.py", "data-exam.py", "data_examination.py"]:
        path = ROOT / "Scripts" / name
        if path.exists():
            spec = importlib.util.spec_from_file_location("data_exam", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.clean
    raise FileNotFoundError("Could not find the data exam script in Scripts/")


def process_file(path: Path, clean) -> int:
    batch = pd.read_csv(path)
    cleaned = clean(batch)

    base = pd.read_csv(CLEAN, parse_dates=["reservation_status_date"])
    before = len(base)
    combined = pd.concat([base, cleaned], ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined.to_csv(CLEAN, index=False)
    added = len(combined) - before

    PROCESSED.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.move(str(path), PROCESSED / f"{stamp}_{path.name}")

    print(f"{path.name}: {len(batch)} rows received, {len(cleaned)} after cleaning, "
          f"{added} new rows added (total {len(combined)})")
    return added


def main():
    INCOMING.mkdir(parents=True, exist_ok=True)
    files = sorted(INCOMING.glob("*.csv"))
    if not files:
        print("No new batch in", INCOMING.relative_to(ROOT), "- nothing to do.")
        return

    clean = load_clean_function()
    total = sum(process_file(f, clean) for f in files)
    print(f"\nProcessed {len(files)} batch file(s), {total} new bookings added.")


if __name__ == "__main__":
    main()
