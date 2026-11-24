import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent          # repo root, so relative paths like Data/... always work

# Same order as the GitHub Actions workflow
PIPELINE = [
    "Scripts/data_exam.py",        # examine and clean the raw data
    "Scripts/simulate_batch.py",   # temporary: create a fake batch (remove once you have the course BatchSimulator)
    "Scripts/process_batch.py",    # clean new batches and append them
    "Scripts/indicators.py",       # indicators for the dashboard (booking analysis, cancellation rate, etc.)
    "Scripts/ml_model.py",         # train the model and score bookings
    "Scripts/generate_report.py",  # build the report
    "Scripts/dashboard.py",        # run the dashboard
]

try:
    for name in PIPELINE:
        print(f"\n{'=' * 10} Running {name} {'=' * 10}")
        result = subprocess.run([sys.executable, str(ROOT / name)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStopped: {name} failed (exit code {result.returncode}).")
            sys.exit(result.returncode)
    print("\nPipeline finished successfully.")
except KeyboardInterrupt:
    print("\nStopped by user.")