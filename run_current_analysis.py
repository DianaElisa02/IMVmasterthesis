"""Run the current preferred analysis pipeline in the required order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "run_binned_did.py",
    "run_late_post_did.py",
    "run_event_study.py",
    "run_preferred_placebo.py",
    "run_leave_one_region_out.py",
    "make_updated_thesis_figures.py",
    "validate_current_analysis.py",
]


def main() -> None:
    for script in SCRIPTS:
        path = BASE_DIR / script
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"\n=== Running {script} ===")
        subprocess.run([sys.executable, str(path)], cwd=BASE_DIR, check=True)

    print("\nCurrent preferred analysis pipeline completed successfully.")


if __name__ == "__main__":
    main()
