#!/usr/bin/env python3
"""Download MIT-licensed GDPR comparison PDFs from GitHub."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.github_regulatory import dataset_status, download_dataset


def main() -> int:
    print("Downloading kornosk/GDPR-similarity-comparison (MIT license)...")
    paths = download_dataset(force=False)
    for path in paths:
        print(f"  {path}")

    status = dataset_status()
    print("\nDataset status:")
    for key, value in status.items():
        if key != "pairs":
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
