"""Download CAMS history, evaluate, and train the AirGuard AQI model."""

import argparse
from pathlib import Path
import json

from ml_model import (
    DEFAULT_TRAINING_LATITUDE,
    DEFAULT_TRAINING_LONGITUDE,
    HISTORY_DAYS,
    fetch_cams_history,
    train_and_save,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, default=DEFAULT_TRAINING_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_TRAINING_LONGITUDE)
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not 30 <= args.history_days <= 92:
        parser.error("--history-days must be between 30 and 92 (the CAMS API limit).")

    history, _ = fetch_cams_history(
        args.latitude,
        args.longitude,
        past_days=args.history_days,
    )
    metadata = train_and_save(
        history,
        args.output_dir,
        source="Open-Meteo Air Quality API, CAMS hourly model estimates",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
