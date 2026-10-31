"""
VisionAI - Telemetry & History Data Store
Appends and reads structured detection, tracking, speed, and line-crossing records.
"""

import csv
from datetime import datetime
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
HISTORY_FILE = PROJECT_FOLDER / "data" / "detection_history.csv"

FIELDNAMES = [
    "timestamp",
    "source",
    "object_class",
    "confidence",
    "tracking_id",
    "event",
    "speed_kmh",
    "direction",
]


def save_detection_history(records, file_path=None, source="unknown"):
    """
    Append detection records to the CSV history.

    Each record may contain:
    object_class, confidence, tracking_id, event, speed_kmh, direction, timestamp.
    """
    if not records:
        return HISTORY_FILE if file_path is None else Path(file_path)

    destination = (
        HISTORY_FILE if file_path is None else Path(file_path)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_exists = (
        destination.exists() and destination.stat().st_size > 0
    )

    with destination.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        for record in records:
            writer.writerow({
                "timestamp": record.get(
                    "timestamp",
                    datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                ),
                "source": record.get("source", source),
                "object_class": record.get(
                    "object_class",
                    record.get("class", ""),
                ),
                "confidence": record.get("confidence", ""),
                "tracking_id": record.get("tracking_id", ""),
                "event": record.get("event", "detected"),
                "speed_kmh": record.get("speed_kmh", ""),
                "direction": record.get("direction", ""),
            })

    return destination


def load_detection_history(file_path=None):
    """Read saved CSV records as pandas DataFrame."""
    import pandas as pd

    source_path = (
        HISTORY_FILE if file_path is None else Path(file_path)
    )

    if not source_path.exists() or source_path.stat().st_size == 0:
        return pd.DataFrame(columns=FIELDNAMES)

    try:
        history = pd.read_csv(source_path, on_bad_lines="skip")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=FIELDNAMES)

    # Support CSV files created by earlier project versions
    for column in FIELDNAMES:
        if column not in history.columns:
            history[column] = ""

    return history[FIELDNAMES]
