
from pathlib import Path
from storage import save_detection_history

project_folder = Path(__file__).resolve().parent.parent
test_file = project_folder / "data" / "test_detection_history.csv"

sample_records = [
    {
        "object_class": "car",
        "confidence": 0.93,
        "tracking_id": 1,
        "event": "detected",
    },
    {
        "object_class": "person",
        "confidence": 0.87,
        "tracking_id": 2,
        "event": "detected",
    },
]

saved_path = save_detection_history(sample_records, test_file)

print("Storage test completed!")
print(f"Saved file: {saved_path}")
print(f"File exists: {saved_path.exists()}")
print(f"File size: {saved_path.stat().st_size} bytes")
