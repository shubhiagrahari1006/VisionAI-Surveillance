
from pathlib import Path
from ultralytics import YOLO

project_folder = Path(__file__).resolve().parent.parent
video_path = project_folder / "input" / "videos" / "test.mp4"
model_path = project_folder / "yolo11n.pt"
output_folder = project_folder / "output" / "tracking_test"

if not video_path.exists():
    print(f"Video not found: {video_path}")
else:
    model = YOLO(str(model_path))

    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        save=True,
        project=str(output_folder),
        name="tracked",
        exist_ok=True,
        imgsz=640,
        verbose=False,
    )

    print("\nTracking test completed!")
    print(f"Video: {video_path}")
    print(f"Annotated output folder: {output_folder / 'tracked'}")
    print(f"Frames processed: {len(results)}")

    frames_with_ids = sum(
        1 for result in results
        if result.boxes is not None
        and result.boxes.id is not None
    )

    print(f"Frames containing tracked IDs: {frames_with_ids}")
