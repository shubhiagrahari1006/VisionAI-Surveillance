
from pathlib import Path
import cv2
from ultralytics import YOLO

project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "yolo11n.pt"
video_path = project_folder / "input" / "videos" / "test.mp4"
output_folder = project_folder / "output" / "video_detection"

if not video_path.exists():
    print(f"Video not found: {video_path}")
else:
    model = YOLO(str(model_path))

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "detected_video.mp4"

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened() or fps <= 0 or width <= 0 or height <= 0:
        print("Could not read the video. Check its format.")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        frame_count = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                results = model.predict(
                    frame, imgsz=640, verbose=False
                )
                annotated_frame = results[0].plot()
                writer.write(annotated_frame)
                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames")

        finally:
            cap.release()
            writer.release()

        print(f"\nVideo detection complete!")
        print(f"Frames processed: {frame_count}")
        print(f"Saved video: {output_path}")
