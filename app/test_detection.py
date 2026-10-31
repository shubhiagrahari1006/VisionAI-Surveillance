
from pathlib import Path
from detection import detect_image

project_folder = Path(__file__).resolve().parent.parent
image_path = project_folder / "input" / "images" / "test.jpg"
output_folder = project_folder / "output"

if not image_path.exists():
    print(f"Image not found: {image_path}")
else:
    detections = detect_image(image_path, str(output_folder))

    print("\nDetection Results")
    print("-----------------")

    if detections:
        for item in detections:
            print(
                f"Object: {item['class']}, "
                f"Confidence: {item['confidence']:.1%}"
            )
        print(f"\nTotal detections: {len(detections)}")
    else:
        print("No objects detected.")

