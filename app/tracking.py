"""
VisionAI - Multi-Object Tracking, Speed Estimation & Boundary Surveillance
Powered by YOLO11n & ByteTrack (bytetrack.yaml).
Features:
- Bidirectional Line-Crossing Detection (Inbound/Entry vs Outbound/Exit).
- Real-time Centroid-based Speed Estimation (km/h) with EMA jitter suppression.
- Overspeeding Violation Detection with visual bounding box alerts and event logging.
- Surveillance HUD Telemetry Overlay.
"""

import os
from pathlib import Path
import subprocess
from typing import Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO

from counting import LineCrossingCounter
from detection import (
    get_model,
    normalize_class_name,
    check_is_forensic_match,
    ENGINE_UNIFIED,
    ENGINE_UNIVERSAL,
    ENGINE_STANDARD,
)

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


def get_tracking_model(engine=ENGINE_UNIFIED):
    """Load and reuse the YOLO model checkpoint using the specified engine."""
    return get_model(engine=engine)


class SpeedEstimator:
    """
    Estimates real-world vehicle speed in km/h from tracked bounding box centroids.
    Uses rolling frame displacement, FPS, and configurable pixels-per-meter (PPM) scaling.
    Includes Exponential Moving Average (EMA) smoothing to eliminate frame-to-frame noise.
    """

    def __init__(self, fps=30.0, ppm=8.0, speed_limit_kmh=30.0):
        self.fps = max(float(fps), 1.0)
        self.ppm = max(float(ppm), 1.0)
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.positions = {}          # track_id -> [(x, y, frame_idx)]
        self.speeds = {}             # track_id -> smoothed_speed_kmh
        self.alerted_ids = set()     # track_ids already flagged for overspeeding
        self.max_speed_observed = 0.0

    def update(self, track_id, center_x, center_y, frame_idx):
        """
        Update centroid for one track_id and estimate velocity.

        Returns:
            (speed_kmh, is_overspeed_event)
        """
        if track_id is None:
            return 0.0, False

        track_id = int(track_id)
        current_pos = (float(center_x), float(center_y), int(frame_idx))

        if track_id not in self.positions:
            self.positions[track_id] = [current_pos]
            return 0.0, False

        history = self.positions[track_id]
        history.append(current_pos)
        if len(history) > 6:
            history.pop(0)

        # Measure displacement over rolling window
        old_x, old_y, old_frame = history[0]
        curr_x, curr_y, curr_frame = history[-1]

        delta_frames = curr_frame - old_frame
        if delta_frames <= 0:
            return self.speeds.get(track_id, 0.0), False

        pixel_dist = ((curr_x - old_x) ** 2 + (curr_y - old_y) ** 2) ** 0.5
        time_sec = delta_frames / self.fps

        # Raw speed calculation in meters/sec, converted to km/h
        meters = pixel_dist / self.ppm
        raw_speed_kmh = (meters / time_sec) * 3.6

        # Filter out extreme teleportation noise (>140 km/h)
        clamped_speed = min(max(raw_speed_kmh, 0.0), 140.0)

        # EMA smoothing
        prev_speed = self.speeds.get(track_id, clamped_speed)
        smoothed_speed = round(0.4 * clamped_speed + 0.6 * prev_speed, 1)
        self.speeds[track_id] = smoothed_speed

        if smoothed_speed > self.max_speed_observed:
            self.max_speed_observed = smoothed_speed

        # Flag overspeeding event once per track_id
        is_overspeed_event = False
        if smoothed_speed > self.speed_limit_kmh:
            if track_id not in self.alerted_ids:
                self.alerted_ids.add(track_id)
                is_overspeed_event = True

        return smoothed_speed, is_overspeed_event

    def get_speed(self, track_id):
        """Return the current estimated speed for a track ID."""
        return self.speeds.get(int(track_id), 0.0) if track_id is not None else 0.0

    def get_violations_count(self):
        """Return total unique vehicles that exceeded the speed limit."""
        return len(self.alerted_ids)


def transcode_to_h264(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Transcode an OpenCV-generated video into universal web/desktop compatible H.264 (AVC) MP4.
    Uses bundled imageio-ffmpeg with yuv420p pixel format and faststart headers.
    Guarantees seamless playback across Chrome, Edge, Safari, Firefox, and Windows Media Player.
    """
    input_path = Path(input_path)
    target_path = Path(output_path) if output_path is not None else input_path.with_name(f"{input_path.stem}_h264.mp4")

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_target = target_path if target_path != input_path else input_path.with_name(f"{input_path.stem}_tmp.mp4")

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(temp_target),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and temp_target.exists() and temp_target.stat().st_size > 0:
            if temp_target != target_path:
                os.replace(temp_target, target_path)
            return target_path
    except Exception as e:
        print(f"[transcode_to_h264] Fallback due to error: {e}")

    return input_path


def create_vehicle_counter(line_y, allowed_classes=None):
    """Create a bidirectional line crossing counter. Defaults to all tracked objects if not specified."""
    return LineCrossingCounter(
        line_y=line_y,
        allowed_classes=allowed_classes,
    )


def create_speed_estimator(fps=30.0, ppm=8.0, speed_limit_kmh=30.0):
    """Factory function for SpeedEstimator."""
    return SpeedEstimator(fps=fps, ppm=ppm, speed_limit_kmh=speed_limit_kmh)


def draw_surveillance_hud(frame, counter, speed_estimator=None, speed_limit=30.0):
    """Draw a semi-transparent HUD telemetry banner at the top of the frame."""
    summary = counter.get_summary()
    inbound = summary["inbound_total"]
    outbound = summary["outbound_total"]
    net_inside = summary["net_inside"]
    violations = speed_estimator.get_violations_count() if speed_estimator else 0

    height, width = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 44), (17, 24, 39), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Telemetry text
    hud_text = (
        f"INBOUND: {inbound}  |  "
        f"OUTBOUND: {outbound}  |  "
        f"NET INSIDE: {net_inside}  |  "
        f"VIOLATIONS: {violations}  |  "
        f"LIMIT: {int(speed_limit)} km/h"
    )
    cv2.putText(
        frame,
        hud_text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def process_tracking_frame(
    frame,
    counter,
    speed_estimator=None,
    model=None,
    confidence=0.25,
    frame_idx=0,
    speed_limit=30.0,
    target_classes=None,
):
    """
    Track objects, estimate speeds, detect bidirectional crossings, and flag overspeeding.

    Returns:
        annotated_frame: frame with visual bounding boxes, speed tags, boundary lines and HUD
        detections: list of detected objects with tracking ID, velocity and coordinates
        crossing_events: list of events (inbound, outbound, overspeeding) triggered in this frame
    """
    if model is None:
        model = get_tracking_model()

    if speed_estimator is None:
        speed_estimator = create_speed_estimator(
            fps=30.0, ppm=8.0, speed_limit_kmh=speed_limit
        )

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=confidence,
        imgsz=640,
        verbose=False,
    )

    result = results[0]
    annotated_frame = result.plot()

    height, width = annotated_frame.shape[:2]
    line_y = int(counter.line_y)

    # Draw Bidirectional Boundary Line
    # Left side: Green for Inbound/Entry, Right side: Cyan for Outbound/Exit
    half_w = width // 2
    cv2.line(annotated_frame, (0, line_y), (half_w, line_y), (0, 230, 118), 2)
    cv2.line(annotated_frame, (half_w, line_y), (width, line_y), (250, 192, 0), 2)

    cv2.putText(
        annotated_frame,
        "| ENTRY (IN) v",
        (15, line_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 230, 118),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_frame,
        "^ EXIT (OUT) |",
        (width - 150, line_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (250, 192, 0),
        2,
        cv2.LINE_AA,
    )

    detections = []
    crossing_events = []

    if result.boxes is not None:
        boxes = result.boxes
        ids = boxes.id

        for index, box in enumerate(boxes):
            class_id = int(box.cls[0].item())
            raw_class = model.names[class_id]
            object_class = normalize_class_name(raw_class)
            score = float(box.conf[0].item())

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

            track_id = None
            speed_kmh = 0.0
            direction = None
            is_overspeed = False

            is_forensic_match = check_is_forensic_match(object_class, raw_class, target_classes)

            if ids is not None and index < len(ids):
                track_id = int(ids[index].item())
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                # 1. Real-time Speed Estimation
                if object_class in VEHICLE_CLASSES:
                    speed_kmh, is_overspeed = speed_estimator.update(
                        track_id=track_id,
                        center_x=center_x,
                        center_y=center_y,
                        frame_idx=frame_idx,
                    )
                else:
                    speed_kmh = 0.0

                # 2. Bidirectional Boundary Crossing
                direction = counter.update(
                    track_id=track_id,
                    object_class=object_class,
                    center_y=center_y,
                )

                if direction:
                    crossing_events.append({
                        "object_class": object_class,
                        "tracking_id": track_id,
                        "confidence": round(score, 4),
                        "event": f"line_crossing_{direction}",
                        "direction": direction,
                        "speed_kmh": speed_kmh,
                    })

                # 3. Overspeeding Violation Event
                if is_overspeed:
                    crossing_events.append({
                        "object_class": object_class,
                        "tracking_id": track_id,
                        "confidence": round(score, 4),
                        "event": "overspeeding",
                        "direction": direction or counter.get_trajectory(track_id) or "in-transit",
                        "speed_kmh": speed_kmh,
                    })

                # Visual Alert Rendering on Frame
                if object_class in VEHICLE_CLASSES:
                    tag_direction = counter.get_trajectory(track_id)
                    dir_str = f" [{tag_direction.upper()[:2]}]" if tag_direction else ""

                    if speed_kmh > speed_limit:
                        # Red Glowing Alert Box for Overspeeding
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 0, 255),
                            3,
                        )
                        label = f"! SPEED: {speed_kmh} km/h (ALERT)"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1) - label_size[1] - 8),
                            (int(x1) + label_size[0] + 6, int(y1)),
                            (0, 0, 255),
                            -1,
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (int(x1) + 3, int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    elif speed_kmh > 0:
                        # Standard Velocity Tag
                        speed_label = f"{speed_kmh} km/h{dir_str}"
                        cv2.putText(
                            annotated_frame,
                            speed_label,
                            (int(x1), int(y2) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (250, 192, 0),
                            2,
                            cv2.LINE_AA,
                        )

            # Forensic Target Highlight (Neon Cyan)
            if is_forensic_match:
                crossing_events.append({
                    "object_class": object_class,
                    "tracking_id": track_id,
                    "confidence": round(score, 4),
                    "event": "forensic_match",
                    "direction": direction or (counter.get_trajectory(track_id) if track_id else "") or "in-transit",
                    "speed_kmh": speed_kmh,
                })
                if not is_overspeed:
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 230, 255),
                        3,
                    )
                    match_label = f"[TARGET MATCH] {object_class.upper()}" + (f" #{track_id}" if track_id else "")
                    m_size, _ = cv2.getTextSize(match_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                    tag_y = max(0, int(y1) - 24)
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), tag_y),
                        (int(x1) + m_size[0] + 10, int(y1)),
                        (0, 230, 255),
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        match_label,
                        (int(x1) + 5, int(y1) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (10, 15, 30),
                        2,
                        cv2.LINE_AA,
                    )

            detections.append({
                "class": object_class,
                "confidence": round(score, 4),
                "tracking_id": track_id,
                "speed_kmh": speed_kmh,
                "direction": direction or (counter.get_trajectory(track_id) if track_id else None),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "is_forensic_match": is_forensic_match,
            })

    # Render HUD at the top of the video
    draw_surveillance_hud(
        annotated_frame,
        counter,
        speed_estimator,
        speed_limit=speed_limit,
    )

    return annotated_frame, detections, crossing_events


def track_video(video_path, output_path=None, confidence=0.25, speed_limit=30.0):
    """
    Track objects through a video, estimate speed, count entries/exits, and save output.
    Returns complete traffic metrics and processed video path.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_folder = (
        PROJECT_FOLDER / "output" / "tracked_video"
        if output_path is None
        else Path(output_path)
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    result_path = output_folder / f"{video_path.stem}_tracked.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video properties.")

    writer = cv2.VideoWriter(
        str(result_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {result_path}")

    model = get_tracking_model()
    counter = create_vehicle_counter(height // 2)
    speed_estimator = create_speed_estimator(fps=fps, ppm=8.0, speed_limit_kmh=speed_limit)

    frame_count = 0
    all_events = []

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, _, events = process_tracking_frame(
                frame=frame,
                counter=counter,
                speed_estimator=speed_estimator,
                model=model,
                confidence=confidence,
                frame_idx=frame_count,
                speed_limit=speed_limit,
            )
            writer.write(annotated_frame)
            frame_count += 1
            all_events.extend(events)
    finally:
        cap.release()
        writer.release()

    # Transcode output to universal H.264
    transcode_to_h264(result_path, result_path)

    summary = counter.get_summary()
    return {
        "output_path": str(result_path),
        "frames_processed": frame_count,
        "inbound_counts": counter.get_inbound_counts(),
        "outbound_counts": counter.get_outbound_counts(),
        "total_crossings": summary["total_crossings"],
        "inbound_total": summary["inbound_total"],
        "outbound_total": summary["outbound_total"],
        "net_inside": summary["net_inside"],
        "violations_count": speed_estimator.get_violations_count(),
        "max_speed_observed": speed_estimator.max_speed_observed,
    }
