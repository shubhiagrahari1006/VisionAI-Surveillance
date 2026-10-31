"""
VisionAI - Detection & Tracking View
Supports:
- Image Inference with YOLO11n (Sample Image or Upload).
- Video Tracking with ByteTrack, Bidirectional Counting & Speed Estimation (Sample Videos or Upload).
- Live Webcam Surveillance with HUD Telemetry & Overspeeding Alerts.
"""

from pathlib import Path
import tempfile

import cv2
import pandas as pd
from PIL import Image
import streamlit as st

from counting import count_objects
from detection import (
    ENGINE_CUSTOM,
    ENGINE_STANDARD,
    ENGINE_UNIVERSAL,
    ENGINE_UNIFIED,
    detect_image,
    get_default_autonomous_classes,
    get_model,
    has_custom_model,
    parse_and_expand_forensic_query,
)
from storage import save_detection_history
from tracking import (
    create_speed_estimator,
    create_vehicle_counter,
    process_tracking_frame,
    transcode_to_h264,
)
from reporting import generate_pdf_report, generate_forensic_dossier
from ui_theme import apply_custom_theme

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_FOLDER / "output"
INPUT_FOLDER = PROJECT_FOLDER / "input"


def make_history_records(detections, source):
    """Convert detection dictionaries into history records with forensic tagging."""
    records = []
    for item in detections:
        event_type = "forensic_match" if item.get("is_forensic_match") else "detected"
        records.append({
            "source": source,
            "object_class": item.get("class", "unknown"),
            "confidence": item.get("confidence", ""),
            "tracking_id": item.get("tracking_id"),
            "event": event_type,
            "speed_kmh": item.get("speed_kmh", ""),
            "direction": item.get("direction", ""),
        })
    return records


def show_counts(detections):
    """Display detected object counts in a styled table."""
    counts = count_objects(detections)
    if counts:
        count_df = pd.DataFrame([
            {"Object Class": name, "Count": total}
            for name, total in counts.items()
        ])
        st.dataframe(count_df, use_container_width=True, hide_index=True)
    else:
        st.info("No objects detected.")


def render_detection_page():
    """Render the Detection and Tracking Hub with Bidirectional & Speed Analytics."""
    apply_custom_theme()

    # Autonomous Unified Detection Architecture (Eliminates manual model switching)
    selected_engine = ENGINE_UNIFIED

    # User-facing Forensic Target Query (Layer 2 Smart Semantic Search)
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 10px; margin-bottom: 6px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00c0fa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <span style="font-weight: 700; font-size: 14.5px; color: var(--sidebar-title);">Forensic Target Query</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    forensic_query = st.sidebar.text_input(
        "Target Query (Optional)",
        placeholder="e.g. gas cylinder, helmet, dog",
        help="Leave blank to run fully autonomous detection on all objects & hazards. Type custom targets to trigger smart semantic forensic matching.",
    )

    expanded_classes = []
    target_classes = []
    parsed_custom_classes = None

    if forensic_query.strip():
        expanded_classes, target_classes = parse_and_expand_forensic_query(forensic_query)
        parsed_custom_classes = list(dict.fromkeys(expanded_classes + get_default_autonomous_classes()))
        st.sidebar.caption(f"Semantic Expansion ({len(expanded_classes)} latent tags):")
        st.sidebar.markdown(
            f"<div style='font-size: 0.78rem; color: #00c0fa; word-break: break-word;'>{', '.join(expanded_classes[:6])}</div>",
            unsafe_allow_html=True,
        )

    # Sidebar Surveillance Controls
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 6px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00c0fa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="4" y1="21" x2="4" y2="14"></line>
                <line x1="4" y1="10" x2="4" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12" y2="3"></line>
                <line x1="20" y1="21" x2="20" y2="16"></line>
                <line x1="20" y1="12" x2="20" y2="3"></line>
                <line x1="1" y1="14" x2="7" y2="14"></line>
                <line x1="9" y1="8" x2="15" y2="8"></line>
                <line x1="17" y1="16" x2="23" y2="16"></line>
            </svg>
            <span style="font-weight: 700; font-size: 14.5px; color: var(--sidebar-title);">Surveillance Controls</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    confidence = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.20,
        step=0.05,
    )
    line_position = st.sidebar.slider(
        "Boundary line position (%)",
        min_value=20,
        max_value=80,
        value=50,
        step=5,
    )
    speed_limit = st.sidebar.slider(
        "Campus speed limit (km/h)",
        min_value=15,
        max_value=70,
        value=30,
        step=5,
    )

    st.sidebar.info(
        "**Bidirectional Traffic Intelligence** monitors cars, motorcycles, buses and trucks. "
        "Vehicles are categorized as **Inbound (Entry)** or **Outbound (Exit)**, and speeds exceeding "
        f"**{speed_limit} km/h** trigger instant overspeeding alerts."
    )

    model_pill = '<span class="badge-pill" style="border-color: #10b981; color: #10b981;">● Autonomous Unified Engine (General + Hazard Specialist)</span>'

    # Top Navbar Header (Image 3 Style)
    st.markdown(
        f"""
        <div class="top-navbar">
            <div>
                <h2 class="top-greeting">VisionAI Detection Hub</h2>
                <div class="top-subtitle">Intelligent deep learning object detection, ByteTrack tracking & speed telemetry</div>
            </div>
            <div>
                {model_pill}
                <span class="badge-pill">ByteTrack ITS</span>
                <span class="badge-pill">Surveillance Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_image, tab_video, tab_webcam = st.tabs([
        "Static Image",
        "Video Tracking",
        "Live Webcam",
    ])

    # =========================================================
    # 1. IMAGE DETECTION
    # =========================================================
    with tab_image:
        st.markdown('<div class="section-header">Image Inference</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Choose a test sample or upload an image to analyze detections</div>', unsafe_allow_html=True)

        sample_img_dir = INPUT_FOLDER / "images"
        raw_sample_files = list(sample_img_dir.glob("*.jpg")) + list(sample_img_dir.glob("*.png"))

        def sample_sort_key(p: Path) -> int:
            name_lower = p.name.lower()
            if "test" in name_lower and "lpg" not in name_lower:
                return 0
            if "lpg" in name_lower:
                return 1
            if "getty" in name_lower or "cylinder" in name_lower:
                return 2
            return 3

        sample_img_files = sorted(raw_sample_files, key=sample_sort_key)

        def format_sample_name(p: Path) -> str:
            name_lower = p.name.lower()
            if "test.jpg" in name_lower:
                return "Test Image 1 (Everyday Scene)"
            if "lpg" in name_lower:
                return "Test Image 2 (Gas Cylinders Array)"
            if "getty" in name_lower:
                return "Test Image 3 (Roadside Cylinders)"
            return f"Test Image ({p.name})"

        img_source_choice = st.radio(
            "Select Image Source:",
            ["Use Sample Test Image", "Upload Custom Image"],
            horizontal=True,
            key="img_source_choice",
        )

        image_path_to_process = None

        if img_source_choice == "Use Sample Test Image" and sample_img_files:
            selected_sample_img = st.selectbox(
                "Choose sample image:",
                sample_img_files,
                index=None,
                placeholder="-- Select a sample image to analyze --",
                format_func=format_sample_name,
            )
            if selected_sample_img is not None:
                st.image(str(selected_sample_img), caption="Sample Image Preview", use_container_width=True)
                image_path_to_process = selected_sample_img
        else:
            uploaded_image = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                key="uploaded_image",
            )
            if uploaded_image is not None:
                img_preview = Image.open(uploaded_image).convert("RGB")
                st.image(img_preview, caption="Source Image Preview", use_container_width=True)
                sample_img_dir.mkdir(parents=True, exist_ok=True)
                save_path = sample_img_dir / Path(uploaded_image.name).name
                img_preview.save(save_path)
                image_path_to_process = save_path

        if image_path_to_process is not None:
            if st.button("Run Object Detection", key="detect_image_button"):
                try:
                    with st.spinner("Analyzing frame with VisionAI Intelligent Deep Learning Engine..."):
                        detections = detect_image(
                            image_path_to_process,
                            output_path=OUTPUT_FOLDER,
                            confidence=confidence,
                            engine=selected_engine,
                            custom_classes=parsed_custom_classes,
                            forensic_query=forensic_query,
                        )

                    result_path = OUTPUT_FOLDER / "image_detection" / image_path_to_process.name
                    result_col, info_col = st.columns([1.4, 1], gap="medium")

                    with result_col:
                        if result_path.exists():
                            st.image(
                                str(result_path),
                                caption="Annotated Inference Result",
                                use_container_width=True,
                            )
                        else:
                            st.warning("Annotated result image not found.")

                    with info_col:
                        engine_note = "Autonomous Unified Engine (General + Hazard Specialist)"
                        st.markdown(
                            f"""
                            <div class="card-featured" style="margin-bottom: 14px;">
                                <div class="metric-label">Objects Recognized</div>
                                <div class="metric-value">{len(detections)}</div>
                                <div class="metric-note">{engine_note} | Conf: {confidence:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            """
                            <div style="background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 10px 14px; margin-bottom: 14px;">
                                <div style="color: #10b981; font-weight: 700; font-size: 0.88rem;">Unified Multi-Model Fusion Active</div>
                                <div style="font-size: 0.82rem; margin-top: 3px;">COCO-80 General Detector + YOLO11s Hazard Specialist (85.6% Precision) seamlessly fused.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Highlight Forensic Query & Matches if active
                        forensic_matches = [d for d in detections if d.get("is_forensic_match")]
                        if forensic_matches:
                            st.markdown(
                                f"""
                                <div style="background: rgba(0, 229, 255, 0.12); border: 1px solid rgba(0, 229, 255, 0.45); border-radius: 12px; padding: 12px 14px; margin-bottom: 14px;">
                                    <div style="color: #00c0fa; font-weight: 700; font-size: 0.90rem;">Target Match Confirmed ({len(forensic_matches)} Found)</div>
                                    <div style="font-size: 0.84rem; margin-top: 3px;">Matched query <b>"{forensic_query}"</b> in evidence frame with glowing cyan indicators.</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        elif forensic_query.strip():
                            st.markdown(
                                f"""
                                <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.25); border-radius: 12px; padding: 10px 14px; margin-bottom: 14px;">
                                    <div style="color: #00e5ff; font-weight: 700; font-size: 0.88rem;">Forensic Query Active</div>
                                    <div style="font-size: 0.82rem; margin-top: 3px;">Searching for: <b>{forensic_query}</b> (No target in current frame)</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Highlight Autonomous Hazard Detections
                        hazards_found = [
                            d for d in detections
                            if any(h in d["class"].lower() for h in ["cylinder", "extinguisher", "cone"])
                        ]
                        if hazards_found:
                            hazard_summary = ", ".join(
                                f"{k.title()} ({v})"
                                for k, v in count_objects(hazards_found).items()
                            )
                            st.markdown(
                                f"""
                                <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 12px; padding: 12px 14px; margin-bottom: 14px;">
                                    <div style="color: #ef4444; font-weight: 700; font-size: 0.9rem;">Safety Hazard Alert</div>
                                    <div style="font-size: 0.85rem; margin-top: 4px;">Identified <b>{hazard_summary}</b> automatically in surveillance stream.</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("**Detected Breakdown**")
                        show_counts(detections)

                        if len(detections) == 0:
                            st.warning(
                                f"**No objects met the {confidence:.2f} threshold.** "
                                "Try lowering the **Confidence threshold** slider in the sidebar (e.g. to 0.15) "
                                "or select another test image."
                            )

                    if detections:
                        st.markdown("**Bounding Box Coordinates**")
                        st.dataframe(
                            pd.DataFrame(detections),
                            use_container_width=True,
                            hide_index=True,
                        )

                    records = make_history_records(detections, "image")
                    if records:
                        save_detection_history(records, source="image")

                    if forensic_matches:
                        col_s1, col_s2 = st.columns([1.2, 1], gap="medium")
                        with col_s1:
                            st.success(f"Inference complete: {len(detections)} items logged ({len(forensic_matches)} forensic target matches).")
                        with col_s2:
                            dossier_bytes = generate_forensic_dossier(
                                pd.DataFrame(make_history_records(forensic_matches, "image")),
                                target_query=forensic_query,
                            )
                            st.download_button(
                                "Download Forensic Dossier (PDF)",
                                data=dossier_bytes,
                                file_name=f"VisionAI_Forensic_Dossier_{Path(image_path_to_process).stem}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    else:
                        st.success(f"Inference complete: {len(detections)} items logged to history.")
                except Exception as error:
                    st.error(f"Inference failed: {error}")

    # =========================================================
    # 2. VIDEO DETECTION AND TRACKING
    # =========================================================
    with tab_video:
        st.markdown('<div class="section-header">Video Stream & Tracking</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Multi-object tracking with bidirectional gate counting, automated speed estimation, and universal H.264 export</div>', unsafe_allow_html=True)

        sample_vid_dir = INPUT_FOLDER / "videos"
        raw_sample_vids = list(sample_vid_dir.glob("*.mp4")) + list(sample_vid_dir.glob("*.avi"))

        def sample_vid_sort_key(p: Path) -> int:
            nl = p.name.lower()
            if "traffic_demo" in nl:
                return 0
            if "test" in nl:
                return 1
            return 2

        sample_vid_files = sorted(raw_sample_vids, key=sample_vid_sort_key)

        def format_sample_vid_name(p: Path) -> str:
            nl = p.name.lower()
            size_mb = p.stat().st_size / (1024 * 1024)
            if "traffic_demo" in nl:
                return f"Test Video 1 (Campus Traffic Demo - 720p, {size_mb:.1f} MB)"
            if "test" in nl:
                return f"Test Video 2 (Street View Traffic, {size_mb:.1f} MB)"
            return f"Test Video 3 (High-Definition Traffic, {size_mb:.1f} MB)"

        vid_source_choice = st.radio(
            "Select Video Source:",
            ["Use Sample Demo Video (Instant Test)", "Upload Custom Video"],
            horizontal=True,
            key="vid_source_choice",
        )

        video_path_to_process = None
        is_temp_file = False
        video_label = "sample_video"

        if vid_source_choice == "Use Sample Demo Video (Instant Test)" and sample_vid_files:
            selected_sample_vid = st.selectbox(
                "Choose a sample test video:",
                sample_vid_files,
                index=None,
                placeholder="-- Select a sample test video to track --",
                format_func=format_sample_vid_name,
            )
            if selected_sample_vid is not None:
                st.caption("Input Surveillance Feed Preview:")
                st.video(str(selected_sample_vid))
                video_path_to_process = selected_sample_vid
                video_label = selected_sample_vid.name
        else:
            uploaded_video = st.file_uploader(
                "Upload Video File",
                type=["mp4", "avi", "mov", "mkv"],
                key="uploaded_video",
            )
            if uploaded_video is not None:
                st.caption("Uploaded Video Feed Preview:")
                st.video(uploaded_video)
                suffix = Path(uploaded_video.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_video.getvalue())
                    video_path_to_process = Path(temp_file.name)
                    is_temp_file = True
                video_label = uploaded_video.name

        if video_path_to_process is not None:
            col_v1, col_v2 = st.columns([2, 1], gap="medium")
            with col_v1:
                st.write("")
            with col_v2:
                pass

            if st.button("Process Video Tracking", key="process_video_button"):
                cap = None
                writer = None
                raw_temp_path = None

                try:
                    output_folder = OUTPUT_FOLDER / "streamlit_video"
                    output_folder.mkdir(parents=True, exist_ok=True)
                    output_path = output_folder / "tracked_video.mp4"
                    raw_temp_path = output_folder / "raw_tracked_temp.mp4"

                    cap = cv2.VideoCapture(str(video_path_to_process))
                    if not cap.isOpened():
                        raise RuntimeError(f"Could not open video source: {video_label}")

                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    writer = cv2.VideoWriter(
                        str(raw_temp_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                    line_y = int(height * line_position / 100)
                    counter = create_vehicle_counter(line_y)
                    speed_estimator = create_speed_estimator(fps=fps, ppm=8.0, speed_limit_kmh=speed_limit)
                    model = get_model(engine=selected_engine, custom_classes=parsed_custom_classes)

                    preview_placeholder = st.empty()
                    progress = st.progress(0.0)
                    status = st.empty()
                    processed = 0
                    history_batch = []
                    all_session_records = []

                    with st.spinner("Executing ByteTrack tracking & velocity pipeline..."):
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break

                            annotated, detections, events = process_tracking_frame(
                                frame=frame,
                                counter=counter,
                                speed_estimator=speed_estimator,
                                model=model,
                                confidence=confidence,
                                frame_idx=processed,
                                speed_limit=speed_limit,
                                target_classes=target_classes,
                            )
                            writer.write(annotated)
                            processed += 1

                            frame_records = make_history_records(detections, "video")
                            history_batch.extend(frame_records)
                            all_session_records.extend(frame_records)

                            for event in events:
                                evt_type = event.get("event", "line_crossing")
                                evt_rec = {
                                    "source": "video",
                                    "object_class": event.get("object_class", "unknown"),
                                    "confidence": event.get("confidence", ""),
                                    "tracking_id": event.get("tracking_id"),
                                    "event": evt_type,
                                    "speed_kmh": event.get("speed_kmh", ""),
                                    "direction": event.get("direction", ""),
                                }
                                history_batch.append(evt_rec)
                                all_session_records.append(evt_rec)

                                if evt_type == "overspeeding":
                                    st.toast(
                                        f"Overspeeding Alert: {event.get('object_class', 'Vehicle').title()} #{event.get('tracking_id')} "
                                        f"exceeded limit ({event.get('speed_kmh')} km/h)!",
                                    )
                                elif evt_type == "forensic_match":
                                    st.toast(
                                        f"Forensic Match: {event.get('object_class', 'Target').title()} #{event.get('tracking_id', 'N/A')} identified!",
                                    )

                            if processed % 4 == 0 or processed == total_frames:
                                preview_placeholder.image(
                                    annotated,
                                    channels="BGR",
                                    caption=f"Live Inference HUD — Frame {processed}/{total_frames}",
                                    use_container_width=True,
                                )
                                if history_batch:
                                    save_detection_history(history_batch, source="video")
                                    history_batch = []
                                if total_frames > 0:
                                    progress.progress(min(processed / total_frames, 1.0))
                                status.write(f"Processed frame {processed} of {total_frames} ({int(processed/max(total_frames,1)*100)}%)")

                    if history_batch:
                        save_detection_history(history_batch, source="video")

                    # Crucial: Immediately release OpenCV writer and capture so file is finalized!
                    if writer is not None:
                        writer.release()
                        writer = None
                    if cap is not None:
                        cap.release()
                        cap = None

                    # Transcode to universal, browser & desktop-compatible H.264
                    with st.spinner("Finalizing universal H.264 video format for playback & download..."):
                        transcode_to_h264(raw_temp_path, output_path)
                        if raw_temp_path.exists():
                            raw_temp_path.unlink(missing_ok=True)

                    # Clear live preview placeholder to show complete interactive video player
                    preview_placeholder.empty()
                    progress.progress(1.0)
                    status.success(f"Tracking complete! {processed} frames analyzed successfully.")

                    # Read finalized universal H.264 video
                    tracked_video_bytes = output_path.read_bytes()

                    st.markdown('<div class="section-header">Annotated Surveillance & Tracking Result</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-desc">Interactive playback with bounding boxes, tracking IDs, speed tags, boundary lines, and live HUD</div>', unsafe_allow_html=True)
                    st.video(tracked_video_bytes)

                    summary = counter.get_summary()
                    inbound_total = summary["inbound_total"]
                    outbound_total = summary["outbound_total"]
                    net_inside = summary["net_inside"]
                    violations_total = speed_estimator.get_violations_count()

                    # 4 Bidirectional Traffic KPI Cards
                    m1, m2, m3, m4 = st.columns(4, gap="medium")
                    with m1:
                        st.markdown(
                            f"""
                            <div class="card-featured">
                                <div class="metric-label">Inbound (Entry)</div>
                                <div class="metric-value">{inbound_total}</div>
                                <div class="metric-note">↓ Entered campus zone</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            f"""
                            <div class="card-cyan">
                                <div class="metric-label">Outbound (Exit)</div>
                                <div class="metric-value">{outbound_total}</div>
                                <div class="metric-note">↑ Exited campus zone</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with m3:
                        st.markdown(
                            f"""
                            <div class="card-mint">
                                <div class="metric-label">Net On-Campus</div>
                                <div class="metric-value">{net_inside}</div>
                                <div class="metric-note">Currently inside facility</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with m4:
                        st.markdown(
                            f"""
                            <div class="card-blue">
                                <div class="metric-label">Speed Violations</div>
                                <div class="metric-value" style="color: {'#ef4444' if violations_total > 0 else 'inherit'};">{violations_total}</div>
                                <div class="metric-note">Exceeded {speed_limit} km/h limit</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.write("")
                    st.markdown("### Directional Breakdown by Vehicle Class")
                    all_classes = sorted(list(set(summary["inbound_by_class"].keys()) | set(summary["outbound_by_class"].keys())))

                    if all_classes:
                        summary_rows = []
                        for cls in all_classes:
                            in_c = summary["inbound_by_class"].get(cls, 0)
                            out_c = summary["outbound_by_class"].get(cls, 0)
                            summary_rows.append({
                                "Vehicle Class": cls.title(),
                                "Inbound (Entry)": in_c,
                                "Outbound (Exit)": out_c,
                                "Total Crossings": in_c + out_c,
                            })
                        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

                    st.write("")
                    forensic_video_matches = [r for r in all_session_records if r.get("event") == "forensic_match"]
                    if forensic_video_matches:
                        col_d1, col_d2, col_d3 = st.columns(3, gap="medium")
                        with col_d1:
                            st.download_button(
                                "Download Tracked Video (MP4)",
                                data=tracked_video_bytes,
                                file_name="tracked_video.mp4",
                                mime="video/mp4",
                                use_container_width=True,
                            )
                        with col_d2:
                            session_records_to_export = all_session_records if all_session_records else make_history_records(detections, "video")
                            if session_records_to_export:
                                pdf_session_bytes = generate_pdf_report(pd.DataFrame(session_records_to_export), speed_limit=speed_limit)
                                st.download_button(
                                    "Download Session Audit (PDF)",
                                    data=pdf_session_bytes,
                                    file_name=f"VisionAI_Session_Report_{Path(video_label).stem}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                        with col_d3:
                            dossier_pdf_bytes = generate_forensic_dossier(
                                pd.DataFrame(forensic_video_matches),
                                target_query=forensic_query or "Video Target",
                            )
                            st.download_button(
                                "Download Forensic Dossier (PDF)",
                                data=dossier_pdf_bytes,
                                file_name=f"VisionAI_Forensic_Dossier_{Path(video_label).stem}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    else:
                        col_d1, col_d2 = st.columns(2, gap="medium")
                        with col_d1:
                            st.download_button(
                                "Download Tracked Video (MP4)",
                                data=tracked_video_bytes,
                                file_name="tracked_video.mp4",
                                mime="video/mp4",
                                use_container_width=True,
                            )
                        with col_d2:
                            session_records_to_export = all_session_records if all_session_records else make_history_records(detections, "video")
                            if session_records_to_export:
                                pdf_session_bytes = generate_pdf_report(pd.DataFrame(session_records_to_export), speed_limit=speed_limit)
                                st.download_button(
                                    "Download Session Audit Report (PDF)",
                                    data=pdf_session_bytes,
                                    file_name=f"VisionAI_Session_Report_{Path(video_label).stem}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                except Exception as error:
                    st.error(f"Video processing failed: {error}")
                finally:
                    if cap is not None:
                        cap.release()
                    if writer is not None:
                        writer.release()
                    if raw_temp_path is not None and raw_temp_path.exists():
                        raw_temp_path.unlink(missing_ok=True)
                    if is_temp_file and video_path_to_process is not None:
                        video_path_to_process.unlink(missing_ok=True)

    # =========================================================
    # 3. LIVE WEBCAM
    # =========================================================
    with tab_webcam:
        st.markdown('<div class="section-header">Real-Time Webcam Stream</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Live camera feed with persistent tracking IDs, speed tags, and overspeeding alerts</div>', unsafe_allow_html=True)

        cam_col, limit_col = st.columns([1, 1])
        with cam_col:
            webcam_index = st.selectbox(
                "Select Camera Device",
                options=[0, 1, 2],
                format_func=lambda i: "Default Camera" if i == 0 else f"Camera {i}",
            )
        with limit_col:
            frame_limit = st.slider(
                "Maximum frames per session",
                min_value=30,
                max_value=600,
                value=150,
                step=30,
            )

        if st.button("Start Live Webcam Feed", key="start_webcam_button"):
            cap = cv2.VideoCapture(webcam_index)
            try:
                if not cap.isOpened():
                    raise RuntimeError("Could not connect to camera device.")

                model = get_model(engine=selected_engine, custom_classes=parsed_custom_classes)
                ok, first_frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera opened, but no frame received.")

                height, width = first_frame.shape[:2]
                line_y = int(height * line_position / 100)
                counter = create_vehicle_counter(line_y)
                speed_estimator = create_speed_estimator(fps=25.0, ppm=8.0, speed_limit_kmh=speed_limit)

                preview = st.empty()
                status = st.empty()
                count_area = st.empty()
                progress = st.progress(0.0)

                processed = 0
                history_batch = []
                all_webcam_records = []

                with st.spinner("Live camera stream active..."):
                    while processed < frame_limit:
                        frame = first_frame if processed == 0 else (cap.read()[1])
                        if frame is None:
                            status.warning("Camera stream interrupted.")
                            break

                        annotated, detections, events = process_tracking_frame(
                            frame=frame,
                            counter=counter,
                            speed_estimator=speed_estimator,
                            model=model,
                            confidence=confidence,
                            frame_idx=processed,
                            speed_limit=speed_limit,
                            target_classes=target_classes,
                        )

                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        preview.image(
                            rgb_frame,
                            channels="RGB",
                            caption="Live Object Tracking & Surveillance",
                            use_container_width=True,
                        )

                        cam_records = make_history_records(detections, "webcam")
                        history_batch.extend(cam_records)
                        all_webcam_records.extend(cam_records)

                        for event in events:
                            evt_type = event.get("event", "line_crossing")
                            evt_rec = {
                                "source": "webcam",
                                "object_class": event.get("object_class", "unknown"),
                                "confidence": event.get("confidence", ""),
                                "tracking_id": event.get("tracking_id"),
                                "event": evt_type,
                                "speed_kmh": event.get("speed_kmh", ""),
                                "direction": event.get("direction", ""),
                            }
                            history_batch.append(evt_rec)
                            all_webcam_records.append(evt_rec)

                            if evt_type == "overspeeding":
                                st.toast(
                                    f"Overspeeding Alert: {event.get('object_class', 'Vehicle').title()} #{event.get('tracking_id')} "
                                    f"({event.get('speed_kmh')} km/h)!",
                                    
                                )
                            elif evt_type == "forensic_match":
                                st.toast(
                                    f"Forensic Match: {event.get('object_class', 'Target').title()} #{event.get('tracking_id', 'N/A')} identified!",
                                    
                                )

                        processed += 1
                        if processed % 10 == 0:
                            if history_batch:
                                save_detection_history(history_batch, source="webcam")
                                history_batch = []
                            progress.progress(min(processed / frame_limit, 1.0))

                        summary = counter.get_summary()
                        count_area.write({
                            "Frames Processed": processed,
                            "Inbound (Entry)": summary["inbound_total"],
                            "Outbound (Exit)": summary["outbound_total"],
                            "Net Inside": summary["net_inside"],
                            "Speed Violations": speed_estimator.get_violations_count(),
                        })
                        status.write(f"Streaming frame {processed} of {frame_limit}")

                if history_batch:
                    save_detection_history(history_batch, source="webcam")

                progress.progress(1.0)
                st.success(f"Session finished after {processed} frames. Telemetry stored.")

                summary = counter.get_summary()
                st.markdown("### Session Traffic Summary")
                st.dataframe(
                    pd.DataFrame([
                        {"Metric": "Inbound (Entry)", "Count": summary["inbound_total"]},
                        {"Metric": "Outbound (Exit)", "Count": summary["outbound_total"]},
                        {"Metric": "Net Inside Campus", "Count": summary["net_inside"]},
                        {"Metric": "Speed Violations (> speed limit)", "Count": speed_estimator.get_violations_count()},
                        {"Metric": "Max Speed Observed", "Count": f"{speed_estimator.max_speed_observed} km/h"},
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

                if all_webcam_records:
                    forensic_cam_matches = [r for r in all_webcam_records if r.get("event") == "forensic_match"]
                    if forensic_cam_matches:
                        c_col1, c_col2 = st.columns(2, gap="medium")
                        with c_col1:
                            pdf_cam_bytes = generate_pdf_report(pd.DataFrame(all_webcam_records), speed_limit=speed_limit)
                            st.download_button(
                                "Download Session Audit (PDF)",
                                data=pdf_cam_bytes,
                                file_name="VisionAI_Webcam_Audit_Report.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        with c_col2:
                            dossier_cam_bytes = generate_forensic_dossier(
                                pd.DataFrame(forensic_cam_matches),
                                target_query=forensic_query or "Webcam Target",
                            )
                            st.download_button(
                                "Download Forensic Dossier (PDF)",
                                data=dossier_cam_bytes,
                                file_name="VisionAI_Webcam_Forensic_Dossier.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    else:
                        pdf_cam_bytes = generate_pdf_report(pd.DataFrame(all_webcam_records), speed_limit=speed_limit)
                        st.download_button(
                            "Download Webcam Session Audit Report (PDF)",
                            data=pdf_cam_bytes,
                            file_name="VisionAI_Webcam_Audit_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
            except Exception as error:
                st.error(f"Camera stream error: {error}")
            finally:
                cap.release()

    st.markdown(
        """
        <div class="footer">
            VisionAI · Real-time Intelligent Computer Vision & Surveillance
        </div>
        """,
        unsafe_allow_html=True,
    )
