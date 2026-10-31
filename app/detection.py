"""
VisionAI - Object Detection Engine
Supports:
1. Autonomous Universal Vocabulary Engine (YOLO-World v2)
   - Zero human effort: Pre-baked autonomous vocabulary for hazards, gas cylinders,
     helmets, safety gear, vehicles, and pedestrians.
   - Powered by OpenAI CLIP ViT-B/32 vision-language embeddings compiled into detection heads.
2. Smart Semantic Forensic Query Parser (Layer 2)
   - Automatic visual synonym expansion (e.g. 'gas cylinder' -> ['lpg cylinder', 'red gas cylinder', etc.]).
   - Target highlighting with glowing neon borders and [TARGET MATCH] telemetry tags.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
from ultralytics import YOLO

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Model weights paths
PATH_STANDARD_MODEL = PROJECT_FOLDER / "yolo11n.pt"
PATH_UNIVERSAL_MODEL = PROJECT_FOLDER / "yolov8s-worldv2.pt"
PATH_CUSTOM_MODEL = PROJECT_FOLDER / "best.pt"

ENGINE_UNIFIED = "unified"
ENGINE_UNIVERSAL = "universal"
ENGINE_STANDARD = "standard"
ENGINE_CUSTOM = "custom"

# Layer 1 Autonomous Universal Vocabulary (0 human prompting required)
DEFAULT_AUTONOMOUS_CLASSES = [
    "person",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "bicycle",
    "lpg cylinder",
    "gas cylinder",
    "helmet",
    "fire extinguisher",
    "traffic cone",
    "cardboard box",
    "backpack",
    "suitcase",
    "cell phone",
]

# Domain-specific semantic visual synonym expansion dictionary (Layer 2)
SEMANTIC_EXPANSION_RULES: Dict[str, List[str]] = {
    "gas cylinder": ["lpg cylinder", "gas cylinder", "red gas cylinder", "metal cylinder"],
    "cylinder": ["lpg cylinder", "gas cylinder", "red gas cylinder"],
    "lpg": ["lpg cylinder", "gas cylinder", "red gas cylinder"],
    "helmet": ["helmet", "safety helmet", "motorcycle helmet"],
    "fire extinguisher": ["fire extinguisher", "red fire extinguisher"],
    "traffic cone": ["traffic cone", "orange safety cone", "construction cone"],
    "box": ["cardboard box", "box", "package", "carton"],
    "bag": ["backpack", "handbag", "suitcase", "travel bag"],
    "backpack": ["backpack", "school bag", "rucksack"],
    "phone": ["cell phone", "mobile phone", "smartphone"],
    "bicycle": ["bicycle", "cycle", "mountain bike"],
    "bike": ["motorcycle", "motorbike", "scooter", "bicycle"],
    "motorcycle": ["motorcycle", "motorbike", "scooter"],
    "car": ["car", "automobile", "sedan", "suv"],
    "truck": ["truck", "lorry", "pickup truck", "heavy vehicle"],
}

# Display name normalizer for clean analytics and reporting
CLASS_DISPLAY_MAP = {
    "lpg cylinder": "gas cylinder",
    "red gas cylinder": "gas cylinder",
    "safety helmet": "helmet",
    "cardboard box": "box",
}

ENGINE_UNIVERSAL = "universal"
ENGINE_STANDARD = "standard"

AVAILABLE_ENGINES = {
    ENGINE_UNIFIED: {
        "id": ENGINE_UNIFIED,
        "name": "Autonomous Unified AI Engine (General + Hazard Specialist)",
        "badge": "⚡ Autonomous Unified Engine",
        "model_file": "yolo11n.pt + best.pt",
        "description": "Self-intelligent automated fusion. Detects everyday objects (people, vehicles, animals) and fine-tuned hazards autonomously without manual switching.",
    },
    ENGINE_UNIVERSAL: {
        "id": ENGINE_UNIVERSAL,
        "name": "Autonomous Universal Engine (YOLO-World v2)",
        "badge": "🧠 Universal Open-Vocab",
        "model_file": "yolov8s-worldv2.pt",
        "description": "Zero-shot extended vocabulary. Detects hazards, LPG gas cylinders, helmets, safety gear & vehicles autonomously without typing prompts.",
    },
    ENGINE_STANDARD: {
        "id": ENGINE_STANDARD,
        "name": "Standard Classical Engine (YOLO11n)",
        "badge": "⚡ Classical COCO-80",
        "model_file": "yolo11n.pt",
        "description": "Closed-vocabulary baseline restricted to standard 80 COCO classes.",
    },
}

# Cache for loaded models
_cached_models: Dict[str, YOLO] = {}
_cached_classes: Dict[str, List[str]] = {}


def normalize_class_name(raw_name: str) -> str:
    """Normalize synonym labels (e.g. 'lpg cylinder' -> 'gas cylinder')."""
    cleaned = raw_name.strip().lower()
    return CLASS_DISPLAY_MAP.get(cleaned, cleaned)


def parse_and_expand_forensic_query(query: str) -> Tuple[List[str], List[str]]:
    """
    Given a user query (e.g. 'gas cylinder, helmet'), expands into semantic visual classes
    and returns (expanded_classes, normalized_target_names).
    """
    if not query or not str(query).strip():
        return [], []

    raw_tokens = [t.strip().lower() for t in str(query).split(",") if t.strip()]
    expanded = []
    targets = []

    for token in raw_tokens:
        clean_target = normalize_class_name(token)
        targets.append(clean_target)
        if token in SEMANTIC_EXPANSION_RULES:
            expanded.extend(SEMANTIC_EXPANSION_RULES[token])
        else:
            matched = False
            for key, val in SEMANTIC_EXPANSION_RULES.items():
                if key in token or token in key:
                    expanded.extend(val)
                    matched = True
                    break
            if not matched:
                expanded.append(token)

    return list(dict.fromkeys(expanded)), list(dict.fromkeys(targets))


def check_is_forensic_match(clean_class: str, raw_class: str, target_classes: List[str]) -> bool:
    """Check if a detection matches any target forensic classes."""
    if not target_classes:
        return False
    c_clean = clean_class.lower()
    c_raw = raw_class.lower()
    for t in target_classes:
        t_clean = t.lower()
        if t_clean in c_clean or c_clean in t_clean or t_clean in c_raw or c_raw in t_clean:
            return True
    return False


def get_available_engines() -> Dict[str, dict]:
    """Return dictionary of available detection engines."""
    return AVAILABLE_ENGINES


def has_custom_model() -> bool:
    """Return True if fine-tuned custom weights (best.pt) exist in project root."""
    return PATH_CUSTOM_MODEL.exists()


def get_default_autonomous_classes() -> List[str]:
    """Return the pre-baked extended vocabulary for Layer 1."""
    return list(DEFAULT_AUTONOMOUS_CLASSES)


def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def apply_inter_class_nms(detections: List[dict], iou_threshold: float = 0.50) -> List[dict]:
    """
    Suppresses lower-confidence conflicting duplicate boxes at the exact same location.
    Ensures that if an object is predicted as both 'dog' (35%) and 'horse' (28%) at the same spot,
    the higher-confidence class wins ('dog') and the conflicting duplicate is discarded.
    """
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept = []
    for cur in sorted_dets:
        boxA = cur.get("box", [cur.get("x1", 0), cur.get("y1", 0), cur.get("x2", 0), cur.get("y2", 0)])
        is_duplicate = False
        for k in kept:
            boxB = k.get("box", [k.get("x1", 0), k.get("y1", 0), k.get("x2", 0), k.get("y2", 0)])
            if compute_box_iou(boxA, boxB) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(cur)
    return kept


def get_model(engine: str = ENGINE_UNIFIED, custom_classes: Optional[List[str]] = None) -> YOLO:
    """
    Load, configure, and cache a detection model.

    Args:
        engine: 'unified', 'universal' (YOLO-World v2), 'standard' (YOLO11n), or 'custom' (best.pt).
        custom_classes: Optional custom class vocabulary (for YOLO-World).

    Returns:
        Ultralytics YOLO model instance.
    """
    global _cached_models, _cached_classes

    # Check for fine-tuned custom specialist model
    if "custom" in engine.lower():
        if not PATH_CUSTOM_MODEL.exists():
            raise FileNotFoundError(f"Custom fine-tuned weights not found: {PATH_CUSTOM_MODEL}")
        if ENGINE_CUSTOM not in _cached_models:
            _cached_models[ENGINE_CUSTOM] = YOLO(str(PATH_CUSTOM_MODEL))
        return _cached_models[ENGINE_CUSTOM]

    normalized_engine = ENGINE_STANDARD if "standard" in engine.lower() or "11" in engine.lower() else ENGINE_UNIVERSAL

    if normalized_engine == ENGINE_STANDARD or "unified" in engine.lower():
        if ENGINE_STANDARD not in _cached_models:
            if not PATH_STANDARD_MODEL.exists():
                raise FileNotFoundError(f"Standard YOLO model not found: {PATH_STANDARD_MODEL}")
            _cached_models[ENGINE_STANDARD] = YOLO(str(PATH_STANDARD_MODEL))
        return _cached_models[ENGINE_STANDARD]

    classes_to_set = custom_classes if custom_classes else DEFAULT_AUTONOMOUS_CLASSES

    if ENGINE_UNIVERSAL not in _cached_models:
        if not PATH_UNIVERSAL_MODEL.exists():
            raise FileNotFoundError(
                f"YOLO-World model not found: {PATH_UNIVERSAL_MODEL}. "
                "Ensure weights are present in workspace root."
            )
        model = YOLO(str(PATH_UNIVERSAL_MODEL))
        model.set_classes(classes_to_set)
        _cached_models[ENGINE_UNIVERSAL] = model
        _cached_classes[ENGINE_UNIVERSAL] = list(classes_to_set)
    else:
        model = _cached_models[ENGINE_UNIVERSAL]
        if _cached_classes.get(ENGINE_UNIVERSAL) != classes_to_set:
            model.set_classes(classes_to_set)
            _cached_classes[ENGINE_UNIVERSAL] = list(classes_to_set)

    return model


def detect_image(
    image_path,
    output_path=None,
    confidence=0.20,
    engine: str = ENGINE_UNIFIED,
    custom_classes: Optional[List[str]] = None,
    forensic_query: Optional[str] = None,
) -> List[dict]:
    """
    Autonomous Unified Deep Learning Detection Engine.
    Intelligently fuses General Object Understanding (COCO-80 / Open-Vocab)
    with Fine-Tuned Domain Hazard Specialists (best.pt).
    Eliminates false hallucinations and manual model switching.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if output_path is None:
        output_path = PROJECT_FOLDER / "output"
    else:
        output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # Layer 2: Semantic Query Expansion
    expanded_classes = []
    target_classes = []
    if forensic_query and str(forensic_query).strip():
        expanded_classes, target_classes = parse_and_expand_forensic_query(forensic_query)

    # Step 1: Run General Base Detector
    if expanded_classes:
        combined_classes = list(dict.fromkeys(expanded_classes + DEFAULT_AUTONOMOUS_CLASSES))
        general_model = get_model(engine=ENGINE_UNIVERSAL, custom_classes=combined_classes)
    else:
        general_model = get_model(engine=ENGINE_STANDARD)

    gen_results = general_model.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=640,
        verbose=False,
    )

    gen_detections = []
    res_gen = gen_results[0] if gen_results else None
    if res_gen is not None and res_gen.boxes is not None:
        for box in res_gen.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            raw_class = general_model.names[class_id]
            clean_class = normalize_class_name(raw_class)
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            gen_detections.append({
                "class": clean_class,
                "raw_class": raw_class,
                "confidence": round(score, 4),
                "box": [x1, y1, x2, y2],
                "source": "general",
            })

    # Step 2: Run Fine-Tuned Hazard Specialist Head (if best.pt available)
    hazard_detections = []
    if has_custom_model():
        hazard_model = get_model(engine=ENGINE_CUSTOM)
        haz_results = hazard_model.predict(
            source=str(image_path),
            conf=max(confidence, 0.22),
            imgsz=640,
            verbose=False,
        )
        res_haz = haz_results[0] if haz_results else None
        if res_haz is not None and res_haz.boxes is not None:
            for box in res_haz.boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                raw_class = hazard_model.names[class_id]
                clean_class = normalize_class_name(raw_class)
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                box_coords = [x1, y1, x2, y2]

                # Anti-Hallucination Filter:
                # Shock absorbers have lower natural prevalence, require higher confidence
                if clean_class == "shock absorbers" and score < 0.40:
                    continue

                # Anti-Hallucination Filter: Check overlap with confident non-hazard general objects
                has_conflict = False
                for gd in gen_detections:
                    if gd["confidence"] > 0.40 and gd["class"] in [
                        "car", "truck", "bus", "person", "horse", "dog", "motorcycle", "bicycle"
                    ]:
                        if compute_box_iou(box_coords, gd["box"]) > 0.35:
                            has_conflict = True
                            break

                if not has_conflict:
                    hazard_detections.append({
                        "class": clean_class,
                        "raw_class": raw_class,
                        "confidence": round(score, 4),
                        "box": box_coords,
                        "source": "hazard_specialist",
                    })

    # Step 3: Suppress generic COCO proxy objects (e.g. fire hydrant, bottle) when actual cylinder is detected
    final_gen_detections = []
    for gd in gen_detections:
        if gd["class"] in ["fire hydrant", "bottle", "vase", "sports ball"]:
            is_overridden = False
            for hd in hazard_detections:
                if compute_box_iou(gd["box"], hd["box"]) > 0.30:
                    is_overridden = True
                    break
            if not is_overridden:
                final_gen_detections.append(gd)
        else:
            final_gen_detections.append(gd)

    combined_raw = final_gen_detections + hazard_detections

    # Step 3b: Multi-Class Non-Maximum Suppression (Eliminates conflicting duplicate boxes at the same spot)
    # If the model emits both 'dog' (35%) and 'horse' (28%) for the same animal, higher confidence ('dog') wins!
    combined_raw = apply_inter_class_nms(combined_raw, iou_threshold=0.50)

    # Step 4: Build normalized output records & check forensic match
    detections = []
    for item in combined_raw:
        c_class = item["class"]
        r_class = item["raw_class"]
        score = item["confidence"]
        x1, y1, x2, y2 = item["box"]
        is_match = check_is_forensic_match(c_class, r_class, target_classes)

        detections.append({
            "class": c_class,
            "raw_class": r_class,
            "confidence": score,
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "tracking_id": None,
            "is_forensic_match": is_match,
            "source": item["source"],
        })

    # Step 5: Render Sleek Multi-Category Bounding Boxes
    annotated_image = cv2.imread(str(image_path))
    if annotated_image is None:
        raise RuntimeError(f"Could not read source image: {image_path}")

    for d in detections:
        ix1, iy1, ix2, iy2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        score = d["confidence"]
        cls_name = d["class"]
        is_match = d["is_forensic_match"]
        is_hazard = any(h in cls_name.lower() for h in ["cylinder", "extinguisher", "shock absorber"])

        if is_match:
            # Forensic Match: Glowing Electric Cyan
            color = (0, 230, 255)
            badge_text = f"[TARGET MATCH] {cls_name.upper()} {score:.0%}"
            thickness = 3
        elif is_hazard:
            # Safety Hazard: High-alert Crimson / Amber
            color = (0, 75, 255)
            badge_text = f"[HAZARD] {cls_name.title()} {score:.0%}"
            thickness = 2
        else:
            # General Everyday Objects: Bright Cyan / Electric Blue
            color = (255, 175, 0)
            badge_text = f"{cls_name.title()} {score:.0%}"
            thickness = 2

        # Draw box
        cv2.rectangle(annotated_image, (ix1, iy1), (ix2, iy2), color, thickness)

        # Draw badge label header
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        if iy1 < 25:
            tag_y1 = iy1
            tag_y2 = iy1 + 22
            text_y = iy1 + 16
        else:
            tag_y1 = iy1 - 22
            tag_y2 = iy1
            text_y = iy1 - 6

        cv2.rectangle(
            annotated_image,
            (ix1, tag_y1),
            (min(annotated_image.shape[1], ix1 + tw + 8), tag_y2),
            color,
            -1,
        )
        cv2.putText(
            annotated_image,
            badge_text,
            (ix1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (15, 20, 30),
            1,
            cv2.LINE_AA,
        )

    # Step 6: Save annotated result image
    result_folder = output_path / "image_detection"
    result_folder.mkdir(parents=True, exist_ok=True)
    result_image_path = result_folder / image_path.name

    if not cv2.imwrite(str(result_image_path), annotated_image):
        raise RuntimeError(f"Could not save annotated image: {result_image_path}")

    return detections

    return detections
