import cv2
import numpy as np
from ultralytics import YOLO
import os
 
# ── Model path ────────────────────────────────────────────────
# Set via env var  DEFECT_MODEL_PATH  or fall back to default location
MODEL_PATH = os.environ.get(
    "DEFECT_MODEL_PATH",
    r"C:\Users\hp\Downloads\best (6).pt"
)
 
_model = None
 
 
def load_model():
    """Load YOLO model once and cache it globally."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}\n"
                "Set the DEFECT_MODEL_PATH environment variable to the correct path."
            )
        _model = YOLO(MODEL_PATH)
    return _model
 
 
def detect_defects(frame, confidence_threshold: float = 0.4):
    """
    Run OBB defect detection on a single BGR frame.
 
    Returns
    -------
    annotated_frame : np.ndarray  (BGR, same size as input)
    detections      : list of dicts with keys:
                      defect_type, confidence, x1, y1, x2, y2
    """
    if frame is None:
        return np.zeros((100, 100, 3), dtype=np.uint8), []
 
    model = load_model()
 
    results = model.predict(
        frame,
        conf=confidence_threshold,
        verbose=False
    )
 
    annotated_frame = frame.copy()
    detections = []
 
    for result in results:
 
        if result.obb is None:
            continue
 
        # xyxyxyxy → shape (N, 4, 2)
        polygons = result.obb.xyxyxyxy.cpu().numpy()
        confs    = result.obb.conf.cpu().numpy()
        classes  = result.obb.cls.cpu().numpy()
 
        for poly, conf, cls_id in zip(polygons, confs, classes):
 
            # poly shape: (4, 2)  — already correct
            pts = poly.astype(np.int32)
 
            # FIX: cv2.polylines needs shape (N, 1, 2)
            pts_draw = pts.reshape(-1, 1, 2)
 
            cv2.polylines(
                annotated_frame,
                [pts_draw],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2
            )
 
            x_min = int(pts[:, 0].min())
            y_min = int(pts[:, 1].min())
            x_max = int(pts[:, 0].max())
            y_max = int(pts[:, 1].max())
 
            label = model.names[int(cls_id)]
 
            cv2.putText(
                annotated_frame,
                f"{label} {conf:.2f}",
                (x_min, max(30, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
 
            detections.append({
                "defect_type": label,
                "confidence":  float(conf),
                "x1": x_min,
                "y1": y_min,
                "x2": x_max,
                "y2": y_max
            })
 
    return annotated_frame, detections