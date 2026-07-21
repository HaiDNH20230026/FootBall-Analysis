"""Load model và chạy detection.

Hai cách lấy model, dùng chung interface `.infer(frame, confidence=...)`:
  - load_models()        : tải tham số từ Roboflow API (cần ROBOFLOW_API_KEY).
  - load_models_local()  : dùng weights .pt local qua Ultralytics YOLO (offline).

detect() tự nhận biết kết quả đến từ Roboflow hay Ultralytics nên phần còn lại của
package không cần biết model được load kiểu nào.
"""
import os

import supervision as sv

from .config import (CONF, DETECTION_WEIGHTS, PITCH_WEIGHTS,
                     DETECTION_IMGSZ, PITCH_IMGSZ)

PLAYER_MODEL_ID = "soccernet-1ae2v-e6wqy/3"
FIELD_MODEL_ID = "football-field-detection-f07vi/15"


def load_models(api_key=None):
    """Tải (player_model, field_model) từ Roboflow API."""
    from inference import get_model

    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Thiếu ROBOFLOW_API_KEY (truyền vào hoặc set biến môi trường).")
    player_model = get_model(model_id=PLAYER_MODEL_ID, api_key=api_key)
    field_model = get_model(model_id=FIELD_MODEL_ID, api_key=api_key)
    return player_model, field_model


class _UltralyticsAdapter:
    """Bọc Ultralytics YOLO để có interface .infer() giống Roboflow inference.

    imgsz PHẢI khớp độ phân giải lúc train: predict() mặc định 640 trong khi
    model detection train @1280 — chạy mặc định làm sót 5-10 người/frame ở xa
    (đo trên video mẫu: @640 thấy 13-19 người, @1280 thấy 22-25).
    """

    def __init__(self, model, imgsz=None):
        self.model = model
        self.imgsz = imgsz

    def infer(self, frame, confidence=CONF):
        # predict trả về list các Results; giữ [0] ở chỗ gọi cho đồng nhất Roboflow.
        kwargs = dict(conf=confidence, verbose=False)
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        return self.model.predict(frame, **kwargs)


def load_models_local(detection_path=None, pitch_path=None, device=None):
    """Tải (player_model, field_model) từ weights .pt local (Ultralytics YOLO)."""
    from ultralytics import YOLO

    detection_path = detection_path or DETECTION_WEIGHTS
    pitch_path = pitch_path or PITCH_WEIGHTS
    for p in (detection_path, pitch_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy weights: {p} (xem models/README.md).")
    det = YOLO(detection_path)
    fld = YOLO(pitch_path)
    if device:
        det.to(device)
        fld.to(device)
    return (_UltralyticsAdapter(det, imgsz=DETECTION_IMGSZ),
            _UltralyticsAdapter(fld, imgsz=PITCH_IMGSZ))


def detect(model, frame, conf=CONF):
    """Chạy detection trên 1 frame, trả về sv.Detections (nguồn nào cũng được)."""
    result = model.infer(frame, confidence=conf)[0]
    if hasattr(result, "boxes"):                     # Ultralytics Results
        return sv.Detections.from_ultralytics(result)
    return sv.Detections.from_inference(result)       # Roboflow inference
