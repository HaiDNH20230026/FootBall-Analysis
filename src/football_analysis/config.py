"""Cấu hình dùng chung — đọc từ configs/default.yaml (fallback về mặc định).

Mọi tham số tinh chỉnh nằm trong configs/default.yaml. Module này nạp file đó một
lần khi import và expose ra:
  - CONFIG: dict đầy đủ (đã merge với mặc định).
  - Các hằng số phẳng (CONF, KP_CONF, ...) để code cũ `from .config import CONF` vẫn chạy.

Đổi tham số: sửa configs/default.yaml (không cần sửa code). Muốn dùng file khác:
    from football_analysis import config
    config.reload("đường/dẫn/khác.yaml")
"""
import copy
import os
from pathlib import Path

import yaml
from sports.configs.soccer import SoccerPitchConfiguration

# Gốc repo = .../football-analysis (file này ở src/football_analysis/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# Giá trị mặc định — dùng khi thiếu file YAML hoặc thiếu khoá.
_DEFAULTS = {
    "models": {
        "detection": "models/players.pt",
        "pitch_keypoints": "models/pitch.pt",
    },
    "inference": {
        "device": "auto",        # auto | cuda | cpu
        "conf_threshold": 0.3,
        "kp_conf": 0.5,
        "nms_iou": 0.5,
        "imgsz_detection": 1280,  # PHẢI khớp lúc train; mặc định predict 640 làm sót người ở xa
        "imgsz_pitch": 640,       # model keypoint sân train @640
    },
    "video": {"fps": 25},
    "team_classifier": {"fit_stride": 20, "switch_ratio": 0.5},
    "homography": {
        "ransac_px": 15.0,        # sàn ngưỡng cắt outlier so với fit toàn cục (px, ~1080p)
        "min_inliers": 5,         # số inlier tối thiểu để tin homography của frame
        "max_reproj_px": 30.0,    # reprojection error tối đa (px) — lớn hơn thì bỏ frame
        "median_window": 5,       # lọc trung vị chuỗi keypoint, chặn outlier (frame, lẻ; 1 = tắt)
        "smooth_window": 5,       # Savitzky-Golay CÓ TÂM trên chuỗi keypoint (frame, lẻ; 1 = tắt = giống notebook)
    },
    "speed": {"pos_smooth": 5, "spd_window": 5, "max_player_ms": 12.0, "max_kmh": 40},
    "ball": {"touch_px": 70.0, "max_gap_s": 2.0},
    "render": {
        "show_radar": True,       # bật radar 2D
        "show_voronoi": True,     # bật biểu đồ Voronoi
        "show_ball": True,        # bật hiển thị quả bóng (trên khung & minimap)
        "out_fps": None,          # FPS video kết quả; None = giữ fps nguồn
    },
    "output": {"dir": "outputs/"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None) -> dict:
    """Đọc YAML rồi merge lên mặc định. Thiếu file -> trả về mặc định."""
    cfg = copy.deepcopy(_DEFAULTS)
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def resolve_device(choice: str) -> str:
    """'auto' -> cuda nếu có GPU, ngược lại cpu. 'cuda' nhưng không có GPU -> cpu."""
    if choice == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if choice == "cuda" and not has_cuda:
        print("⚠️  Cấu hình yêu cầu CUDA nhưng không thấy GPU -> dùng CPU.")
        return "cpu"
    return "cuda" if has_cuda else "cpu"


def _abspath(rel: str) -> str:
    """Đổi đường dẫn (tương đối gốc repo) thành tuyệt đối để chạy ở mọi cwd."""
    p = Path(rel)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


def _apply(cfg: dict) -> None:
    """Đổ giá trị từ dict cfg ra các hằng số module-level."""
    g = globals()
    g["CONFIG"] = cfg

    # --- Model ---
    g["DETECTION_WEIGHTS"] = _abspath(cfg["models"]["detection"])
    g["PITCH_WEIGHTS"] = _abspath(cfg["models"]["pitch_keypoints"])

    # --- Inference / ngưỡng ---
    g["DEVICE"] = resolve_device(cfg["inference"]["device"])
    g["CONF"] = cfg["inference"]["conf_threshold"]
    g["KP_CONF"] = cfg["inference"]["kp_conf"]
    g["NMS_THRESHOLD"] = cfg["inference"]["nms_iou"]
    g["DETECTION_IMGSZ"] = cfg["inference"]["imgsz_detection"]
    g["PITCH_IMGSZ"] = cfg["inference"]["imgsz_pitch"]

    # --- Video / team / homography / speed / ball ---
    g["FPS"] = cfg["video"]["fps"]
    g["FIT_STRIDE"] = cfg["team_classifier"]["fit_stride"]
    g["SWITCH_RATIO"] = cfg["team_classifier"]["switch_ratio"]
    g["HOMOGRAPHY_RANSAC_PX"] = cfg["homography"]["ransac_px"]
    g["HOMOGRAPHY_MIN_INLIERS"] = cfg["homography"]["min_inliers"]
    g["HOMOGRAPHY_MAX_REPROJ_PX"] = cfg["homography"]["max_reproj_px"]
    g["HOMOGRAPHY_MEDIAN_WINDOW"] = cfg["homography"]["median_window"]
    g["HOMOGRAPHY_SMOOTH_WINDOW"] = cfg["homography"]["smooth_window"]
    g["SPEED_POS_SMOOTH"] = cfg["speed"]["pos_smooth"]
    g["SPEED_WINDOW"] = cfg["speed"]["spd_window"]
    g["SPEED_MAX_PLAYER_MS"] = cfg["speed"]["max_player_ms"]
    g["SPEED_MAX_KMH"] = cfg["speed"]["max_kmh"]
    g["BALL_TOUCH_PX"] = cfg["ball"]["touch_px"]
    g["BALL_MAX_GAP_S"] = cfg["ball"]["max_gap_s"]

    # --- Render (bật/tắt overlay + fps đầu ra) ---
    g["SHOW_RADAR"] = cfg["render"]["show_radar"]
    g["SHOW_VORONOI"] = cfg["render"]["show_voronoi"]
    g["SHOW_BALL"] = cfg["render"]["show_ball"]
    g["OUT_FPS"] = cfg["render"]["out_fps"]

    g["OUTPUT_DIR"] = _abspath(cfg["output"]["dir"])


def reload(path=None) -> dict:
    """Nạp lại cấu hình từ YAML (mặc định hoặc đường dẫn truyền vào)."""
    _apply(load_config(path))
    return CONFIG


# --- Cấu trúc cố định theo model (không nằm trong YAML) ---
# class id của model detection
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# màu hiển thị (hex; thư viện vẽ tự đổi sang BGR)
COLOR_TEAM_1 = "00BFFF"     # xanh
COLOR_TEAM_2 = "FF1493"     # hồng
COLOR_REFEREE = "FFD700"    # vàng

# toạ độ/kích thước sân chuẩn (cm)
PITCH = SoccerPitchConfiguration()

# Nạp cấu hình ngay khi import (cho phép ghi đè bằng biến môi trường FOOTBALL_CONFIG)
_apply(load_config(os.environ.get("FOOTBALL_CONFIG")))
