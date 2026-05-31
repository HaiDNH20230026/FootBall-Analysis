"""Quỹ đạo bóng trên map 2D = đường gấp khúc nối các lần chạm bóng.

Tại lần chạm, bóng sát mặt sân (gần bàn chân) -> homography đáng tin. Giữa hai
lần chạm (kể cả khi bóng bay) -> nội suy thẳng, nên không còn vòng cung.
"""
import numpy as np

from .config import BALL_TOUCH_PX


def smooth_ball_path(records, fps, touch_px=BALL_TOUCH_PX):
    """records: list dict mỗi frame (xem pipeline.analyze_video).

    touch_px: ngưỡng khoảng cách bóng -> bàn chân gần nhất trong ảnh (~1080p).
    """
    F = len(records)
    raw = [r["ball_pitch"] for r in records]
    detected = [f for f, p in enumerate(raw) if p is not None]
    if len(detected) < 2:
        return raw

    touch = []
    for f, r in enumerate(records):
        bb = r["ball_box"]
        if raw[f] is None or bb is None or len(r["xyxy"]) == 0:
            continue
        bcx = (bb[0] + bb[2]) / 2.0
        bcy = (bb[1] + bb[3]) / 2.0
        foot_x = (r["xyxy"][:, 0] + r["xyxy"][:, 2]) / 2.0
        foot_y = r["xyxy"][:, 3]
        if np.min(np.hypot(foot_x - bcx, foot_y - bcy)) < touch_px:
            touch.append(f)

    if len(touch) < 2:
        anchors = detected                       # không đủ mốc -> giữ nguyên
    else:
        anchors = sorted(set(touch))
        if detected[0] < anchors[0]:
            anchors = [detected[0]] + anchors
        if detected[-1] > anchors[-1]:
            anchors = anchors + [detected[-1]]

    axs = np.array(anchors, float)
    ap = np.array([np.asarray(raw[i], float) for i in anchors])
    out = [None] * F
    for f in range(F):
        if f <= anchors[0]:
            out[f] = ap[0]
        elif f >= anchors[-1]:
            out[f] = ap[-1]
        else:
            out[f] = np.array([np.interp(f, axs, ap[:, 0]), np.interp(f, axs, ap[:, 1])])
    return out
