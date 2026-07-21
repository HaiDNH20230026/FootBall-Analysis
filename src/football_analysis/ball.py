"""Quỹ đạo bóng trên map 2D: nối các lần chạm bóng, KHÔNG bịa vị trí khi mất dấu.

Tại lần chạm, bóng sát mặt sân (gần bàn chân) -> homography đáng tin. Giữa hai
lần chạm gần nhau (kể cả khi bóng bay) -> nội suy thẳng để khử vòng cung do bóng
rời mặt sân. Khoảng trống dài (mất dấu bóng lâu hơn max_gap_s) và đoạn ngoài
vùng phát hiện -> trả None để ẨN bóng trên map, thay vì đóng băng ở vị trí cuối.
"""
import numpy as np

from .config import BALL_TOUCH_PX, BALL_MAX_GAP_S


def smooth_ball_path(records, fps, touch_px=BALL_TOUCH_PX, max_gap_s=BALL_MAX_GAP_S):
    """records: list dict mỗi frame (xem pipeline.analyze_video).

    touch_px: ngưỡng khoảng cách bóng -> bàn chân gần nhất trong ảnh (~1080p).
    max_gap_s: khoảng trống tối đa (giây) còn được nội suy; dài hơn -> ẩn bóng.
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
        anchors = list(detected)                 # không đủ mốc chạm -> mốc = detect
    else:
        anchors = sorted(set(touch))
        if detected[0] < anchors[0]:
            anchors = [detected[0]] + anchors
        if detected[-1] > anchors[-1]:
            anchors = anchors + [detected[-1]]

    # Mặc định: chỉ hiện bóng ở frame detect trực tiếp (ngoài span mốc -> raw/None,
    # hết cảnh "bóng ma" đứng yên sau khi mất dấu).
    out = list(raw)
    max_gap = max(1, int(round(max_gap_s * fps)))
    for k in range(len(anchors) - 1):
        a0, a1 = anchors[k], anchors[k + 1]
        if a1 - a0 > max_gap:                    # mất dấu quá lâu -> không nội suy
            continue
        p0 = np.asarray(raw[a0], dtype=float)
        p1 = np.asarray(raw[a1], dtype=float)
        for f in range(a0, a1 + 1):
            t = (f - a0) / (a1 - a0)
            out[f] = (1.0 - t) * p0 + t * p1
    return out
