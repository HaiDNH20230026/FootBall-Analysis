"""Tốc độ tức thời (km/h) theo tracker_id, tính trên toạ độ sân (cm)."""
from collections import defaultdict

import numpy as np

from .config import SPEED_MAX_PLAYER_MS, SPEED_POS_SMOOTH, SPEED_WINDOW, SPEED_MAX_KMH


def compute_speeds(records, fps, max_player_ms=SPEED_MAX_PLAYER_MS, pos_smooth=SPEED_POS_SMOOTH,
                   spd_window=SPEED_WINDOW, max_kmh=SPEED_MAX_KMH):
    """Trả về list (dài bằng số frame) gồm dict {tracker_id: km/h}.

    - Làm sạch cú nhảy phi lý (id-swap) bằng max_player_ms.
    - Làm mượt vị trí theo từng track rồi tính vi sai trung tâm trên cửa sổ.
    """
    tracks = defaultdict(list)
    for f, r in enumerate(records):
        for k in range(len(r["tid"])):
            if r["isref"][k]:
                continue
            p = r["pitch"][k]
            if not np.isnan(p).any():
                tracks[int(r["tid"][k])].append((f, np.asarray(p, float)))

    def moving_avg(xy, w):
        if w <= 1 or len(xy) < 3:
            return xy
        ker = np.ones(w) / w
        return np.stack([np.convolve(xy[:, 0], ker, "same"),
                         np.convolve(xy[:, 1], ker, "same")], 1)

    speed_by_frame = [dict() for _ in records]
    for tid, seq in tracks.items():
        seq.sort()
        frames = [seq[0][0]]
        xy = [seq[0][1]]
        for f, p in seq[1:]:                         # bỏ jump phi lý (id-swap)
            df = f - frames[-1]
            if df > 0 and (np.linalg.norm(p - xy[-1]) / 100.0) / (df / fps) <= max_player_ms:
                frames.append(f)
                xy.append(p)
        frames = np.array(frames)
        xy = np.array(xy)
        if len(frames) < 2:
            continue
        xy_s = moving_avg(xy, pos_smooth)
        for j in range(len(frames)):                 # vi sai trung tâm
            j0 = max(0, j - spd_window)
            j1 = min(len(frames) - 1, j + spd_window)
            df = frames[j1] - frames[j0]
            if df > 0:
                d = np.linalg.norm(xy_s[j1] - xy_s[j0]) / 100.0
                speed_by_frame[frames[j]][tid] = min(d / (df / fps) * 3.6, max_kmh)
    return speed_by_frame
