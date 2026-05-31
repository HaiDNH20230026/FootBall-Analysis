"""Phân đội (SigLIP + UMAP + KMeans), chống flicker theo tracker_id, gán thủ môn."""
from collections import defaultdict, deque, Counter

import cv2
import numpy as np
import supervision as sv
from sports.common.team import TeamClassifier

from .config import CONF, PLAYER_ID, DEVICE, FIT_STRIDE, SWITCH_RATIO
from .detection import detect


def torso_crop(frame, xyxy, top=0.0, bottom=0.55):
    """Cắt vùng thân áo (nửa trên box) — đặc trưng tốt hơn để phân đội."""
    x1, y1, x2, y2 = xyxy
    h = y2 - y1
    return sv.crop_image(frame, [x1, y1 + h * top, x2, y1 + h * bottom])


def resolve_goalkeepers_team_id(players, goalkeepers):
    """Gán thủ môn về đội có centroid gần hơn (trong toạ độ ảnh)."""
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team0 = pl_xy[players.class_id == 0].mean(axis=0)
    team1 = pl_xy[players.class_id == 1].mean(axis=0)
    out = []
    for xy in gk_xy:
        out.append(0 if np.linalg.norm(xy - team0) < np.linalg.norm(xy - team1) else 1)
    return np.array(out)


def fit_team_classifier(player_model, source_path, is_video, stride=FIT_STRIDE, device=DEVICE):
    """Fit TeamClassifier trên crop thân áo lấy từ nguồn đầu vào."""
    crops = []
    if is_video:
        frames = sv.get_video_frames_generator(source_path, stride=stride)
    else:
        frames = [cv2.imread(source_path)]

    for frame in frames:
        det = detect(player_model, frame, conf=CONF)
        det = det[det.class_id == PLAYER_ID]
        crops += [torso_crop(frame, xyxy) for xyxy in det.xyxy]

    classifier = TeamClassifier(device=device)
    classifier.fit(crops)
    return classifier


class TeamResolver:
    """Bọc TeamClassifier, làm mượt nhãn đội theo tracker_id để chống flicker.

    window: số frame gần nhất dùng bỏ phiếu.
    switch_ratio: hysteresis — chỉ đổi đội khi đội mới chiếm > tỉ lệ này.
    """

    def __init__(self, classifier, window=30, crop_fn=torso_crop, switch_ratio=SWITCH_RATIO):
        self.clf = classifier
        self.window = window
        self.crop_fn = crop_fn
        self.switch_ratio = switch_ratio
        self.history = defaultdict(lambda: deque(maxlen=window))
        self.last = {}

    def _vote(self, tracker_ids, raw):
        out = []
        for tid, p in zip(tracker_ids, raw):
            tid = int(tid)
            h = self.history[tid]
            h.append(int(p))
            label, count = Counter(h).most_common(1)[0]
            prev = self.last.get(tid, label)
            if label != prev and count / len(h) < self.switch_ratio:
                label = prev
            self.last[tid] = label
            out.append(label)
        return np.array(out, dtype=int)

    def predict_players(self, frame, players):
        if not len(players):
            return np.array([], dtype=int)
        raw = self.clf.predict([self.crop_fn(frame, b) for b in players.xyxy])
        return self._vote(players.tracker_id, raw)

    def smooth_goalkeepers(self, goalkeepers, resolved):
        if not len(goalkeepers):
            return np.array([], dtype=int)
        return self._vote(goalkeepers.tracker_id, resolved)
