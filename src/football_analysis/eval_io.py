"""Nạp dữ liệu SoccerNet + chạy pipeline trên THƯ MỤC ẢNH (img1/) để đánh giá.

SoccerNet-GSR và SoccerNet-Tracking đều lưu mỗi chuỗi 30s dưới dạng thư mục ảnh
(img1/000001.jpg ...) chứ không phải video, nên ở đây có phiên bản chạy pipeline
đọc từ thư mục (khác pipeline.py đọc từ file video).

⚠️ CHƯA KIỂM THỬ trên file SoccerNet thật (máy chưa có dataset). Tên trường JSON
bám theo format trong paper (Fig. S3) và MOT-Challenge chuẩn; nếu bản dataset của
bạn khác, chỉnh `load_gsr_labels` / `load_mot_gt`. Mọi hàm đều in chẩn đoán để
bạn đối chiếu.
"""
import configparser
import glob
import json
import os

import cv2
import numpy as np
import supervision as sv

from .config import (CONF, NMS_THRESHOLD, BALL_ID, GOALKEEPER_ID, PLAYER_ID,
                     REFEREE_ID, PITCH)
from .detection import detect
from .pitch import HomographyEstimator, build_transformers, estimate_frame_motion

# Template sân của thư viện `sports` (cm, gốc ở GÓC).
_TEMPLATE_L = float(PITCH.length)   # 12000 cm = 120 m
_TEMPLATE_W = float(PITCH.width)    # 7000  cm = 70 m
# Sân thật SoccerNet-GSR (mét, gốc ở TÂM).
_SNGS_L = 105.0
_SNGS_W = 68.0


def iter_image_paths(img_dir):
    """Danh sách đường dẫn ảnh trong thư mục, SẮP THEO TÊN (frame theo thứ tự)."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(img_dir, e))
    return sorted(paths)


def template_cm_to_sngs_m(xy_cm, flip_x=False, flip_y=False):
    """Quy đổi toạ độ sân: template(cm, gốc góc, 120x70) -> SNGS(m, gốc tâm, 105x68).

    flip_x/flip_y: đảo trục khi hệ quy chiếu của model keypoint ngược chiều SNGS
    (xem chẩn đoán trong eval_gsr.py — nếu sai lệch median hàng chục mét, thử đảo).
    """
    xy_cm = np.asarray(xy_cm, float).reshape(-1, 2)
    if len(xy_cm) == 0:
        return xy_cm
    u = xy_cm[:, 0] / _TEMPLATE_L            # 0..1 dọc chiều dài
    v = xy_cm[:, 1] / _TEMPLATE_W            # 0..1 dọc chiều rộng
    x = (u - 0.5) * _SNGS_L
    y = (v - 0.5) * _SNGS_W
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    return np.column_stack([x, y]).astype(np.float32)


# ---------------------------------------------------------------------------
# Chạy pipeline trên thư mục ảnh
# ---------------------------------------------------------------------------
def _split_roles(others):
    """Tách detection người theo class GỐC (trước khi ghi đè team) -> giữ vai trò."""
    gk = others[others.class_id == GOALKEEPER_ID]
    pl = others[others.class_id == PLAYER_ID]
    rf = others[others.class_id == REFEREE_ID]
    return pl, gk, rf


def run_gsr_on_folder(img_dir, player_model, field_model, resolver,
                      smooth_window=None, verbose=True):
    """Chạy pipeline two-pass trên thư mục ảnh, trả (filenames, records).

    Mỗi record: dict {file, box(N,4), tid(N), role(N str), team_cluster(N int/-1),
    pitch_cm(N,2)}. role ∈ {player, goalkeeper, referee}. team_cluster: 0/1 cho
    player+goalkeeper, -1 cho referee. pitch_cm: toạ độ template (cm) — quy đổi
    sang SNGS ở tầng gọi.
    """
    from .config import HOMOGRAPHY_SMOOTH_WINDOW
    sw = HOMOGRAPHY_SMOOTH_WINDOW if smooth_window is None else smooth_window
    paths = iter_image_paths(img_dir)
    if not paths:
        raise FileNotFoundError(f"Không thấy ảnh trong {img_dir}")
    tracker = sv.ByteTrack(); tracker.reset()
    homo = HomographyEstimator(field_model)
    recs = []; prev_gray = None
    H_list = []; dH_list = []

    for path in paths:
        frame = cv2.imread(path)
        det = detect(player_model, frame, conf=CONF)
        others = det[det.class_id != BALL_ID].with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
        others = tracker.update_with_detections(detections=others)
        pl, gk, rf = _split_roles(others)
        team_pl = resolver.predict_players(frame, pl) if len(pl) else np.array([], int)
        team_gk = (resolver.smooth_goalkeepers(gk, _nearest_team(pl, gk, team_pl))
                   if len(gk) and len(pl) else np.full(len(gk), -1, int))

        boxes, tids, roles, teams = [], [], [], []
        for d, role, team in ((pl, "player", team_pl), (gk, "goalkeeper", team_gk),
                              (rf, "referee", None)):
            for k in range(len(d)):
                boxes.append(d.xyxy[k]); tids.append(int(d.tracker_id[k]))
                roles.append(role)
                teams.append(-1 if team is None else int(team[k]))

        H = homo.update(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dH = estimate_frame_motion(prev_gray, gray) if prev_gray is not None else None
        prev_gray = gray
        H_list.append(H); dH_list.append(dH)
        recs.append(dict(file=os.path.basename(path),
                         box=np.array(boxes, float).reshape(-1, 4),
                         tid=np.array(tids, int),
                         role=np.array(roles, dtype=object),
                         team_cluster=np.array(teams, int)))

    transformers = build_transformers(H_list, dh_list=dH_list, window=sw)
    for f, r in enumerate(recs):
        tf = transformers[f] if f < len(transformers) else None
        n = len(r["box"])
        if tf is not None and n:
            foot = np.column_stack([(r["box"][:, 0] + r["box"][:, 2]) / 2.0,
                                    r["box"][:, 3]]).astype(np.float32)
            r["pitch_cm"] = tf.transform_points(points=foot)
        else:
            r["pitch_cm"] = np.full((n, 2), np.nan)
    if verbose:
        n_pos = sum(int(np.isfinite(r["pitch_cm"][:, 0]).any()) for r in recs)
        print(f"  pipeline: {len(recs)} frame, có toạ độ sân ở {n_pos} frame")
    return [r["file"] for r in recs], recs


def _nearest_team(players, goalkeepers, team_pl):
    """Gán thủ môn về đội có centroid gần hơn (ảnh) — bản rút gọn của team.py."""
    if not len(goalkeepers) or not len(players):
        return np.full(len(goalkeepers), -1, int)
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    c0 = pl_xy[team_pl == 0].mean(axis=0) if (team_pl == 0).any() else pl_xy.mean(0)
    c1 = pl_xy[team_pl == 1].mean(axis=0) if (team_pl == 1).any() else pl_xy.mean(0)
    return np.array([0 if np.linalg.norm(xy - c0) < np.linalg.norm(xy - c1) else 1
                     for xy in gk_xy], int)


def fit_resolver_on_folder(player_model, img_dir, stride, device, switch_ratio, fps):
    """Fit TeamClassifier trên crop thân áo lấy từ thư mục ảnh (thay cho video)."""
    from sports.common.team import TeamClassifier
    from .team import torso_crop, TeamResolver
    paths = iter_image_paths(img_dir)[::max(1, stride)]
    crops = []
    for path in paths:
        frame = cv2.imread(path)
        det = detect(player_model, frame, conf=CONF)
        det = det[det.class_id == PLAYER_ID]
        crops += [torso_crop(frame, b) for b in det.xyxy]
    clf = TeamClassifier(device=device); clf.fit(crops)
    return TeamResolver(clf, window=int(round(fps)), switch_ratio=switch_ratio)


def run_tracking_on_folder(img_dir, player_model, keep_classes=None, verbose=True):
    """Chạy detection + ByteTrack (chỉ toạ độ ẢNH) trên thư mục — cho eval MOT.

    keep_classes: set class_id giữ lại (mặc định người: gk/player/referee, bỏ bóng).
    Trả (filenames, per_frame) với per_frame[t] = {"ids":[...], "boxes":(n,4)}.
    """
    if keep_classes is None:
        keep_classes = {GOALKEEPER_ID, PLAYER_ID, REFEREE_ID}
    paths = iter_image_paths(img_dir)
    if not paths:
        raise FileNotFoundError(f"Không thấy ảnh trong {img_dir}")
    tracker = sv.ByteTrack(); tracker.reset()
    files, per_frame = [], []
    for path in paths:
        frame = cv2.imread(path)
        det = detect(player_model, frame, conf=CONF)
        keep = np.isin(det.class_id, list(keep_classes))
        det = det[keep].with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
        det = tracker.update_with_detections(detections=det)
        files.append(os.path.basename(path))
        per_frame.append({"ids": [int(t) for t in det.tracker_id],
                          "boxes": det.xyxy.reshape(-1, 4).astype(float)})
    if verbose:
        print(f"  tracking: {len(per_frame)} frame, "
              f"{sum(len(p['ids']) for p in per_frame)} detection")
    return files, per_frame


# ---------------------------------------------------------------------------
# Nạp ground-truth
# ---------------------------------------------------------------------------
_GSR_ROLES = {"player", "goalkeeper", "referee"}


def load_gsr_labels(json_path):
    """Nạp Labels-GameState.json (SoccerNet-GSR). Trả dict:
        {file_name -> [ {id, xy(mét, gốc tâm), role, team|None}, ... ]}
    Chỉ giữ role ∈ {player, goalkeeper, referee} (bỏ 'other' và bóng, đúng như paper).
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    id_to_file = {img["image_id"]: img["file_name"] for img in data["images"]}
    out = {img["file_name"]: [] for img in data["images"]}
    for a in data.get("annotations", []):
        if a.get("supercategory") != "object":
            continue
        attr = a.get("attributes", {}) or {}
        role = attr.get("role")
        if role not in _GSR_ROLES:
            continue
        bp = a.get("bbox_pitch")
        if not bp or bp.get("x_bottom_middle") is None:
            continue
        fn = id_to_file.get(a["image_id"])
        if fn is None:
            continue
        team = attr.get("team") if role in ("player", "goalkeeper") else None
        out[fn].append(dict(id=int(a["track_id"]),
                            xy=(float(bp["x_bottom_middle"]), float(bp["y_bottom_middle"])),
                            role=role, team=team))
    return out


def load_mot_gt(seq_dir, keep_classes=None):
    """Nạp ground-truth MOT-Challenge (SoccerNet-Tracking): gt/gt.txt + seqinfo.ini.

    gt.txt: frame,id,x,y,w,h,conf,class,visibility (x,y = góc trên-trái, 1-index frame).
    keep_classes: nếu set, chỉ giữ dòng có class ∈ tập này (tuỳ phiên bản dataset;
    None = giữ tất cả). Trả (ordered_files, per_frame) khớp thứ tự run_tracking.
    """
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Không thấy {gt_path}")
    ini = os.path.join(seq_dir, "seqinfo.ini")
    seq_len = None
    if os.path.exists(ini):
        cfg = configparser.ConfigParser(); cfg.read(ini)
        if cfg.has_option("Sequence", "seqLength"):
            seq_len = int(cfg.get("Sequence", "seqLength"))
    rows = np.loadtxt(gt_path, delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    max_f = int(rows[:, 0].max())
    F = seq_len or max_f
    per_frame = [{"ids": [], "boxes": []} for _ in range(F)]
    for r in rows:
        fr = int(r[0]) - 1
        if fr < 0 or fr >= F:
            continue
        if keep_classes is not None and len(r) > 7 and int(r[7]) not in keep_classes:
            continue
        if len(r) > 6 and r[6] == 0:          # conf/flag=0 -> bỏ (chuẩn MOT)
            continue
        x, y, w, h = r[2], r[3], r[4], r[5]
        per_frame[fr]["ids"].append(int(r[1]))
        per_frame[fr]["boxes"].append([x, y, x + w, y + h])
    for p in per_frame:
        p["boxes"] = np.array(p["boxes"], float).reshape(-1, 4)
    img_dir = os.path.join(seq_dir, "img1")
    files = [os.path.basename(p) for p in iter_image_paths(img_dir)] if os.path.isdir(img_dir) else []
    return files, per_frame
