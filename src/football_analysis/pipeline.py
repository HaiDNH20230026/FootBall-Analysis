"""Pipeline video 2 lượt (analyze -> hậu kỳ -> render) và inference ảnh đơn."""
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from .config import (CONF, NMS_THRESHOLD, BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID,
                     DEVICE, FIT_STRIDE, SWITCH_RATIO, BALL_TOUCH_PX, FPS)
from .detection import detect
from .tracking import BallTracker
from .team import TeamResolver, fit_team_classifier, resolve_goalkeepers_team_id
from .pitch import HomographySmoother
from .ball import smooth_ball_path
from .speed import compute_speeds
from .annotate import (ellipse_annotator, label_annotator, triangle_annotator,
                       build_radar, build_voronoi, compose_frame)


def analyze_video(source_path, player_model, field_model, team_resolver, max_frames=None):
    """Lượt 1: chạy model 1 lần, trả về (info, records) gọn nhẹ."""
    info = sv.VideoInfo.from_video_path(source_path)
    tracker = sv.ByteTrack()
    tracker.reset()
    ball_tracker = BallTracker()
    homo = HomographySmoother(field_model)
    records = []
    total = info.total_frames if max_frames is None else min(max_frames, info.total_frames)

    gen = sv.get_video_frames_generator(source_path)
    for i, frame in enumerate(tqdm(gen, total=total, desc="analyzing")):
        if max_frames is not None and i >= max_frames:
            break
        det = detect(player_model, frame, conf=CONF)

        ball = det[det.class_id == BALL_ID]
        ball_xyxy = ball_tracker.update(ball)

        others = det[det.class_id != BALL_ID].with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
        others = tracker.update_with_detections(detections=others)
        gk = others[others.class_id == GOALKEEPER_ID]
        pl = others[others.class_id == PLAYER_ID]
        rf = others[others.class_id == REFEREE_ID]
        if len(pl):
            pl.class_id = team_resolver.predict_players(frame, pl)
        if len(gk) and len(pl):
            gk.class_id = team_resolver.smooth_goalkeepers(gk, resolve_goalkeepers_team_id(pl, gk))
        if len(rf):
            rf.class_id = rf.class_id - 1

        transformer = homo.update(frame)

        parts = [d for d in (pl, gk, rf) if len(d)]
        if parts:
            merged = sv.Detections.merge(parts)
            xyxy, cid, tid = merged.xyxy, merged.class_id.astype(int), merged.tracker_id
            isref = np.concatenate([np.zeros(len(pl), bool), np.zeros(len(gk), bool),
                                    np.ones(len(rf), bool)])
            if transformer is not None:
                foot = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch = transformer.transform_points(points=foot)
            else:
                pitch = np.full((len(merged), 2), np.nan)
        else:
            xyxy = np.empty((0, 4)); cid = np.empty((0,), int); tid = np.empty((0,), int)
            isref = np.empty((0,), bool); pitch = np.empty((0, 2))

        ball_pitch = None
        if transformer is not None and ball_xyxy is not None:
            bc = np.array([[(ball_xyxy[0, 0] + ball_xyxy[0, 2]) / 2, ball_xyxy[0, 3]]])
            ball_pitch = transformer.transform_points(points=bc)[0]

        records.append(dict(xyxy=xyxy, cid=cid, tid=tid, isref=isref, pitch=pitch,
                            ball_box=(None if ball_xyxy is None else ball_xyxy[0]),
                            ball_pitch=ball_pitch))
    return info, records


def render_video(source_path, target_path, info, records, ball_pitch_s,
                 speed_by_frame, max_frames=None):
    """Lượt 2c: vẽ lại bằng dữ liệu đã hậu kỳ (KHÔNG infer lại)."""
    total = info.total_frames if max_frames is None else min(max_frames, info.total_frames)
    gen = sv.get_video_frames_generator(source_path)
    with sv.VideoSink(target_path, info) as sink:
        for f, frame in enumerate(tqdm(gen, total=total, desc="rendering")):
            if max_frames is not None and f >= max_frames:
                break
            r = records[f]
            annotated = frame.copy()

            if len(r["xyxy"]):
                det = sv.Detections(xyxy=r["xyxy"], class_id=r["cid"].astype(int),
                                    tracker_id=r["tid"])
                annotated = ellipse_annotator.annotate(scene=annotated, detections=det)
                labels = []
                for k, tid in enumerate(r["tid"]):
                    v = speed_by_frame[f].get(int(tid))
                    labels.append(f"{v:.0f} km/h" if (v is not None and not r["isref"][k])
                                  else f"#{int(tid)}")
                annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)

            if r["ball_box"] is not None:
                bd = sv.Detections(xyxy=sv.pad_boxes(xyxy=r["ball_box"][None, :], px=10),
                                   class_id=np.array([BALL_ID]))
                annotated = triangle_annotator.annotate(scene=annotated, detections=bd)

            radar = voronoi = None
            pitch = r["pitch"]
            if len(pitch):
                valid = ~np.isnan(pitch).any(axis=1)
                cid, isref = r["cid"], r["isref"]
                t0 = pitch[valid & (cid == 0) & (~isref)]
                t1 = pitch[valid & (cid == 1) & (~isref)]
                rfp = pitch[valid & isref]
                bp = ball_pitch_s[f]
                bp_arr = bp[None, :] if bp is not None else np.empty((0, 2))
                if valid.any() or bp is not None:
                    radar = build_radar(bp_arr, t0, t1, rfp)
                    voronoi = build_voronoi(t0, t1, bp_arr)

            sink.write_frame(compose_frame(annotated, radar=radar, voronoi=voronoi))
    print("✅ Đã lưu video:", target_path)
    return target_path


def run_video(source_path, target_path, player_model, field_model,
              max_frames=None, fit_stride=FIT_STRIDE, switch_ratio=SWITCH_RATIO,
              touch_px=BALL_TOUCH_PX, device=DEVICE):
    """Toàn bộ pipeline video: fit đội -> phân tích -> hậu kỳ -> vẽ."""
    classifier = fit_team_classifier(player_model, source_path, is_video=True,
                                     stride=fit_stride, device=device)
    info = sv.VideoInfo.from_video_path(source_path)
    resolver = TeamResolver(classifier, window=int(round(info.fps)), switch_ratio=switch_ratio)

    print("— Lượt 1: phân tích —")
    _, records = analyze_video(source_path, player_model, field_model, resolver,
                               max_frames=max_frames)
    print("— Lượt 2: hậu kỳ + vẽ —")
    ball_s = smooth_ball_path(records, info.fps, touch_px=touch_px)
    speeds = compute_speeds(records, info.fps)
    return render_video(source_path, target_path, info, records, ball_s, speeds,
                        max_frames=max_frames)


def run_image(image_path, player_model, field_model, device=DEVICE):
    """Inference 1 ảnh: annotate + radar + voronoi (không có tốc độ)."""
    frame = cv2.imread(image_path)
    classifier = fit_team_classifier(player_model, image_path, is_video=False, device=device)
    resolver = TeamResolver(classifier, window=1)
    tracker = sv.ByteTrack()
    tracker.reset()
    ball_tracker = BallTracker()

    _, records = _analyze_one(frame, player_model, field_model, resolver, tracker, ball_tracker)
    ball_s = smooth_ball_path(records, fps=float(FPS))
    speeds = compute_speeds(records, fps=float(FPS))
    # vẽ frame duy nhất
    info = sv.VideoInfo(width=frame.shape[1], height=frame.shape[0], fps=FPS, total_frames=1)
    r = records[0]
    annotated = frame.copy()
    if len(r["xyxy"]):
        det = sv.Detections(xyxy=r["xyxy"], class_id=r["cid"].astype(int), tracker_id=r["tid"])
        annotated = ellipse_annotator.annotate(scene=annotated, detections=det)
        labels = [f"#{int(t)}" for t in r["tid"]]
        annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)
    if r["ball_box"] is not None:
        bd = sv.Detections(xyxy=sv.pad_boxes(xyxy=r["ball_box"][None, :], px=10),
                           class_id=np.array([BALL_ID]))
        annotated = triangle_annotator.annotate(scene=annotated, detections=bd)
    radar = voronoi = None
    pitch = r["pitch"]
    if len(pitch):
        valid = ~np.isnan(pitch).any(axis=1)
        cid, isref = r["cid"], r["isref"]
        bp = ball_s[0]
        bp_arr = bp[None, :] if bp is not None else np.empty((0, 2))
        radar = build_radar(bp_arr, pitch[valid & (cid == 0) & (~isref)],
                            pitch[valid & (cid == 1) & (~isref)], pitch[valid & isref])
        voronoi = build_voronoi(pitch[valid & (cid == 0) & (~isref)],
                               pitch[valid & (cid == 1) & (~isref)], bp_arr)
    return compose_frame(annotated, radar=radar, voronoi=voronoi)


def _analyze_one(frame, player_model, field_model, resolver, tracker, ball_tracker):
    """Phân tích 1 frame (dùng cho ảnh đơn)."""
    homo = HomographySmoother(field_model)
    det = detect(player_model, frame, conf=CONF)
    ball = det[det.class_id == BALL_ID]
    ball_xyxy = ball_tracker.update(ball)
    others = det[det.class_id != BALL_ID].with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
    others = tracker.update_with_detections(detections=others)
    gk = others[others.class_id == GOALKEEPER_ID]
    pl = others[others.class_id == PLAYER_ID]
    rf = others[others.class_id == REFEREE_ID]
    if len(pl):
        pl.class_id = resolver.predict_players(frame, pl)
    if len(gk) and len(pl):
        gk.class_id = resolver.smooth_goalkeepers(gk, resolve_goalkeepers_team_id(pl, gk))
    if len(rf):
        rf.class_id = rf.class_id - 1
    transformer = homo.update(frame)
    parts = [d for d in (pl, gk, rf) if len(d)]
    if parts:
        merged = sv.Detections.merge(parts)
        xyxy, cid, tid = merged.xyxy, merged.class_id.astype(int), merged.tracker_id
        isref = np.concatenate([np.zeros(len(pl), bool), np.zeros(len(gk), bool),
                                np.ones(len(rf), bool)])
        if transformer is not None:
            foot = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch = transformer.transform_points(points=foot)
        else:
            pitch = np.full((len(merged), 2), np.nan)
    else:
        xyxy = np.empty((0, 4)); cid = np.empty((0,), int); tid = np.empty((0,), int)
        isref = np.empty((0,), bool); pitch = np.empty((0, 2))
    ball_pitch = None
    if transformer is not None and ball_xyxy is not None:
        bc = np.array([[(ball_xyxy[0, 0] + ball_xyxy[0, 2]) / 2, ball_xyxy[0, 3]]])
        ball_pitch = transformer.transform_points(points=bc)[0]
    rec = dict(xyxy=xyxy, cid=cid, tid=tid, isref=isref, pitch=pitch,
               ball_box=(None if ball_xyxy is None else ball_xyxy[0]), ball_pitch=ball_pitch)
    return None, [rec]
