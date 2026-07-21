"""Pipeline video 2 lượt (analyze -> hậu kỳ -> render) và inference ảnh đơn.

Lượt 1 (analyze_video) chỉ thu dữ liệu thô: detection, tracking, đội, quan sát
keypoint sân của mỗi frame, và chuyển động camera liên khung (optical flow). Toạ độ
sân (pitch) được tính ở LƯỢT 2 — sau khi chuỗi keypoint đã được làm mượt CÓ TÂM
theo thời gian và fit lại homography từng frame (xem pitch.py); frame model mất sân
được bắc cầu bằng dead-reckoning quang học nên bản đồ 2D bám camera cả khi lia.
"""
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from .config import (CONF, NMS_THRESHOLD, BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID,
                     DEVICE, FIT_STRIDE, SWITCH_RATIO, BALL_TOUCH_PX, FPS,
                     HOMOGRAPHY_SMOOTH_WINDOW, SHOW_RADAR, SHOW_VORONOI, SHOW_BALL, OUT_FPS)
from .detection import detect
from .tracking import BallTracker
from .team import TeamResolver, fit_team_classifier, resolve_goalkeepers_team_id
from .pitch import (HomographyEstimator, build_transformers, transformer_from_obs,
                    estimate_frame_motion, _smooth_keypoint_series)
from .ball import smooth_ball_path
from .speed import compute_speeds
from .annotate import (ellipse_annotator, label_annotator, triangle_annotator,
                       build_radar, build_voronoi, compose_frame)


def analyze_video(source_path, player_model, field_model, team_resolver, max_frames=None):
    """Lượt 1: chạy model 1 lần, trả về (info, records). Chưa tính toạ độ sân."""
    info = sv.VideoInfo.from_video_path(source_path)
    tracker = sv.ByteTrack()
    tracker.reset()
    ball_tracker = BallTracker()
    homo = HomographyEstimator(field_model)
    records = []
    prev_gray = None
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

        H = homo.update(frame)                   # quan sát keypoint sân {xy, conf} hoặc None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dH = estimate_frame_motion(prev_gray, gray) if prev_gray is not None else None
        prev_gray = gray

        parts = [d for d in (pl, gk, rf) if len(d)]
        if parts:
            merged = sv.Detections.merge(parts)
            xyxy, cid, tid = merged.xyxy, merged.class_id.astype(int), merged.tracker_id
            isref = np.concatenate([np.zeros(len(pl), bool), np.zeros(len(gk), bool),
                                    np.ones(len(rf), bool)])
        else:
            xyxy = np.empty((0, 4)); cid = np.empty((0,), int); tid = np.empty((0,), int)
            isref = np.empty((0,), bool)

        records.append(dict(xyxy=xyxy, cid=cid, tid=tid, isref=isref, H=H, dH=dH,
                            ball_box=(None if ball_xyxy is None else ball_xyxy[0]),
                            pitch=None, ball_pitch=None))
    return info, records


def apply_pitch_coords(records, transformers):
    """Lượt 2a: dùng homography ĐÃ làm mượt, tính toạ độ sân cho người + bóng."""
    for f, r in enumerate(records):
        tf = transformers[f] if f < len(transformers) else None
        n = len(r["xyxy"])
        if tf is not None and n:
            foot = np.column_stack([(r["xyxy"][:, 0] + r["xyxy"][:, 2]) / 2.0,
                                    r["xyxy"][:, 3]]).astype(np.float32)
            r["pitch"] = tf.transform_points(points=foot)
        else:
            r["pitch"] = np.full((n, 2), np.nan)

        bb = r["ball_box"]
        if tf is not None and bb is not None:
            bc = np.array([[(bb[0] + bb[2]) / 2.0, bb[3]]], dtype=np.float32)
            r["ball_pitch"] = tf.transform_points(points=bc)[0]
        else:
            r["ball_pitch"] = None
    return records


def smooth_radar_tracks(records, fps, max_gap_s=1.0, window=7):
    """Lượt 2b: chống flicker cho radar/Voronoi — xử lý CHUỖI VỊ TRÍ SÂN của từng
    tracker_id: lấp khoảng trống ngắn (detection chớp tắt vài frame) bằng nội suy
    và làm mượt CÓ TÂM (median + Savitzky-Golay, tái dùng bộ lọc chuỗi keypoint).
    Chỉ nội suy TRONG khoảng track thực sự được thấy — track mất hẳn thì dot biến
    mất, không vẽ "bóng ma". Đội/vai trò lấy theo đa số của track. Kết quả ghi vào
    r["radar_xy"/"radar_cid"/"radar_isref"]; khung hình gốc (ellipse, nhãn) giữ
    nguyên theo detection thật."""
    F = len(records)
    tracks = {}
    for f, r in enumerate(records):
        pitch = r["pitch"]
        if pitch is None or not len(pitch):
            continue
        for k, tid in enumerate(r["tid"]):
            if np.isnan(pitch[k]).any():
                continue
            t = tracks.setdefault(int(tid), {"f": [], "xy": [], "cid": [], "isref": []})
            t["f"].append(f)
            t["xy"].append(pitch[k])
            t["cid"].append(int(r["cid"][k]))
            t["isref"].append(bool(r["isref"][k]))

    per_frame = [[] for _ in range(F)]
    max_gap = max(1, int(round(max_gap_s * float(fps))))
    for t in tracks.values():
        fs = np.asarray(t["f"])
        f0, f1 = int(fs[0]), int(fs[-1])
        series = np.full((f1 - f0 + 1, 2), np.nan)
        series[fs - f0] = np.asarray(t["xy"], dtype=float)
        series = _smooth_keypoint_series(series, window, 5, max_gap)
        cid = int(np.bincount(t["cid"]).argmax())
        isref = bool(np.mean(t["isref"]) > 0.5)
        for i in range(len(series)):
            if np.isfinite(series[i, 0]):
                per_frame[f0 + i].append((series[i], cid, isref))

    for f, r in enumerate(records):
        pts = per_frame[f]
        r["radar_xy"] = np.array([p[0] for p in pts], dtype=np.float32) if pts else np.empty((0, 2), np.float32)
        r["radar_cid"] = np.array([p[1] for p in pts], dtype=int)
        r["radar_isref"] = np.array([p[2] for p in pts], dtype=bool)
    return records


def render_video(source_path, target_path, info, records, ball_pitch_s, speed_by_frame,
                 max_frames=None, show_radar=SHOW_RADAR, show_voronoi=SHOW_VORONOI,
                 show_ball=SHOW_BALL, out_fps=OUT_FPS):
    """Lượt 2c: vẽ lại bằng dữ liệu đã hậu kỳ (KHÔNG infer lại).

    show_radar / show_voronoi / show_ball: bật/tắt từng overlay.
    out_fps: FPS video kết quả (None = giữ fps nguồn). Khác fps nguồn -> lấy/nhân
    mẫu frame theo thời gian, giữ nguyên thời lượng video.
    """
    total = info.total_frames if max_frames is None else min(max_frames, info.total_frames)
    src_fps = float(info.fps) or float(FPS)
    o_fps = src_fps if out_fps in (None, 0) else float(out_fps)
    ratio = o_fps / src_fps
    info_out = sv.VideoInfo(width=info.width, height=info.height, fps=o_fps,
                            total_frames=int(round(total * ratio)))

    gen = sv.get_video_frames_generator(source_path)
    with sv.VideoSink(target_path, info_out) as sink:
        emitted = 0
        for f, frame in enumerate(tqdm(gen, total=total, desc="rendering")):
            if max_frames is not None and f >= max_frames:
                break
            target = int(round((f + 1) * ratio))
            if target <= emitted:                # frame bị bỏ khi giảm fps
                continue

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

            if show_ball and r["ball_box"] is not None:
                bd = sv.Detections(xyxy=sv.pad_boxes(xyxy=r["ball_box"][None, :], px=10),
                                   class_id=np.array([BALL_ID]))
                annotated = triangle_annotator.annotate(scene=annotated, detections=bd)

            radar = voronoi = None
            if r.get("radar_xy") is not None:        # chuỗi đã chống flicker (video 2 lượt)
                pitch, cid, isref = r["radar_xy"], r["radar_cid"], r["radar_isref"]
            else:                                    # fallback: detection thô của frame
                pitch, cid, isref = r["pitch"], r["cid"], r["isref"]
            if (show_radar or show_voronoi) and pitch is not None and len(pitch):
                valid = ~np.isnan(pitch).any(axis=1)
                t0 = pitch[valid & (cid == 0) & (~isref)]
                t1 = pitch[valid & (cid == 1) & (~isref)]
                rfp = pitch[valid & isref]
                bp = ball_pitch_s[f] if show_ball else None
                bp_arr = bp[None, :] if bp is not None else np.empty((0, 2))
                if valid.any() or bp is not None:
                    if show_radar:
                        radar = build_radar(bp_arr, t0, t1, rfp)
                    if show_voronoi:
                        voronoi = build_voronoi(t0, t1, bp_arr)

            composed = compose_frame(annotated, radar=radar, voronoi=voronoi)
            while emitted < target:              # frame được lặp khi tăng fps
                sink.write_frame(composed)
                emitted += 1
    print("✅ Đã lưu video:", target_path)
    return target_path


def run_video(source_path, target_path, player_model, field_model,
              max_frames=None, fit_stride=FIT_STRIDE, switch_ratio=SWITCH_RATIO,
              touch_px=BALL_TOUCH_PX, device=DEVICE, show_radar=SHOW_RADAR,
              show_voronoi=SHOW_VORONOI, show_ball=SHOW_BALL, out_fps=OUT_FPS,
              smooth_window=HOMOGRAPHY_SMOOTH_WINDOW):
    """Toàn bộ pipeline video: fit đội -> phân tích -> làm mượt homography -> vẽ."""
    classifier = fit_team_classifier(player_model, source_path, is_video=True,
                                     stride=fit_stride, device=device)
    info = sv.VideoInfo.from_video_path(source_path)
    resolver = TeamResolver(classifier, window=int(round(info.fps)), switch_ratio=switch_ratio)

    print("— Lượt 1: phân tích —")
    _, records = analyze_video(source_path, player_model, field_model, resolver,
                               max_frames=max_frames)
    print("— Lượt 2: làm mượt homography + hậu kỳ + vẽ —")
    transformers = build_transformers([r["H"] for r in records],
                                      dh_list=[r.get("dH") for r in records],
                                      window=smooth_window)
    apply_pitch_coords(records, transformers)
    smooth_radar_tracks(records, info.fps)
    ball_s = smooth_ball_path(records, info.fps, touch_px=touch_px)
    speeds = compute_speeds(records, info.fps)
    return render_video(source_path, target_path, info, records, ball_s, speeds,
                        max_frames=max_frames, show_radar=show_radar,
                        show_voronoi=show_voronoi, show_ball=show_ball, out_fps=out_fps)


def run_image(image_path, player_model, field_model, device=DEVICE,
              show_radar=SHOW_RADAR, show_voronoi=SHOW_VORONOI, show_ball=SHOW_BALL):
    """Inference 1 ảnh: annotate + radar + voronoi (không có tốc độ, không two-pass)."""
    frame = cv2.imread(image_path)
    classifier = fit_team_classifier(player_model, image_path, is_video=False, device=device)
    resolver = TeamResolver(classifier, window=1)
    tracker = sv.ByteTrack()
    tracker.reset()
    ball_tracker = BallTracker()

    _, records = _analyze_one(frame, player_model, field_model, resolver, tracker, ball_tracker)
    ball_s = smooth_ball_path(records, fps=float(FPS))
    speeds = compute_speeds(records, fps=float(FPS))
    info = sv.VideoInfo(width=frame.shape[1], height=frame.shape[0], fps=FPS, total_frames=1)
    r = records[0]
    annotated = frame.copy()
    if len(r["xyxy"]):
        det = sv.Detections(xyxy=r["xyxy"], class_id=r["cid"].astype(int), tracker_id=r["tid"])
        annotated = ellipse_annotator.annotate(scene=annotated, detections=det)
        labels = [f"#{int(t)}" for t in r["tid"]]
        annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)
    if show_ball and r["ball_box"] is not None:
        bd = sv.Detections(xyxy=sv.pad_boxes(xyxy=r["ball_box"][None, :], px=10),
                           class_id=np.array([BALL_ID]))
        annotated = triangle_annotator.annotate(scene=annotated, detections=bd)
    radar = voronoi = None
    pitch = r["pitch"]
    if (show_radar or show_voronoi) and len(pitch):
        valid = ~np.isnan(pitch).any(axis=1)
        cid, isref = r["cid"], r["isref"]
        bp = ball_s[0] if show_ball else None
        bp_arr = bp[None, :] if bp is not None else np.empty((0, 2))
        t0 = pitch[valid & (cid == 0) & (~isref)]
        t1 = pitch[valid & (cid == 1) & (~isref)]
        if show_radar:
            radar = build_radar(bp_arr, t0, t1, pitch[valid & isref])
        if show_voronoi:
            voronoi = build_voronoi(t0, t1, bp_arr)
    return compose_frame(annotated, radar=radar, voronoi=voronoi)


def _analyze_one(frame, player_model, field_model, resolver, tracker, ball_tracker):
    """Phân tích 1 frame (dùng cho ảnh đơn) — homography fit trực tiếp, không làm mượt."""
    homo = HomographyEstimator(field_model)
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
    transformer = transformer_from_obs(homo.update(frame))
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
        ball_pitch = transformer.transform_points(points=bc.astype(np.float32))[0]
    rec = dict(xyxy=xyxy, cid=cid, tid=tid, isref=isref, pitch=pitch,
               ball_box=(None if ball_xyxy is None else ball_xyxy[0]), ball_pitch=ball_pitch)
    return None, [rec]
