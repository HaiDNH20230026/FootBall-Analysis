"""Pitch localization: fit LS toàn bộ keypoint + cắt outlier thô, làm mượt CHUỖI
KEYPOINT theo thời gian (có tâm) rồi fit lại homography mỗi frame.

Nguyên tắc rút ra sau nhiều vòng thử nghiệm: LÀM MƯỢT PHÉP ĐO (vị trí keypoint
trong ảnh — luôn nằm trong khung hình), ĐỪNG làm mượt MÔ HÌNH (ma trận H, tham
số camera). Mọi cách áp mô hình camera lý tưởng lên footage thật đều dính sai số
hệ thống vài mét (méo ống kính, template sân 120x70 khác sân thật ~105x68, điểm
chính lệch tâm) — nặng nhất ở vùng xa (biên trên khung hình). Homography fit
trực tiếp từ keypoint quan sát "nuốt" được toàn bộ các sai lệch đó, nên chấm
trên minimap bám vạch sân đúng như chất lượng model keypoint cho phép.

Fit per-frame thuần chỉ hỏng vì 3 lý do, xử lý từng cái:
  a. Keypoint nhiễu đơn lẻ  -> fit LS trên TOÀN BỘ điểm rồi cắt outlier lệch xa
     đồng thuận toàn cục (xem _fit_frame_H) + lọc trung vị theo thời gian.
     KHÔNG dùng cv2.RANSAC: với nhiễu keypoint thật 15-35px, RANSAC từng khoá vào
     cụm 5-6 điểm sát nhau (cùng vòng cấm) tự-nhất-quán rồi vứt các điểm trải
     rộng -> H văng 4-23m ở phần sân còn lại, gate reproj không bắt được (đo trên
     chính cụm đó), và hold-last đóng băng H hỏng suốt lúc model mất sân.
  b. Rung nhỏ frame-to-frame -> Savitzky-Golay CÓ TÂM trên từng chuỗi keypoint
     (đa thức bậc 2 bám được chuyển động lia nhanh, không trễ như EMA nhân quả;
     khả thi nhờ kiến trúc two-pass).
  c. Frame thiếu điểm (blur khi lia...) -> GIỮ NGUYÊN homography hợp lệ gần nhất
     (hold-last, đúng hành vi notebook gốc). KHÔNG nội suy ma trận xuyên khoảng
     trống dài — thực nghiệm cho thấy ma trận nội suy giữa hai góc nhìn khác nhau
     làm cụm điểm méo dần; đứng yên rồi nhảy sang H mới ít khó chịu hơn nhiều.
"""
import cv2
import numpy as np
import supervision as sv

from .config import (CONF, KP_CONF, PITCH, HOMOGRAPHY_RANSAC_PX,
                     HOMOGRAPHY_MIN_INLIERS, HOMOGRAPHY_MAX_REPROJ_PX,
                     HOMOGRAPHY_SMOOTH_WINDOW, HOMOGRAPHY_MEDIAN_WINDOW)

_PITCH_VERTS = np.array(PITCH.vertices, dtype=np.float32)   # (32, 2) toạ độ sân (cm)
_N = len(_PITCH_VERTS)


def _keypoints_from_result(result):
    """Convert keypoint từ cả 2 nguồn: Ultralytics YOLO-pose (local) và Roboflow."""
    if hasattr(result, "keypoints"):              # Ultralytics Results (.pt local)
        return sv.KeyPoints.from_ultralytics(result)
    return sv.KeyPoints.from_inference(result)    # Roboflow inference


class _MatrixTransformer:
    """Bọc một ma trận homography ảnh->sân, interface giống ViewTransformer."""

    def __init__(self, m):
        self.m = np.asarray(m, dtype=np.float32)

    def transform_points(self, points):
        points = np.asarray(points)
        if points.size == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.m).reshape(-1, 2).astype(np.float32)


class HomographyEstimator:
    """Quan sát keypoint sân của 1 frame.

    update(frame) -> dict {"xy": (32,2) toạ độ ảnh, "conf": (32,)} hoặc None nếu
    model không bắt được sân. Việc lọc nhiễu/làm mượt/fit H làm ở lượt 2
    (build_transformers) khi đã có toàn bộ chuỗi thời gian.
    """

    def __init__(self, field_model):
        self.model = field_model

    def update(self, frame):
        result = self.model.infer(frame, confidence=CONF)[0]
        kp = _keypoints_from_result(result)
        if len(kp.xy) == 0 or kp.xy[0].shape[0] != _N:
            return None
        conf = kp.confidence[0] if kp.confidence is not None else np.ones(_N)
        return {"xy": kp.xy[0].astype(np.float32), "conf": np.asarray(conf, dtype=np.float32)}


def _fit_frame_H(xy, vis, ransac_px=HOMOGRAPHY_RANSAC_PX,
                 min_inliers=HOMOGRAPHY_MIN_INLIERS, max_reproj_px=HOMOGRAPHY_MAX_REPROJ_PX):
    """Fit H sân->ảnh cho 1 frame: least-squares trên TOÀN BỘ keypoint, rồi cắt
    outlier thô so với fit toàn cục và fit lại. Trả (H, keep_mask_32) hoặc (None, None).

    KHÔNG dùng cv2.RANSAC ở đây: nhiễu keypoint thật (~15-35px) lớn hơn ngưỡng
    hợp lý, nên RANSAC hay khoá vào một cụm điểm sát nhau tự-nhất-quán (vd 5 điểm
    cùng vòng cấm) và vứt các điểm trải rộng — H ngoại suy văng hàng chục mét ở
    phần sân còn lại mà gate reprojection không bắt được (đo trên chính cụm đó).
    Fit LS toàn bộ điểm giữ baseline rộng nên không thể suy biến kiểu ấy; outlier
    chỉ bị loại khi lệch >max(ransac_px*2, 3*median) so với đồng thuận toàn cục.
    """
    if int(vis.sum()) < max(4, int(min_inliers)):
        return None, None
    img_pts = xy[vis]
    pitch_pts = _PITCH_VERTS[vis]

    def _ls_fit(pp, ip):
        H, _ = cv2.findHomography(pp, ip, 0)
        if H is None or not np.all(np.isfinite(H)):
            return None, None
        proj = cv2.perspectiveTransform(pp.reshape(-1, 1, 2), H).reshape(-1, 2)
        return H, np.linalg.norm(proj - ip, axis=1)

    H, res = _ls_fit(pitch_pts, img_pts)
    if H is None:
        return None, None
    keep = res <= max(2.0 * float(ransac_px), 3.0 * float(np.median(res)))
    if not keep.all() and int(keep.sum()) >= max(4, int(min_inliers)):
        H2, res2 = _ls_fit(pitch_pts[keep], img_pts[keep])
        if H2 is not None:
            H, res = H2, res2
        else:
            keep = np.ones(len(img_pts), dtype=bool)
    else:
        keep = np.ones(len(img_pts), dtype=bool)
    if float(np.mean(res)) > float(max_reproj_px):
        return None, None
    full = np.zeros(_N, dtype=bool)
    full[np.where(vis)[0][keep]] = True
    return H, full


def estimate_frame_motion(prev_gray, gray):
    """ΔH ảnh_{t-1} -> ảnh_t từ optical flow toàn khung hình.

    Camera broadcast quay/zoom quanh trục cố định nên chuyển động giữa 2 frame là
    MỘT homography toàn cục đúng cho mọi độ sâu (không cần thấy vạch sân). Dùng để
    dead-reckoning qua các frame model keypoint mất sân. cv2.RANSAC ở đây hợp lệ
    (khác fit keypoint): điểm nền tĩnh chiếm đa số áp đảo và trải khắp khung hình,
    outlier (cầu thủ đang chạy) là thiểu số sai thật sự. Trả 3x3 float32 hoặc None.
    """
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01,
                                 minDistance=25, blockSize=7)
    if p0 is None or len(p0) < 40:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                         winSize=(21, 21), maxLevel=3)
    if p1 is None:
        return None
    ok = st.ravel() == 1
    if int(ok.sum()) < 40:
        return None
    H, mask = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 3.0)
    if H is None or not np.all(np.isfinite(H)) or int(mask.sum()) < 25:
        return None
    return H.astype(np.float32)


def transformer_from_obs(obs):
    """Transformer ảnh->sân cho 1 frame đơn (đường inference ảnh, không làm mượt)."""
    if obs is None:
        return None
    H, _ = _fit_frame_H(obs["xy"], obs["conf"] > KP_CONF)
    if H is None:
        return None
    try:
        return _MatrixTransformer(np.linalg.inv(H))
    except np.linalg.LinAlgError:
        return None


def _median_time(arr, window):
    """Lọc trung vị theo trục thời gian (chặn outlier đơn lẻ). arr: (F, K)."""
    F = arr.shape[0]
    w = int(window)
    if w <= 1 or F < 3:
        return arr
    if w % 2 == 0:
        w += 1
    w = min(w, F if F % 2 == 1 else F - 1)
    if w < 3:
        return arr
    # mode="nearest": đệm biên bằng giá trị đầu/cuối. KHÔNG dùng scipy.signal.medfilt —
    # nó đệm số 0 nên kéo lệch 2 frame đầu/cuối mỗi dải quan sát (H bị hỏng ngay
    # trước khi mất sân rồi hold-last đóng băng H hỏng đó).
    from scipy.ndimage import median_filter
    return np.stack([median_filter(arr[:, c], size=w, mode="nearest")
                     for c in range(arr.shape[1])], axis=1)


def _smooth_time(arr, window):
    """Làm mượt CÓ TÂM theo trục thời gian (Savitzky-Golay). arr: (F, K)."""
    F = arr.shape[0]
    w = int(window)
    if F < 5 or w <= 1:
        return arr
    if w % 2 == 0:
        w += 1
    w = min(w, F if F % 2 == 1 else F - 1)
    if w < 3:
        return arr
    from scipy.signal import savgol_filter
    return savgol_filter(arr, w, 2, axis=0, mode="interp")


def _smooth_keypoint_series(series, window, med_window, gap_fill):
    """Làm sạch + làm mượt chuỗi thời gian của MỘT keypoint. series: (F, 2) chứa
    NaN ở frame không thấy. Chỉ lấp khoảng trống nội bộ ngắn (<= gap_fill frame)
    rồi lọc trung vị + Savitzky-Golay trên từng đoạn liên tục — KHÔNG ngoại suy
    ra ngoài khoảng keypoint thực sự được quan sát."""
    F = len(series)
    valid = np.isfinite(series[:, 0])
    if int(valid.sum()) < 3:
        return series
    out = series.copy()
    fvalid = valid.copy()
    vi = np.where(valid)[0]
    for a, b in zip(vi[:-1], vi[1:]):            # lấp khoảng trống nội bộ ngắn
        if 1 < b - a <= gap_fill:
            t = (np.arange(a + 1, b) - a) / float(b - a)
            out[a + 1:b] = (1 - t)[:, None] * series[a] + t[:, None] * series[b]
            fvalid[a + 1:b] = True

    # xử lý từng đoạn liên tục
    padded = np.concatenate([[False], fvalid, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]
    for s, e in zip(starts, ends):
        seg = out[s:e]
        seg = _median_time(seg, med_window)
        seg = _smooth_time(seg, window)
        out[s:e] = seg
    out[~fvalid] = np.nan
    return out


def build_transformers(obs_list, dh_list=None, window=HOMOGRAPHY_SMOOTH_WINDOW,
                       median=HOMOGRAPHY_MEDIAN_WINDOW,
                       ransac_px=HOMOGRAPHY_RANSAC_PX,
                       min_inliers=HOMOGRAPHY_MIN_INLIERS,
                       max_reproj_px=HOMOGRAPHY_MAX_REPROJ_PX):
    """obs_list: list dài F các quan sát keypoint (dict của HomographyEstimator
    hoặc None). dh_list (tuỳ chọn): list F ma trận chuyển động liên khung
    (estimate_frame_motion) để bắc cầu qua frame mất sân. Trả về list F
    transformer ảnh->sân (hoặc None).

    Các bước: fit LS toàn bộ điểm từng frame, cắt outlier lệch xa fit toàn cục
    -> làm mượt CÓ TÂM chuỗi thời gian của từng keypoint (window=1 để tắt,
    giống notebook gốc 100%) -> fit lại H mỗi frame từ keypoint đã mượt ->
    frame thiếu điểm: cập nhật H theo chuyển động camera đo bằng optical flow
    (dead-reckoning; map bám camera cả trong quãng mù dài — quãng mù 3.6s của
    video mẫu từng làm cụm điểm sai hàng chục mét khi chỉ đóng băng H), thiếu
    cả flow thì mới GIỮ H hợp lệ gần nhất.
    """
    F = len(obs_list)
    if F == 0:
        return []

    # Gom chuỗi keypoint; fit per-frame chỉ để LOẠI outlier thô khỏi chuỗi.
    series = np.full((F, _N, 2), np.nan, dtype=float)
    for f, ob in enumerate(obs_list):
        if ob is None:
            continue
        vis = ob["conf"] > KP_CONF
        if int(vis.sum()) < 4:
            continue
        H, inl = _fit_frame_H(ob["xy"], vis, ransac_px, min_inliers, max_reproj_px)
        keep = inl if inl is not None else vis   # fit fail -> giữ raw, lọc thời gian xử lý
        series[f][keep] = ob["xy"][keep]

    gap_fill = max(3, int(window))
    for k in range(_N):                          # làm mượt từng chuỗi keypoint
        series[:, k, :] = _smooth_keypoint_series(series[:, k, :], window, median, gap_fill)

    # Fit lại H mỗi frame từ keypoint đã mượt (least-squares, dữ liệu đã sạch).
    Hs = [None] * F
    n_fit = 0
    residuals = []
    for f in range(F):
        m = np.isfinite(series[f, :, 0])
        if int(m.sum()) < max(4, int(min_inliers)):
            continue
        src = _PITCH_VERTS[m]
        dst = series[f][m].astype(np.float32)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None or not np.all(np.isfinite(H)):
            continue
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
        if err > float(max_reproj_px):
            continue
        Hs[f] = H
        n_fit += 1
        residuals.append(err)

    if n_fit == 0:
        return [None] * F

    # Frame thiếu H: dead-reckoning bằng chuyển động camera (dh_list), thiếu flow
    # thì giữ H gần nhất. Trước frame fit đầu tiên không bắc cầu (chưa có mốc).
    cur = next(H for H in Hs if H is not None)
    seen_fit = False
    n_bridge = 0
    out = []
    for f in range(F):
        if Hs[f] is not None:
            cur = Hs[f]
            seen_fit = True
        elif (seen_fit and dh_list is not None and f < len(dh_list)
              and dh_list[f] is not None):
            cur = dh_list[f].astype(np.float64) @ cur
            n_bridge += 1
        try:
            out.append(_MatrixTransformer(np.linalg.inv(cur)))
        except np.linalg.LinAlgError:
            out.append(None)

    med_res = float(np.median(residuals))
    n_hold = F - n_fit - n_bridge
    print(f"  homography: fit trực tiếp {n_fit}/{F} frame "
          f"(residual median {med_res:.1f} px), dead-reckoning quang học "
          f"{n_bridge} frame, giữ H gần nhất {n_hold} frame")
    return out
