"""Engine đánh giá tự chứa cho báo cáo: HOTA / DetA / AssA / LocA (ports thuật toán
TrackEval) + CLEAR-MOT (MOTA / IDF1 / IDSW), và hàm tương đồng GS-HOTA
(LocSim Gaussian trên sân + IdSim theo thuộc tính).

Vì sao tự viết thay vì gọi TrackEval/SoccerNet: (1) không phải cài thư viện eval
ngoài (hay vỡ theo phiên bản), (2) mọi con số TRUY VẾT ĐƯỢC về code — giảng viên
đọc được từng bước, (3) có `--selftest`: chạy trên dữ liệu tổng hợp để kiểm chứng
engine đúng ngay cả khi CHƯA có dataset (dự đoán hoàn hảo -> HOTA≈1, MOTA≈1...).

Thuật toán HOTA bám đúng Luiten et al. 2020 (bản TrackEval): global alignment
score -> Hungarian mỗi frame theo score * similarity -> gate theo alpha -> tích
phân DetA·AssA trên alpha ∈ [0.05, 0.95] bước 0.05. GS-HOTA (paper SoccerNet-GSR)
chỉ thay similarity IoU ảnh bằng LocSim(sân) × IdSim(thuộc tính); phần HOTA giữ
nguyên. Xem hàm `gs_hota` và `mot_clear`.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

_EPS = 1e-10
# alpha ∈ {0.05, 0.10, ..., 0.95} — 19 mốc, đúng như GS-HOTA của paper.
ALPHAS = np.arange(0.05, 0.99, 0.05)


# ----------------------------------------------------------------------------
# Hàm tương đồng
# ----------------------------------------------------------------------------
def locsim(pred_xy, gt_xy, tau=5.0):
    """LocSim (Eq.4 của paper): Gaussian trên khoảng cách Euclid (mét) trên sân.

    pred_xy: (N,2), gt_xy: (M,2) — ĐƠN VỊ MÉT. tau=5m: khoảng cách 5m -> 0.05.
    Trả ma trận (M, N) trong [0,1] (hàng = GT, cột = prediction — theo quy ước engine).
    """
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return np.zeros((len(gt_xy), len(pred_xy)))
    d2 = ((gt_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(-1)  # (M,N)
    return np.exp(np.log(0.05) * d2 / (tau ** 2))


def idsim(gt_attrs, pred_attrs, keys=("role", "team")):
    """IdSim (Eq.5, biến thể BỎ JERSEY): 1 nếu MỌI thuộc tính trong `keys` khớp,
    ngược lại 0. Thuộc tính GT = None thì BỎ QUA (paper: "attributes not provided
    in G are ignored" — vd trọng tài không có team). gt_attrs/pred_attrs: list dict.
    Trả ma trận (M, N) 0/1.
    """
    M, N = len(gt_attrs), len(pred_attrs)
    out = np.ones((M, N))
    for i, g in enumerate(gt_attrs):
        for j, p in enumerate(pred_attrs):
            for k in keys:
                gv = g.get(k)
                if gv is None:                 # thuộc tính GT trống -> bỏ qua khoá này
                    continue
                if p.get(k) != gv:
                    out[i, j] = 0.0
                    break
    return out


def iou_matrix(gt_boxes, pred_boxes):
    """IoU ảnh (M,N) giữa GT và prediction boxes dạng xyxy. Dùng cho MOT & HOTA ảnh."""
    M, N = len(gt_boxes), len(pred_boxes)
    if M == 0 or N == 0:
        return np.zeros((M, N))
    g = gt_boxes[:, None, :]; p = pred_boxes[None, :, :]
    ix1 = np.maximum(g[..., 0], p[..., 0]); iy1 = np.maximum(g[..., 1], p[..., 1])
    ix2 = np.minimum(g[..., 2], p[..., 2]); iy2 = np.minimum(g[..., 3], p[..., 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ga = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    pa = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    union = ga[:, None] + pa[None, :] - inter
    return np.where(union > _EPS, inter / np.maximum(union, _EPS), 0.0)


# ----------------------------------------------------------------------------
# HOTA (dùng chung cho GS-HOTA và HOTA ảnh — chỉ khác similarity đầu vào)
# ----------------------------------------------------------------------------
def hota_eval_sequence(gt_ids_per_t, pr_ids_per_t, sim_per_t, alphas=ALPHAS):
    """HOTA trên MỘT chuỗi. Đầu vào đã 'dày đặc hoá' id về 0..K-1.

    gt_ids_per_t/pr_ids_per_t: list dài T các mảng int (id đã map về dense).
    sim_per_t: list dài T các ma trận (n_gt_t, n_pr_t) trong [0,1] (hàng=GT).
    Trả dict: HOTA, DetA, AssA, LocA (trung bình trên alpha) + per-alpha arrays.
    """
    T = len(gt_ids_per_t)
    all_gt = np.concatenate([g for g in gt_ids_per_t if len(g)]) if any(len(g) for g in gt_ids_per_t) else np.array([], int)
    all_pr = np.concatenate([p for p in pr_ids_per_t if len(p)]) if any(len(p) for p in pr_ids_per_t) else np.array([], int)
    num_gt = int(all_gt.max()) + 1 if len(all_gt) else 0
    num_pr = int(all_pr.max()) + 1 if len(all_pr) else 0
    A = len(alphas)
    zero = dict(HOTA=0.0, DetA=0.0, AssA=0.0, LocA=0.0,
                HOTA_a=np.zeros(A), DetA_a=np.zeros(A), AssA_a=np.zeros(A))
    if num_gt == 0 or num_pr == 0:
        return zero

    # Lượt 1: potential matches (IoU của similarity) + đếm số lần xuất hiện mỗi id.
    potential = np.zeros((num_gt, num_pr))
    gt_count = np.zeros(num_gt); pr_count = np.zeros(num_pr)
    for g, p, sim in zip(gt_ids_per_t, pr_ids_per_t, sim_per_t):
        if len(g) and len(p):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.where(denom > _EPS, sim / np.maximum(denom, _EPS), 0.0)
            potential[np.ix_(g, p)] += sim_iou
        gt_count[g] += 1
        pr_count[p] += 1

    denom = gt_count[:, None] + pr_count[None, :] - potential
    global_align = np.where(denom > _EPS, potential / np.maximum(denom, _EPS), 0.0)

    HOTA_a = np.zeros(A); DetA_a = np.zeros(A); AssA_a = np.zeros(A); LocA_a = np.zeros(A)
    for ai, alpha in enumerate(alphas):
        TP = FN = FP = 0.0
        loc_sum = 0.0
        matches_count = np.zeros((num_gt, num_pr))
        for g, p, sim in zip(gt_ids_per_t, pr_ids_per_t, sim_per_t):
            ng, npr = len(g), len(p)
            if ng == 0:
                FP += npr; continue
            if npr == 0:
                FN += ng; continue
            score = global_align[np.ix_(g, p)] * sim
            r, c = linear_sum_assignment(-score)
            keep = sim[r, c] >= alpha - _EPS
            r, c = r[keep], c[keep]
            n = len(r)
            TP += n; FN += ng - n; FP += npr - n
            loc_sum += sim[r, c].sum()
            matches_count[g[r], p[c]] += 1
        ass_denom = gt_count[:, None] + pr_count[None, :] - matches_count
        ass = np.where(ass_denom > _EPS, matches_count / np.maximum(ass_denom, _EPS), 0.0)
        AssA_a[ai] = (ass * matches_count).sum() / max(1.0, TP)
        DetA_a[ai] = TP / max(1.0, TP + FN + FP)
        LocA_a[ai] = loc_sum / max(1.0, TP)
        HOTA_a[ai] = np.sqrt(DetA_a[ai] * AssA_a[ai])
    return dict(HOTA=float(HOTA_a.mean()), DetA=float(DetA_a.mean()),
                AssA=float(AssA_a.mean()), LocA=float(LocA_a.mean()),
                HOTA_a=HOTA_a, DetA_a=DetA_a, AssA_a=AssA_a)


def _densify(ids_per_t):
    """Map id tuỳ ý (có thể thưa/âm) về 0..K-1, giữ nhất quán theo thời gian."""
    uniq = {}
    out = []
    for ids in ids_per_t:
        row = np.array([uniq.setdefault(int(i), len(uniq)) for i in ids], dtype=int)
        out.append(row)
    return out, len(uniq)


# ----------------------------------------------------------------------------
# GS-HOTA: gộp similarity = LocSim × IdSim rồi gọi HOTA
# ----------------------------------------------------------------------------
def gs_hota(gt_seq, pred_seq, tau=5.0, id_keys=("role", "team")):
    """GS-HOTA (biến thể bỏ jersey) trên MỘT chuỗi.

    gt_seq/pred_seq: list dài T, mỗi phần tử là dict frame:
        {"ids": [int...], "xy": (n,2) mét, "attrs": [ {"role":..,"team":..}, ... ]}
    Similarity mỗi frame = LocSim(xy) × IdSim(attrs). Trả dict HOTA/DetA/AssA/LocA.
    """
    T = len(gt_seq)
    gt_ids = [np.asarray(f["ids"], int) for f in gt_seq]
    pr_ids = [np.asarray(f["ids"], int) for f in pred_seq]
    gt_ids, _ = _densify(gt_ids)
    pr_ids, _ = _densify(pr_ids)
    sims = []
    for fg, fp in zip(gt_seq, pred_seq):
        gxy = np.asarray(fg["xy"], float).reshape(-1, 2)
        pxy = np.asarray(fp["xy"], float).reshape(-1, 2)
        s = locsim(pxy, gxy, tau) * idsim(fg["attrs"], fp["attrs"], id_keys)
        sims.append(s)
    return hota_eval_sequence(gt_ids, pr_ids, sims)


# ----------------------------------------------------------------------------
# CLEAR-MOT: MOTA, IDF1, IDSW (image space, IoU >= threshold)
# ----------------------------------------------------------------------------
def mot_clear(gt_seq, pred_seq, iou_thr=0.5):
    """CLEAR-MOT + Identity trên MỘT chuỗi (toạ độ ảnh, box xyxy).

    gt_seq/pred_seq: list dài T, mỗi phần tử {"ids":[...], "boxes":(n,4) xyxy}.
    Trả dict: MOTA, MOTP, IDF1, IDSW, FP, FN, TP, num_gt.
    Quy tắc CLEAR: ưu tiên giữ cặp match của frame trước (nếu còn IoU>=thr) để đếm
    IDSW đúng; phần còn lại match Hungarian theo IoU.
    """
    prev = {}                       # gt_id -> pred_id của frame trước (để bắt IDSW)
    IDSW = FP = FN = TP = 0
    dist_sum = 0.0
    per_gt_id_matches = {}          # cho IDF1
    for f in range(len(gt_seq)):
        g_ids = list(map(int, gt_seq[f]["ids"]))
        p_ids = list(map(int, pred_seq[f]["ids"]))
        gb = np.asarray(gt_seq[f]["boxes"], float).reshape(-1, 4)
        pb = np.asarray(pred_seq[f]["boxes"], float).reshape(-1, 4)
        iou = iou_matrix(gb, pb)
        matched_g, matched_p = set(), set()
        cur = {}
        # 1) giữ match cũ nếu còn hợp lệ
        for gi, g in enumerate(g_ids):
            pj = prev.get(g)
            if pj is not None and pj in p_ids:
                pjj = p_ids.index(pj)
                if iou[gi, pjj] >= iou_thr:
                    matched_g.add(gi); matched_p.add(pjj)
                    cur[g] = pj; TP += 1; dist_sum += iou[gi, pjj]
                    per_gt_id_matches.setdefault(g, []).append(pj)
        # 2) Hungarian phần còn lại
        rg = [i for i in range(len(g_ids)) if i not in matched_g]
        rp = [j for j in range(len(p_ids)) if j not in matched_p]
        if rg and rp:
            sub = iou[np.ix_(rg, rp)]
            r, c = linear_sum_assignment(-sub)
            for a, b in zip(r, c):
                if sub[a, b] >= iou_thr:
                    gi, pj = rg[a], rp[b]
                    g, p = g_ids[gi], p_ids[pj]
                    matched_g.add(gi); matched_p.add(pj)
                    cur[g] = p; TP += 1; dist_sum += sub[a, b]
                    per_gt_id_matches.setdefault(g, []).append(p)
                    if g in prev and prev[g] != p:      # từng match id khác -> IDSW
                        IDSW += 1
        FN += len(g_ids) - len(matched_g)
        FP += len(p_ids) - len(matched_p)
        # cập nhật prev: giữ id cũ cho GT không match frame này (chuẩn CLEAR)
        newprev = dict(prev)
        for g, p in cur.items():
            newprev[g] = p
        prev = newprev

    num_gt = sum(len(f["ids"]) for f in gt_seq)
    MOTA = 1.0 - (FN + FP + IDSW) / max(1, num_gt)
    MOTP = dist_sum / max(1, TP)          # IoU trung bình của match (cao = tốt)
    idf1 = _idf1(gt_seq, pred_seq, iou_thr)
    return dict(MOTA=MOTA, MOTP=MOTP, IDF1=idf1, IDSW=IDSW,
                FP=FP, FN=FN, TP=TP, num_gt=num_gt)


def _idf1(gt_seq, pred_seq, iou_thr):
    """IDF1: match toàn cục id_GT <-> id_pred để tối đa số frame khớp (IDTP)."""
    gt_ids = sorted({int(i) for f in gt_seq for i in f["ids"]})
    pr_ids = sorted({int(i) for f in pred_seq for i in f["ids"]})
    if not gt_ids or not pr_ids:
        return 0.0
    gi = {v: k for k, v in enumerate(gt_ids)}
    pi = {v: k for k, v in enumerate(pr_ids)}
    overlap = np.zeros((len(gt_ids), len(pr_ids)))
    for f in range(len(gt_seq)):
        g_ids = list(map(int, gt_seq[f]["ids"]))
        p_ids = list(map(int, pred_seq[f]["ids"]))
        gb = np.asarray(gt_seq[f]["boxes"], float).reshape(-1, 4)
        pb = np.asarray(pred_seq[f]["boxes"], float).reshape(-1, 4)
        iou = iou_matrix(gb, pb)
        for a, g in enumerate(g_ids):
            for b, p in enumerate(p_ids):
                if iou[a, b] >= iou_thr:
                    overlap[gi[g], pi[p]] += 1
    # Hungarian tối đa IDTP
    r, c = linear_sum_assignment(-overlap)
    IDTP = sum(overlap[a, b] for a, b in zip(r, c))
    gt_total = sum(len(f["ids"]) for f in gt_seq)
    pr_total = sum(len(f["ids"]) for f in pred_seq)
    IDFN = gt_total - IDTP
    IDFP = pr_total - IDTP
    return 2 * IDTP / max(1, 2 * IDTP + IDFN + IDFP)


# ----------------------------------------------------------------------------
# Self-test: chứng minh engine đúng mà KHÔNG cần dataset
# ----------------------------------------------------------------------------
def selftest():
    """Kiểm chứng engine trên dữ liệu tổng hợp. In PASS/FAIL cho từng khẳng định."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1) LocSim: 0m -> 1, 5m -> 0.05
    check("LocSim(0m)=1", abs(locsim(np.array([[0, 0]]), np.array([[0, 0]]))[0, 0] - 1) < 1e-9)
    check("LocSim(5m)=0.05", abs(locsim(np.array([[5, 0]]), np.array([[0, 0]]))[0, 0] - 0.05) < 1e-6)

    # 2) IdSim: khớp role+team = 1; lệch team = 0; GT team None -> bỏ qua
    a = [{"role": "player", "team": "left"}]
    check("IdSim khớp=1", idsim(a, [{"role": "player", "team": "left"}])[0, 0] == 1)
    check("IdSim lệch team=0", idsim(a, [{"role": "player", "team": "right"}])[0, 0] == 0)
    check("IdSim GT team None -> bỏ qua",
          idsim([{"role": "referee", "team": None}], [{"role": "referee", "team": "left"}])[0, 0] == 1)

    # 3) GS-HOTA: dự đoán TRÙNG KHỚP HOÀN HẢO -> HOTA ~ 1
    T = 10
    rng = np.random.default_rng(0)
    gt = []
    for t in range(T):
        xy = rng.uniform(-40, 40, (6, 2))
        gt.append({"ids": list(range(6)), "xy": xy,
                   "attrs": [{"role": "player", "team": "left"}] * 3 + [{"role": "player", "team": "right"}] * 3})
    perfect = gs_hota(gt, gt)
    check("GS-HOTA(perfect) > 0.99", perfect["HOTA"] > 0.99)

    # 4) GS-HOTA: dịch mọi điểm 5m -> LocSim=0.05 <= mọi alpha>0.05 nên gần như không match
    shifted = []
    for f in gt:
        g = dict(f); g["xy"] = np.asarray(f["xy"]) + np.array([5.0, 0.0]); shifted.append(g)
    degraded = gs_hota(gt, shifted)
    check("GS-HOTA(+5m) < GS-HOTA(perfect)", degraded["HOTA"] < perfect["HOTA"])

    # 5) GS-HOTA: sai HẾT team -> IdSim=0 -> HOTA=0
    wrong = []
    for f in gt:
        g = dict(f); g["attrs"] = [{"role": "player", "team": "wrong"}] * 6; wrong.append(g)
    check("GS-HOTA(sai hết team)=0", gs_hota(gt, wrong)["HOTA"] < 1e-9)

    # 6) MOTA: box trùng khớp hoàn hảo -> MOTA=1, IDSW=0, IDF1=1
    gtb = []
    for t in range(T):
        boxes = np.array([[x, x, x + 10, x + 20] for x in range(0, 60, 10)], float)
        gtb.append({"ids": list(range(6)), "boxes": boxes})
    m = mot_clear(gtb, gtb)
    check("MOTA(perfect)=1", abs(m["MOTA"] - 1) < 1e-9)
    check("IDSW(perfect)=0", m["IDSW"] == 0)
    check("IDF1(perfect)=1", abs(m["IDF1"] - 1) < 1e-9)

    # 7) MOTA: thiếu 1 nửa số box -> MOTA=0.5
    half = [{"ids": f["ids"][:3], "boxes": f["boxes"][:3]} for f in gtb]
    check("MOTA(thiếu nửa)=0.5", abs(mot_clear(gtb, half)["MOTA"] - 0.5) < 1e-9)

    print(f"\n  => {'TẤT CẢ PASS ✅' if ok else 'CÓ FAIL ❌'}")
    return ok


if __name__ == "__main__":
    selftest()
