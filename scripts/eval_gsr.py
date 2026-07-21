"""Đánh giá GS-HOTA rút gọn (bỏ jersey) trên SoccerNet-GSR — cách khuyến nghị.

Chạy pipeline two-pass, chiếu cầu thủ lên sân, so với ground-truth toạ độ sân +
role + team. IdSim chỉ xét role + team (project không nhận diện số áo). Engine
HOTA/LocSim/IdSim trong football_analysis.evaluation (đã self-test).

Ví dụ:
  python scripts/eval_gsr.py --local --field roboflow --seq data/SoccerNetGSR/valid/SNGS-021
  python scripts/eval_gsr.py --local --split-dir data/SoccerNetGSR/valid

Mỗi chuỗi cần: <seq>/img1/*.jpg và <seq>/Labels-GameState.json.

⚠️ QUY CHIẾU TOẠ ĐỘ: model keypoint của bạn có thể định hướng sân ngược trục so
với hệ SNGS. Script tự chẩn đoán (in sai lệch median dưới mỗi phương án lật trục)
và gợi ý --flip-x/--flip-y. HÃY kiểm tra dòng '[chẩn đoán trục]' trước khi tin số
GS-HOTA — lật sai làm điểm vô nghĩa.
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from football_analysis.config import resolve_device, FIT_STRIDE, SWITCH_RATIO, FPS
from football_analysis.detection import load_models, load_models_local
from football_analysis.eval_io import (run_gsr_on_folder, fit_resolver_on_folder,
                                       load_gsr_labels, template_cm_to_sngs_m)
from football_analysis.evaluation import gs_hota, locsim


def _median_nn_error(gt_seq, pred_seq):
    """Sai lệch median của cặp gt-pred gần nhất (mét), bỏ qua thuộc tính — để dò lật trục."""
    ds = []
    for g, p in zip(gt_seq, pred_seq):
        gxy = np.asarray(g["xy"], float).reshape(-1, 2)
        pxy = np.asarray(p["xy"], float).reshape(-1, 2)
        if len(gxy) == 0 or len(pxy) == 0:
            continue
        d = np.sqrt(((gxy[:, None] - pxy[None, :]) ** 2).sum(-1))
        ds.extend(d.min(axis=1).tolist())
    return float(np.median(ds)) if ds else float("nan")


def _build_pred_seq(files, recs, gt_by_file, flip_x, flip_y):
    """Dựng pred_seq (mét, gốc tâm) khớp thứ tự frame của GT; gán team cluster->left/right."""
    # 1) quy đổi toạ độ + gom X theo cluster để quyết định left/right
    sngs = [template_cm_to_sngs_m(r["pitch_cm"], flip_x, flip_y) for r in recs]
    xs = {0: [], 1: []}
    for r, xy in zip(recs, sngs):
        for k in range(len(r["tid"])):
            c = int(r["team_cluster"][k])
            if c in (0, 1) and np.isfinite(xy[k, 0]):
                xs[c].append(xy[k, 0])
    mean0 = np.mean(xs[0]) if xs[0] else 0.0
    mean1 = np.mean(xs[1]) if xs[1] else 0.0
    label = {0: "left", 1: "right"} if mean0 <= mean1 else {0: "right", 1: "left"}

    rec_by_file = {r["file"]: (r, xy) for r, xy in zip(recs, sngs)}
    gt_seq, pred_seq = [], []
    for fn in files:
        gts = gt_by_file.get(fn, [])
        if fn not in rec_by_file:
            continue
        r, xy = rec_by_file[fn]
        pred_ids, pred_xy, pred_attrs = [], [], []
        for k in range(len(r["tid"])):
            if not np.isfinite(xy[k, 0]):
                continue
            role = str(r["role"][k])
            c = int(r["team_cluster"][k])
            team = label.get(c) if role in ("player", "goalkeeper") else None
            pred_ids.append(int(r["tid"][k])); pred_xy.append(xy[k])
            pred_attrs.append({"role": role, "team": team})
        gt_seq.append({"ids": [g["id"] for g in gts],
                       "xy": np.array([g["xy"] for g in gts], float).reshape(-1, 2),
                       "attrs": [{"role": g["role"], "team": g["team"]} for g in gts]})
        pred_seq.append({"ids": pred_ids,
                         "xy": np.array(pred_xy, float).reshape(-1, 2),
                         "attrs": pred_attrs})
    return gt_seq, pred_seq


def eval_sequence(seq_dir, player_model, field_model, device, tau, flip_x, flip_y,
                  auto_flip, id_keys, fit_stride, switch_ratio):
    json_path = os.path.join(seq_dir, "Labels-GameState.json")
    img_dir = os.path.join(seq_dir, "img1")
    gt_by_file = load_gsr_labels(json_path)
    resolver = fit_resolver_on_folder(player_model, img_dir, fit_stride, device, switch_ratio, FPS)
    files, recs = run_gsr_on_folder(img_dir, player_model, field_model, resolver)

    # Chẩn đoán trục: thử 4 phương án lật, in sai lệch median.
    print("  [chẩn đoán trục] median sai lệch NN theo phương án lật (mét):")
    best, best_err = (flip_x, flip_y), float("inf")
    for fx in (False, True):
        for fy in (False, True):
            gs, ps = _build_pred_seq(files, recs, gt_by_file, fx, fy)
            err = _median_nn_error(gs, ps)
            tag = f"flip_x={fx}, flip_y={fy}"
            mark = ""
            if err < best_err:
                best_err, best = err, (fx, fy);
            print(f"    {tag:26s} -> {err:6.2f} m")
    print(f"    => nhỏ nhất: flip_x={best[0]}, flip_y={best[1]} ({best_err:.2f} m)")
    if auto_flip:
        flip_x, flip_y = best
        print(f"    (--auto-flip: dùng flip_x={flip_x}, flip_y={flip_y})")

    gt_seq, pred_seq = _build_pred_seq(files, recs, gt_by_file, flip_x, flip_y)
    res = gs_hota(gt_seq, pred_seq, tau=tau, id_keys=id_keys)
    res["median_err_m"] = _median_nn_error(gt_seq, pred_seq)
    res["n_frames"] = len(gt_seq)
    return res


def main():
    ap = argparse.ArgumentParser(description="Đánh giá GS-HOTA rút gọn trên SoccerNet-GSR")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq", help="Một thư mục chuỗi (img1/ + Labels-GameState.json)")
    g.add_argument("--split-dir", help="Thư mục cha chứa nhiều chuỗi")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--field", choices=["local", "roboflow"], default=None,
                    help="Nguồn model sân riêng (roboflow cho map 2D tốt hơn)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pitch-weights", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tau", type=float, default=5.0, help="Ngưỡng khoảng cách LocSim (m)")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--auto-flip", action="store_true",
                    help="Tự chọn phương án lật trục cho sai lệch median nhỏ nhất (dùng thận trọng)")
    ap.add_argument("--role-only", action="store_true", help="IdSim chỉ xét role (bỏ cả team)")
    ap.add_argument("--fit-stride", type=int, default=FIT_STRIDE)
    ap.add_argument("--switch-ratio", type=float, default=SWITCH_RATIO)
    args = ap.parse_args()

    device = resolve_device(args.device)
    use_local = args.local
    field_src = args.field or ("local" if use_local else "roboflow")
    # Nạp model detection
    if use_local:
        player_model, field_local = load_models_local(
            detection_path=args.weights, pitch_path=args.pitch_weights, device=device)
    else:
        player_model, field_local = load_models(api_key=args.api_key)
    # Nạp model sân theo nguồn chọn
    if field_src == "roboflow":
        _, field_model = load_models(api_key=args.api_key)
    else:
        field_model = field_local

    id_keys = ("role",) if args.role_only else ("role", "team")
    seqs = ([args.seq] if args.seq else
            sorted(os.path.join(args.split_dir, d) for d in os.listdir(args.split_dir)
                   if os.path.isdir(os.path.join(args.split_dir, d))))
    rows = []
    for s in seqs:
        print(f"▶ {os.path.basename(s)}")
        try:
            r = eval_sequence(s, player_model, field_model, device, args.tau,
                              args.flip_x, args.flip_y, args.auto_flip, id_keys,
                              args.fit_stride, args.switch_ratio)
            rows.append((os.path.basename(s), r))
            print(f"  GS-HOTA={r['HOTA']*100:.2f}  DetA={r['DetA']*100:.2f}  "
                  f"AssA={r['AssA']*100:.2f}  (median err {r['median_err_m']:.1f} m)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ lỗi: {e}")

    if not rows:
        print("Không đánh giá được chuỗi nào."); return
    keys = ["HOTA", "DetA", "AssA", "LocA"]
    print("\n=== KẾT QUẢ THEO CHUỖI (GS-HOTA rút gọn, bỏ jersey) ===")
    print("seq".ljust(16), *[k.rjust(8) for k in keys], "err_m".rjust(8))
    for name, r in rows:
        print(name[:16].ljust(16), *[f"{r[k]*100:8.2f}" for k in keys],
              f"{r['median_err_m']:8.1f}")
    mean = {k: np.mean([r[k] for _, r in rows]) for k in keys}
    print("-" * 60)
    print("TRUNG BÌNH".ljust(16), *[f"{mean[k]*100:8.2f}" for k in keys])

    kind = "role" if args.role_only else "role+team"
    print(f"\n=== SO SÁNH VỚI PAPER (Bảng 1, tập Test) — cấu hình Pitch+{kind}, KHÔNG jersey ===")
    print("Cấu hình".ljust(34), "GS-HOTA".rjust(8))
    for n, v in [("GSR-Baseline full (Pitch+R+T+J)", 22.26),
                 ("Ablation Pitch+Role", 40.76),
                 ("Ablation Pitch+Team", 37.03),
                 ("GSR-Baseline valid (full)", 18.05)]:
        print(n.ljust(34), f"{v:8.2f}")
    print(f"Project của bạn (Pitch+{kind})".ljust(34), f"{mean['HOTA']*100:8.2f}")


if __name__ == "__main__":
    main()
