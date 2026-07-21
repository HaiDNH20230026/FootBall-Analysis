"""Đánh giá MOT ảnh (phương án dự phòng) trên SoccerNet-Tracking.

Chạy detection + ByteTrack của project trên các chuỗi ảnh, so với ground-truth
MOT-Challenge, tính HOTA / DetA / AssA / MOTA / IDF1 (engine tự chứa trong
football_analysis.evaluation — đã self-test). Đối chiếu Bảng S1 của paper.

Ví dụ:
  python scripts/eval_mot.py --local --split-dir data/SoccerNetTracking/test
  python scripts/eval_mot.py --local --seq data/SoccerNetTracking/test/SNMOT-116

Mỗi chuỗi cần: <seq>/img1/*.jpg và <seq>/gt/gt.txt (+ seqinfo.ini nếu có).
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from football_analysis.config import resolve_device
from football_analysis.detection import load_models, load_models_local
from football_analysis.eval_io import run_tracking_on_folder, load_mot_gt
from football_analysis.evaluation import hota_eval_sequence, mot_clear, iou_matrix, _densify


def eval_sequence(seq_dir, player_model, iou_thr, keep_classes):
    files, pred = run_tracking_on_folder(seq_dir, player_model)
    gt_files, gt = load_mot_gt(seq_dir, keep_classes=keep_classes)
    T = min(len(pred), len(gt))
    if len(pred) != len(gt):
        print(f"  ⚠️ số frame lệch (pred {len(pred)} vs gt {len(gt)}) -> cắt còn {T}")
    pred, gt = pred[:T], gt[:T]

    # HOTA ảnh: similarity = IoU box, id dày đặc hoá theo thời gian.
    gt_ids, _ = _densify([np.asarray(g["ids"], int) for g in gt])
    pr_ids, _ = _densify([np.asarray(p["ids"], int) for p in pred])
    sims = [iou_matrix(np.asarray(g["boxes"], float).reshape(-1, 4),
                       np.asarray(p["boxes"], float).reshape(-1, 4))
            for g, p in zip(gt, pred)]
    h = hota_eval_sequence(gt_ids, pr_ids, sims)
    m = mot_clear(gt, pred, iou_thr=iou_thr)
    return {**h, **m}


def main():
    ap = argparse.ArgumentParser(description="Đánh giá MOT ảnh trên SoccerNet-Tracking")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq", help="Một thư mục chuỗi (img1/ + gt/gt.txt)")
    g.add_argument("--split-dir", help="Thư mục cha chứa nhiều chuỗi")
    ap.add_argument("--local", action="store_true", help="Dùng weights .pt local")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU cho MOTA/IDF1")
    ap.add_argument("--classes", default=None,
                    help="Lọc class GT theo cột 8 của gt.txt, vd '1,2,3' (mặc định: giữ hết)")
    args = ap.parse_args()

    device = resolve_device(args.device)
    if args.local:
        player_model, _ = load_models_local(detection_path=args.weights, device=device)
    else:
        player_model, _ = load_models(api_key=args.api_key)
    keep = set(int(x) for x in args.classes.split(",")) if args.classes else None

    seqs = ([args.seq] if args.seq else
            sorted(os.path.join(args.split_dir, d) for d in os.listdir(args.split_dir)
                   if os.path.isdir(os.path.join(args.split_dir, d))))
    rows = []
    for s in seqs:
        print(f"▶ {os.path.basename(s)}")
        try:
            rows.append((os.path.basename(s), eval_sequence(s, player_model, args.iou, keep)))
        except Exception as e:
            print(f"  ❌ lỗi: {e}")

    if not rows:
        print("Không đánh giá được chuỗi nào."); return
    keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
    print("\n=== KẾT QUẢ THEO CHUỖI ===")
    print("seq".ljust(16), *[k.rjust(8) for k in keys])
    for name, r in rows:
        print(name[:16].ljust(16), *[f"{r[k]*100:8.2f}" if k not in ("IDSW",)
                                     else f"{int(r[k]):8d}" for k in keys])
    mean = {k: np.mean([r[k] for _, r in rows]) for k in keys}
    print("-" * 70)
    print("TRUNG BÌNH".ljust(16),
          *[f"{mean[k]*100:8.2f}" if k != "IDSW" else f"{mean[k]:8.1f}" for k in keys])

    print("\n=== SO SÁNH VỚI PAPER (Bảng S1, SoccerNet-Tracking) ===")
    print("Method".ljust(24), "HOTA".rjust(7), "DetA".rjust(7), "AssA".rjust(7), "MOTA".rjust(7))
    for n, h, d, a, mo in [("ByteTrack [paper]", 47.23, 44.49, 50.26, 31.74),
                           ("GSR-Baseline [paper]", 57.64, 67.42, 49.42, 80.79),
                           ("FairMOT-ft [paper]", 57.88, 66.56, 50.49, 83.56)]:
        print(n.ljust(24), f"{h:7.2f}", f"{d:7.2f}", f"{a:7.2f}", f"{mo:7.2f}")
    print("Project của bạn".ljust(24),
          f"{mean['HOTA']*100:7.2f}", f"{mean['DetA']*100:7.2f}",
          f"{mean['AssA']*100:7.2f}", f"{mean['MOTA']*100:7.2f}")


if __name__ == "__main__":
    main()
