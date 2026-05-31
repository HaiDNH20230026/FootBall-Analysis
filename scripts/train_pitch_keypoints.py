"""Huấn luyện YOLOv11s-pose phát hiện keypoint sân — §3.

Tương ứng notebooks/pitch_keypoint_detection.ipynb: 100 epoch, imgsz 640, batch 32,
TẮT augmentation mosaic (mosaic=0.0). Cần ROBOFLOW_API_KEY; W&B là tuỳ chọn.

Ví dụ:
    python scripts/train_pitch_keypoints.py --epochs 100 --imgsz 640 --batch 32
    python scripts/train_pitch_keypoints.py --data data/raw/football-field/data.yaml
    python scripts/train_pitch_keypoints.py --deploy
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11s-pose pitch keypoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--weights", default="yolo11s-pose.pt", help="trọng số khởi đầu")
    p.add_argument("--device", default=None, help="vd '0' cho GPU 0, 'cpu' (mặc định: tự chọn)")
    p.add_argument("--data", default=None, help="đường dẫn data.yaml có sẵn (bỏ qua tải Roboflow)")
    # Roboflow dataset
    p.add_argument("--workspace", default="ngc-hong-hi-s-workspace")
    p.add_argument("--project", default="football-field-detection-f07vi-o9ocz")
    p.add_argument("--version", type=int, default=2)
    p.add_argument("--data-dir", default="data/raw", help="nơi tải dataset về")
    # tuỳ chọn
    p.add_argument("--wandb", action="store_true", help="bật theo dõi Weights & Biases")
    p.add_argument("--deploy", action="store_true", help="deploy weights tốt nhất lên Roboflow")
    p.add_argument("--api-key", default=None, help="Roboflow API key (mặc định lấy từ môi trường)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    if args.api_key:
        os.environ["ROBOFLOW_API_KEY"] = args.api_key

    # 1) Lấy data.yaml
    if args.data:
        data_yaml = args.data
    else:
        if not os.environ.get("ROBOFLOW_API_KEY"):
            sys.exit("Lỗi: cần ROBOFLOW_API_KEY để tải dataset (hoặc truyền --data path/data.yaml).")
        from roboflow import Roboflow
        rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
        project = rf.workspace(args.workspace).project(args.project)
        dataset = project.version(args.version).download("yolov8", location=args.data_dir)
        data_yaml = os.path.join(dataset.location, "data.yaml")

    # 2) (tuỳ chọn) W&B
    if args.wandb:
        from ultralytics import settings
        settings.update({"wandb": True})

    # 3) Train (pose, tắt mosaic)
    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(
        task="pose", data=data_yaml, epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, mosaic=0.0, device=args.device, plots=True,
    )
    print("✅ Train xong. Weights tốt nhất:", model.trainer.best)

    # 4) (tuỳ chọn) deploy
    if args.deploy and not args.data:
        train_dir = os.path.dirname(os.path.dirname(str(model.trainer.best)))
        project.version(args.version).deploy(model_type="yolov11-pose", model_path=train_dir)
        print("✅ Đã deploy lên Roboflow.")


if __name__ == "__main__":
    main()
