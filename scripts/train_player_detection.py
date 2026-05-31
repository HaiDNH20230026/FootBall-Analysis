"""Huấn luyện YOLOv11s phát hiện player / ball / goalkeeper / referee — §3.

Tương ứng notebooks/player_dection.ipynb: transfer learning từ yolo11s.pt, 32 epoch,
imgsz 1280, batch 32. Cần ROBOFLOW_API_KEY (tải dataset); W&B là tuỳ chọn.

Ví dụ:
    # Tải dataset từ Roboflow rồi train
    python scripts/train_player_detection.py --epochs 32 --imgsz 1280 --batch 32

    # Dùng data.yaml đã có sẵn (bỏ qua bước tải)
    python scripts/train_player_detection.py --data data/raw/soccernet/data.yaml

    # Train xong deploy ngược lên Roboflow
    python scripts/train_player_detection.py --deploy
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11s player/ball detection")
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--weights", default="yolo11s.pt", help="trọng số khởi đầu (transfer learning)")
    p.add_argument("--device", default=None, help="vd '0' cho GPU 0, 'cpu' (mặc định: tự chọn)")
    p.add_argument("--data", default=None, help="đường dẫn data.yaml có sẵn (bỏ qua tải Roboflow)")
    # Roboflow dataset
    p.add_argument("--workspace", default="ngc-hong-hi-s-workspace")
    p.add_argument("--project", default="soccernet-1ae2v-e6wqy")
    p.add_argument("--version", type=int, default=3)
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

    # 1) Lấy data.yaml (tải Roboflow hoặc dùng sẵn)
    dataset_version = args.version
    if args.data:
        data_yaml = args.data
    else:
        if not os.environ.get("ROBOFLOW_API_KEY"):
            sys.exit("Lỗi: cần ROBOFLOW_API_KEY để tải dataset (hoặc truyền --data path/data.yaml).")
        from roboflow import Roboflow
        rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
        project = rf.workspace(args.workspace).project(args.project)
        dataset = project.version(args.version).download("yolov11", location=args.data_dir)
        data_yaml = os.path.join(dataset.location, "data.yaml")

    # 2) (tuỳ chọn) bật W&B
    if args.wandb:
        from ultralytics import settings
        settings.update({"wandb": True})

    # 3) Train
    from ultralytics import YOLO
    model = YOLO(args.weights)
    results = model.train(
        task="detect", data=data_yaml, epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, device=args.device, plots=True,
    )
    print("✅ Train xong. Weights tốt nhất:", model.trainer.best)

    # 4) (tuỳ chọn) deploy lên Roboflow
    if args.deploy and not args.data:
        train_dir = os.path.dirname(os.path.dirname(str(model.trainer.best)))
        project.version(dataset_version).deploy(model_type="yolov11", model_path=train_dir)
        print("✅ Đã deploy lên Roboflow.")


if __name__ == "__main__":
    main()
