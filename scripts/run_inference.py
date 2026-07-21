"""Chạy inference từ dòng lệnh.

Ví dụ:
    # Video
    python scripts/run_inference.py --source data/raw/match.mp4 --target outputs/out.mp4 --max-frames 150

    # Cấu hình cho kết quả map TỐT NHẤT hiện tại: cầu thủ local + model sân hosted
    # Roboflow f07vi/15 (thấy sân 100% frame trên video mẫu, cần ROBOFLOW_API_KEY)
    python scripts/run_inference.py --local --field roboflow --no-ball \
        --source data/samples/2e57b9_0.mp4 --target outputs/out.mp4

    # Ảnh
    python scripts/run_inference.py --source data/raw/frame.jpg --target outputs/out.jpg --image

API key: đặt biến môi trường ROBOFLOW_API_KEY, hoặc tạo file .env ở gốc repo:
    ROBOFLOW_API_KEY=your_key_here
hoặc truyền trực tiếp bằng --api-key.
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Football analysis inference")
    p.add_argument("--source", required=True, help="Đường dẫn video hoặc ảnh đầu vào")
    p.add_argument("--target", required=True, help="Đường dẫn file kết quả")
    p.add_argument("--image", action="store_true", help="Xử lý ảnh thay vì video")
    p.add_argument("--max-frames", type=int, default=None, help="Giới hạn số frame (test nhanh)")
    p.add_argument("--fit-stride", type=int, default=20, help="Stride gom crop để fit phân đội")
    p.add_argument("--switch-ratio", type=float, default=0.5, help="Hysteresis chống flicker đội")
    p.add_argument("--touch-px", type=float, default=70.0, help="Ngưỡng chạm bóng (px, ~1080p)")
    p.add_argument("--no-radar", action="store_true", help="Tắt radar 2D trên video")
    p.add_argument("--no-voronoi", action="store_true", help="Tắt biểu đồ Voronoi trên video")
    p.add_argument("--no-ball", action="store_true",
                   help="Tắt hiển thị quả bóng (cả trên khung lẫn minimap)")
    p.add_argument("--out-fps", type=float, default=None,
                   help="FPS video kết quả (mặc định = fps nguồn; giữ nguyên thời lượng)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Thiết bị chạy TeamClassifier")
    p.add_argument("--local", action="store_true",
                   help="Dùng weights .pt local (Ultralytics) thay vì Roboflow API")
    p.add_argument("--field", choices=["local", "roboflow"], default=None,
                   help="Nguồn model KEYPOINT SÂN, tách riêng khỏi --local. "
                        "'--local --field roboflow' = detect cầu thủ local + sân hosted "
                        "f07vi/15 (thấy sân ổn định hơn hẳn model tự train, cần API key)")
    p.add_argument("--weights", default=None, help="Đường dẫn weights detection (chỉ khi --local)")
    p.add_argument("--pitch-weights", default=None, help="Đường dẫn weights keypoint sân (chỉ khi --local)")
    p.add_argument("--api-key", default=None, help="Roboflow API key (mặc định lấy từ môi trường)")
    return p.parse_args()


def resolve_device(choice):
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main():
    args = parse_args()

    # Nạp .env nếu có (không bắt buộc)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    if args.api_key:
        os.environ["ROBOFLOW_API_KEY"] = args.api_key
    need_key = (not args.local) or args.field == "roboflow"
    if need_key and not os.environ.get("ROBOFLOW_API_KEY"):
        sys.exit("Lỗi: chưa có ROBOFLOW_API_KEY. Dùng Roboflow API (đặt key trong .env / --api-key) "
                 "hoặc chạy bằng weights local với cờ --local.")

    if not os.path.exists(args.source):
        sys.exit(f"Lỗi: không tìm thấy file đầu vào: {args.source}")

    # Tạo thư mục đích nếu cần
    out_dir = os.path.dirname(os.path.abspath(args.target))
    os.makedirs(out_dir, exist_ok=True)

    device = resolve_device(args.device)
    field_src = args.field or ("local" if args.local else "roboflow")
    print(f"Thiết bị: {device} | model người: {'local .pt' if args.local else 'Roboflow API'} "
          f"| model sân: {field_src}")

    from football_analysis import load_models, load_models_local, run_video, run_image

    print("Đang load model...")
    if args.local:
        player_model, field_model = load_models_local(args.weights, args.pitch_weights, device=device)
    else:
        player_model, field_model = load_models()

    # --field ghi đè riêng model sân (vd: cầu thủ local + sân hosted Roboflow)
    if args.field == "roboflow" and args.local:
        from inference import get_model
        from football_analysis.detection import FIELD_MODEL_ID
        field_model = get_model(model_id=FIELD_MODEL_ID,
                                api_key=os.environ["ROBOFLOW_API_KEY"])
    elif args.field == "local" and not args.local:
        _, field_model = load_models_local(args.weights, args.pitch_weights, device=device)

    if args.image:
        import cv2
        print("Đang xử lý ảnh...")
        composite = run_image(args.source, player_model, field_model, device=device,
                              show_radar=not args.no_radar, show_voronoi=not args.no_voronoi,
                              show_ball=not args.no_ball)
        cv2.imwrite(args.target, composite)
        print("✅ Đã lưu:", args.target)
    else:
        print("Đang xử lý video...")
        run_video(args.source, args.target, player_model, field_model,
                  max_frames=args.max_frames, fit_stride=args.fit_stride,
                  switch_ratio=args.switch_ratio, touch_px=args.touch_px, device=device,
                  show_radar=not args.no_radar, show_voronoi=not args.no_voronoi,
                  show_ball=not args.no_ball, out_fps=args.out_fps)


if __name__ == "__main__":
    main()