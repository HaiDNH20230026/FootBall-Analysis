# Trọng số model (weights)

> Các file `.pt` **không commit vào Git** (xem `.gitignore`). Thư mục này chỉ giữ
> hướng dẫn lấy weights.

## Các file cần có

| File         | Bài toán                              | Kiến trúc      | ~Dung lượng |
|--------------|---------------------------------------|----------------|-------------|
| `players.pt` | player / ball / goalkeeper / referee  | YOLOv11s       | ~19 MB      |
| `pitch.pt`   | keypoint sân (homography)             | YOLOv11s-pose  | ~21 MB      |

> Một model `players.pt` phát hiện cả 4 lớp (quả bóng là `class_id = 0`), không có
> file model bóng riêng. Tên file khớp với `configs/default.yaml` và phần ghi chú
> trong `src/football_analysis/detection.py`.

## Cách lấy weights

1. **Gọi qua Roboflow API (mặc định hiện tại).** `detection.load_models()` tải tham số
   trực tiếp từ Roboflow, không cần file local — chỉ cần `ROBOFLOW_API_KEY` trong `.env`.
   - `PLAYER_MODEL_ID = "soccernet-1ae2v-e6wqy/3"`
   - `FIELD_MODEL_ID  = "football-field-detection-f07vi/15"`

2. **Chạy bằng weights local.** Huấn luyện lại bằng `scripts/` (cần dataset Roboflow):
   ```bash
   python scripts/train_player_detection.py   # -> runs/detect/.../weights/best.pt -> đổi tên players.pt
   python scripts/train_pitch_keypoints.py    # -> runs/pose/.../weights/best.pt   -> đổi tên pitch.pt
   ```
   rồi sửa `load_models()` trả về `YOLO("models/players.pt"), YOLO("models/pitch.pt")`
   (gợi ý đã có sẵn trong `detection.py`).

## Provenance

- Dataset: `soccernet` (4 lớp) và `football-field-detection` (keypoint sân) trên Roboflow.
- Chi tiết huấn luyện: `reports/Project2.pdf` §3 và notebook tương ứng trong `notebooks/`.
