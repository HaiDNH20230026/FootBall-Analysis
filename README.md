# ⚽ Football Analysis — Nhận diện & mapping 2D trận đấu bóng đá từ video

Hệ thống thị giác máy tính phân tích cảnh quay bóng đá phát sóng (broadcast, một
camera): nhận diện cầu thủ / bóng / thủ môn / trọng tài, phân biệt hai đội, dựng
**bản đồ 2D (radar)** nhìn từ trên xuống, vẽ **biểu đồ Voronoi** vùng kiểm soát và
ước lượng **tốc độ (km/h)** của từng cầu thủ.

> Project 2 — Trường CNTT & TT, ĐHBK Hà Nội. GVHD: Nguyễn Quốc Tuấn.
> SV: Đỗ Ngọc Hoàng Hải (20230026). Báo cáo đầy đủ: [`reports/Project2.pdf`](reports/Project2.pdf).

---

## ✨ Tính năng

- Phát hiện 4 lớp đối tượng bằng **YOLOv11s** (player, ball, goalkeeper, referee).
- Phát hiện keypoint sân bằng **YOLOv11s-pose** → ước lượng **homography**.
- Theo dõi ID ổn định bằng **ByteTrack**.
- Phân đội theo màu áo: **SigLIP** embedding → **UMAP** → **KMeans**, kèm bỏ phiếu
  đa số + hysteresis để chống nhấp nháy nhãn.
- **Radar 2D** + **Voronoi** chiếu lên sân chuẩn.
- Ước lượng tốc độ trên toạ độ sân thực (m → km/h).
- Xử lý quả bóng: chống nhấp nháy (EMA + coasting) và làm thẳng quỹ đạo bay trên
  bản đồ 2D bằng nội suy giữa các lần chạm bóng.
- Kiến trúc **two-pass**: lượt 1 phân tích & lưu dữ liệu gọn nhẹ, lượt 2 hậu kỳ & vẽ.

---

## 📁 Cấu trúc dự án

```
football-analysis/
├── configs/                 # Cấu hình pipeline (đường dẫn, ngưỡng, cửa sổ làm mượt)
│   └── default.yaml
├── data/                    # Dataset (gitignored) + video/ảnh mẫu
│   ├── samples/             #   → đặt video/ảnh đầu vào để thử nghiệm
│   └── README.md
├── models/                  # Trọng số .pt (gitignored — xem models/README.md)
│   └── README.md
├── notebooks/               # Notebook huấn luyện & demo (Colab)
│   ├── football_analysis.ipynb         # Bản Colab gốc (toàn bộ pipeline)
│   ├── football_inference.ipynb        # Pipeline inference đã dọn (nguồn của src/)
│   ├── player_dection.ipynb            # Train YOLOv11s detection  (*nên đổi tên → player_detection*)
│   └── pitch_keypoint_detection.ipynb  # Train YOLOv11s-pose keypoint sân
├── outputs/                 # Kết quả sinh ra (gitignored)
├── reports/                 # Báo cáo PDF, hình minh hoạ, metrics
│   ├── Project2.pdf
│   ├── figures/ · metrics/
│   └── bao_cao_tom_tat.md
├── scripts/                 # Điểm chạy CLI
│   ├── run_inference.py            # Chạy inference ảnh/video
│   ├── train_player_detection.py
│   └── train_pitch_keypoints.py
├── src/football_analysis/   # Package chính (pip install -e .)
│   ├── config.py            # Hằng số: class id, ngưỡng, màu, SoccerPitchConfiguration
│   ├── detection.py         # load_models() + detect() (Roboflow API / YOLO local)
│   ├── tracking.py          # BallTracker (chống nhấp nháy bóng, toạ độ ảnh)
│   ├── team.py              # SigLIP+UMAP+KMeans + TeamResolver + gán thủ môn
│   ├── pitch.py             # HomographySmoother (EMA keypoint → ViewTransformer)
│   ├── ball.py              # smooth_ball_path (làm thẳng quỹ đạo theo lần chạm)
│   ├── speed.py             # compute_speeds (km/h theo tracker_id)
│   ├── annotate.py          # ellipse/triangle/label + build_radar/voronoi/compose_frame
│   └── pipeline.py          # analyze_video → render_video (two-pass) + run_image
├── tests/
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## 🚀 Cài đặt

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/bin/activate     # macOS / Linux

# 2. Cài dependency
pip install -r requirements.txt
pip install -e .                # cài package football_analysis ở chế độ editable

# 3. Khai báo khóa API
copy .env.example .env          # rồi điền ROBOFLOW_API_KEY / WANDB_API_KEY

# 4. Tải trọng số model (xem models/README.md) vào thư mục models/
```

---

## 🏃 Inference

### Chọn nguồn model

| Cách | Cờ | Yêu cầu | Ghi chú |
|------|----|---------|---------|
| **Local weights** | `--local` | `models/players.pt`, `models/pitch.pt` | Chạy offline, **không cần API key** (khuyến nghị). |
| **Roboflow API** | *(mặc định)* | `ROBOFLOW_API_KEY` trong `.env` | Tự tải weights về chạy local lần đầu. |

> Lần chạy đầu (cả 2 cách) tải model **SigLIP (~400 MB)** từ HuggingFace để phân đội.

### Ví dụ nhanh

```bash
# VIDEO, weights local — test nhanh 40 frame (bỏ --max-frames để render toàn bộ)
python scripts/run_inference.py --local \
    --source data/samples/2e57b9_0.mp4 --target outputs/out.mp4 --max-frames 40

# ẢNH, weights local (không có tốc độ, không two-pass)
python scripts/run_inference.py --local --image \
    --source data/samples/frame.jpg --target outputs/out.jpg

# Dùng Roboflow API (bỏ --local; cần ROBOFLOW_API_KEY)
python scripts/run_inference.py \
    --source data/samples/2e57b9_0.mp4 --target outputs/out.mp4

# Tinh chỉnh ngay trên dòng lệnh
python scripts/run_inference.py --local --source in.mp4 --target out.mp4 \
    --device cuda --switch-ratio 0.6 --touch-px 80 --fit-stride 15
```

### Tham số CLI — `scripts/run_inference.py`

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--source` | *(bắt buộc)* | Đường dẫn video hoặc ảnh đầu vào. |
| `--target` | *(bắt buộc)* | Đường dẫn file kết quả (tự tạo thư mục cha). |
| `--image` | tắt | Xử lý **ảnh** thay vì video (bỏ qua tốc độ + two-pass). |
| `--local` | tắt | Dùng weights `.pt` local (Ultralytics) thay cho Roboflow API. |
| `--weights` | `models/players.pt` | Weights detection (chỉ khi `--local`). |
| `--pitch-weights` | `models/pitch.pt` | Weights keypoint sân (chỉ khi `--local`). |
| `--api-key` | lấy từ `.env` | Roboflow API key (chỉ đường API). |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda`. `auto` = có GPU thì dùng CUDA. |
| `--max-frames` | hết video | Giới hạn số frame xử lý (test nhanh). |
| `--fit-stride` | `20` | Bước lấy frame để gom crop **fit phân đội** (nhỏ hơn = nhiều mẫu hơn, chậm hơn). |
| `--switch-ratio` | `0.5` | Hysteresis chống nhấp nháy đội (cao hơn = "lì" hơn, đổi đội khó hơn). |
| `--touch-px` | `70` | Ngưỡng khoảng cách bóng–bàn chân coi là "chạm bóng" (px, ~1080p). |

> CLI chỉ phủ một số tham số hay đổi. Toàn bộ tham số còn lại nằm trong
> [`configs/default.yaml`](configs/default.yaml) (CLI sẽ ưu tiên hơn YAML cho các cờ trùng).

### Tham số trong `configs/default.yaml`

Sửa file này là cả package dùng theo (không cần sửa code). Đổi file khác bằng biến môi
trường `FOOTBALL_CONFIG=path.yaml` hoặc `football_analysis.config.reload("path.yaml")`.

| Khoá | Mặc định | Ý nghĩa / khi nào chỉnh |
|------|----------|-------------------------|
| `models.detection` | `models/players.pt` | Weights phát hiện 4 lớp (đường `--local`). |
| `models.pitch_keypoints` | `models/pitch.pt` | Weights keypoint sân. |
| `inference.device` | `auto` | Thiết bị mặc định (`auto`/`cuda`/`cpu`). |
| `inference.conf_threshold` | `0.3` | Ngưỡng confidence detection. Giảm nếu sót bóng/cầu thủ; tăng nếu nhiều box rác. |
| `inference.kp_conf` | `0.5` | Ngưỡng keypoint sân để tính homography (cần ≥4 điểm/frame). |
| `inference.nms_iou` | `0.5` | IoU của NMS gộp box người trùng. |
| `video.fps` | `25` | FPS dùng quy đổi km/h (khi render video sẽ lấy fps thật của file). |
| `team_classifier.fit_stride` | `20` | = `--fit-stride`. |
| `team_classifier.switch_ratio` | `0.5` | = `--switch-ratio`. |
| `homography.ema_alpha` | `0.3` | Làm mượt keypoint sân theo thời gian. **Nhỏ hơn = mượt/ổn định hơn** (bản đồ đỡ rung) nhưng phản ứng chậm khi camera lia. |
| `speed.pos_smooth` | `5` | Cửa sổ làm mượt vị trí trước khi tính tốc độ. |
| `speed.spd_window` | `5` | Cửa sổ vi sai trung tâm để ra km/h (lớn hơn = mượt hơn). |
| `speed.max_player_ms` | `12.0` | Loại bước nhảy phi lý do tráo ID (m/s). |
| `speed.max_kmh` | `40` | Chặn trên tốc độ để cắt đỉnh nhiễu. |
| `ball.touch_px` | `70.0` | = `--touch-px`. |
| `output.dir` | `outputs/` | Thư mục kết quả mặc định. |

### Mẹo tinh chỉnh theo lỗi hay gặp

- **Nhãn đội nhấp nháy** → tăng `switch_ratio` (0.6) và/hoặc giảm `fit_stride` (15).
- **Bóng trên radar vẽ vòng cung** (lúc bay) → tăng `touch_px`; **bị bẻ thẳng nhầm khi rê dắt** → giảm `touch_px`.
- **Bản đồ/radar rung** → giảm `homography.ema_alpha` (vd 0.2).
- **Tốc độ giật/nhảy số** → tăng `speed.pos_smooth` & `speed.spd_window`, hoặc giảm `speed.max_player_ms`.
- **Sót bóng / cầu thủ** → giảm `inference.conf_threshold` (vd 0.2).
- **Bố cục overlay** (kích thước/vị trí radar & Voronoi) → sửa `compose_frame()` trong
  [annotate.py](src/football_analysis/annotate.py): `radar_scale`, `voronoi_scale`, `margin`, `alpha`, `voronoi_side`.

### Gọi trong code Python

```python
from football_analysis import load_models_local, run_video, run_image

player_model, field_model = load_models_local()           # weights .pt local
# player_model, field_model = load_models()               # hoặc Roboflow API (cần key)

# Video: trả về đường dẫn file đã ghi
run_video("data/samples/2e57b9_0.mp4", "outputs/out.mp4", player_model, field_model,
          max_frames=40, switch_ratio=0.6, touch_px=80)

# Ảnh: trả về khung hình BGR (numpy) đã chú giải
# composite = run_image("frame.jpg", player_model, field_model)
```

### Huấn luyện lại (tuỳ chọn)

```bash
python scripts/train_player_detection.py --epochs 32 --imgsz 1280 --batch 32   # detection
python scripts/train_pitch_keypoints.py  --epochs 100 --imgsz 640 --batch 32   # keypoint sân
```
Cần `ROBOFLOW_API_KEY` (tải dataset); thêm `--wandb` để theo dõi, `--deploy` để đẩy weights
lên Roboflow. Xem thêm `--help` của từng script.

---

## 📊 Kết quả (trên tập kiểm định)

**YOLOv11s — phát hiện 4 lớp**

| Lớp        | Precision | Recall | mAP@50 | mAP@50–95 |
|------------|-----------|--------|--------|-----------|
| Tất cả     | 0.902     | 0.882  | 0.897  | 0.655     |
| player     | 0.965     | 0.964  | 0.985  | 0.792     |
| referee    | 0.941     | 0.968  | 0.982  | 0.788     |
| goalkeeper | 0.931     | 0.934  | 0.937  | 0.696     |
| ball       | 0.772     | 0.664  | 0.686  | 0.346     |

**YOLOv11s-pose — keypoint sân** (lưu ý: tập kiểm định nhỏ, 34 ảnh)

| Chỉ số            | Box       | Pose (keypoint) |
|-------------------|-----------|-----------------|
| mAP@50            | 0.995     | 0.995           |
| mAP@50–95         | 0.984     | 0.813           |
| Precision/Recall  | 1.0 / 1.0 | 1.0 / 1.0       |

---

## ⚠️ Hạn chế & hướng phát triển

- Phát hiện **bóng** yếu nhất (vật thể nhỏ, dễ bị che) → bổ sung dữ liệu, tăng độ
  phân giải, hoặc model chuyên cho vật thể nhỏ.
- Vị trí 2D & tốc độ phụ thuộc chất lượng homography từ một camera — là **ước lượng**,
  không khôi phục được độ cao bóng khi bay.
- Tập kiểm định keypoint nhỏ → cần đánh giá trên nhiều trận/sân khác nhau.
- Chuyển từ gọi API sang chạy trọng số tải về máy; bổ sung chỉ số chiến thuật
  (cự ly đội hình, khoảng chuyền, kiểm soát khu vực); tối ưu tốc độ thời gian thực.

---

## 🛠️ Công nghệ

YOLOv11 · ByteTrack · SigLIP · UMAP · KMeans · OpenCV · Roboflow · Weights & Biases
