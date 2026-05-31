# Dữ liệu

> Nội dung thư mục này **không commit vào Git** (trừ README và `samples/.gitkeep`).

## Cấu trúc đề xuất

```
data/
├── samples/      # video/ảnh ngắn để chạy thử inference (đặt file của bạn vào đây)
├── raw/          # dataset gốc tải từ Roboflow (yolov11 format)
└── processed/    # dữ liệu đã tiền xử lý (nếu có)
```

## Dataset (qua Roboflow)

- **Phát hiện đối tượng:** `football-players-detection-3zvbc` — 4 lớp
  (player, ball, goalkeeper, referee). ~1.988 ảnh train / ~1.989 ảnh val.
- **Keypoint sân:** `football-field-detection` — các điểm mốc giao vạch kẻ sân.

Tải bằng API (cần `ROBOFLOW_API_KEY` trong `.env`):

```python
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
dataset = project.version(12).download("yolov11", location="data/raw")
```
