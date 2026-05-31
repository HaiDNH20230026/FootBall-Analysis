# Notebooks

Notebook chạy trên **Google Colab** (mount Drive, dùng Colab Secrets cho API key).
Code inference đã ổn định được tách thành package `src/football_analysis/`.

| Notebook                          | Vai trò                                                        |
|-----------------------------------|---------------------------------------------------------------|
| `football_inference.ipynb`        | Pipeline inference đã dọn (ảnh + video, two-pass) — **nguồn của `src/`** |
| `football_analysis.ipynb`         | Bản Colab gốc, khám phá toàn bộ pipeline từng bước            |
| `player_dection.ipynb`            | Huấn luyện YOLOv11s phát hiện player/ball/GK/referee           |
| `pitch_keypoint_detection.ipynb`  | Huấn luyện YOLOv11s-pose keypoint sân                          |

## Lưu ý

- Đổi tên `player_dection.ipynb` → `player_detection.ipynb` (sửa lỗi chính tả).
- `football_analysis.ipynb` nặng ~10 MB và `player_dection.ipynb` ~7 MB do **ảnh kết quả
  nhúng trong output**. Xoá output trước khi commit để repo gọn:
  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
  ```
  (hoặc cài `nbstripout` để tự xoá output mỗi lần commit). Hình minh hoạ nên để trong
  `reports/figures/`.
- Hai notebook training cài `ultralytics<=8.3.40`; bản đó cố định cho Colab — khi chạy
  local hãy theo `requirements.txt`.
