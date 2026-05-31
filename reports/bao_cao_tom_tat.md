# Báo cáo tóm tắt dự án: Phân tích trận đấu bóng đá bằng Computer Vision

## 1. Mục tiêu

Dự án xây dựng một hệ thống thị giác máy tính phân tích cảnh quay bóng đá phát sóng từ **một camera** (broadcast), với hai mức:

- **Nhận diện** bốn đối tượng trên sân: cầu thủ (*player*), bóng (*ball*), thủ môn (*goalkeeper*) và trọng tài (*referee*).
- **Phân tích chiến thuật**: phân biệt hai đội, dựng bản đồ 2D (radar) chiếu vị trí cầu thủ và bóng nhìn từ trên xuống, vẽ biểu đồ Voronoi thể hiện vùng kiểm soát của hai đội, và ước lượng tốc độ di chuyển tức thời của từng cầu thủ.

Đầu vào có thể là ảnh hoặc video; đầu ra là khung hình được chú giải kèm radar 2D và Voronoi hiển thị đồng thời theo phong cách đồ hoạa thể thao.

## 2. Dữ liệu

Dự án dùng hai bộ dữ liệu riêng cho hai bài toán, đều quản lý qua Roboflow:

- **Phát hiện đối tượng:** bộ `soccernet` (4 lớp), chia tập huấn luyện ~1.988 ảnh / 29.982 đối tượng và tập kiểm định ~1.989 ảnh / 28.213 đối tượng.
- **Keypoint sân:** bộ `football-field-detection` gồm các điểm mốc đặc trưng trên sân (giao điểm vạch kẻ), dùng để ước lượng phép biến đổi phối cảnh.

## 3. Kỹ thuật sử dụng

### 3.1. Hai mô hình nhận diện (huấn luyện riêng)

**Mô hình phát hiện cầu thủ/bóng** dựa trên kiến trúc **YOLOv11s**, huấn luyện theo lối transfer learning từ trọng số `yolo11s.pt` trong 32 epoch, ảnh đầu vào 1280px, batch 32; theo dõi quá trình bằng Weights & Biases.

**Mô hình keypoint sân** dùng **YOLOv11s-pose** (bài toán pose/keypoint), huấn luyện 100 epoch, ảnh 640px, batch 32, tắt augmentation mosaic. Mô hình này phát hiện các điểm mốc cố định trên sân làm cơ sở tính homography.

Cả hai mô hình sau khi huấn luyện được triển khai (deploy) lên Roboflow để gọi inference qua API.

### 3.2. Pipeline inference

Luồng xử lý mỗi khung hình kết hợp nhiều kỹ thuật:

- **Theo dõi đối tượng (tracking):** dùng ByteTrack để gán ID ổn định cho từng người qua các khung hình.
- **Phân đội:** trích đặc trưng vùng thân áo bằng embedding ảnh (SigLIP), giảm chiều bằng UMAP rồi phân cụm KMeans thành hai đội. Để chống hiện tượng nhãn đội nhấp nháy, hệ thống bỏ phiếu đa số theo ID theo thời gian; thủ môn được gán đội theo khoảng cách tới trọng tâm mỗi đội.
- **Ánh xạ 2D (homography):** từ các keypoint sân, ước lượng phép biến đổi phối cảnh đưa toạ độ ảnh về toạ độ sân nhìn từ trên xuống; keypoint được làm mượt theo thời gian để giảm rung.
- **Bản đồ radar & Voronoi:** chiếu vị trí cầu thủ/bóng lên sân chuẩn để dựng minimap, đồng thời vẽ biểu đồ Voronoi (có chuyển màu mượt giữa hai vùng kiểm soát).
- **Tốc độ cầu thủ:** tính quãng đường di chuyển trên toạ độ sân giữa các khung hình theo từng ID, làm sạch nhiễu và làm mượt, quy ra km/h hiển thị dưới chân mỗi cầu thủ.
- **Chú giải hình ảnh:** ellipse dưới chân người, tam giác trên quả bóng, nhãn thông tin; ghép radar (giữa, sát đáy) và Voronoi (bên cạnh) vào khung hình.

Pipeline video chạy theo cơ chế **hai lượt** (phân tích một lượt rồi hậu kỳ và vẽ ở lượt sau) để xử lý những vấn đề chỉ giải được khi có thông tin toàn cục, ví dụ làm thẳng quỹ đạo bóng bay trên bản đồ và ổn định số đo tốc độ.

## 4. Kết quả hiện tại

### 4.1. Mô hình phát hiện cầu thủ/bóng (YOLOv11s, trên tập kiểm định)

| Lớp | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---|---|---|---|
| Tất cả | 0.902 | 0.882 | 0.897 | 0.655 |
| player | 0.965 | 0.964 | 0.985 | 0.792 |
| referee | 0.941 | 0.968 | 0.982 | 0.788 |
| goalkeeper | 0.931 | 0.934 | 0.937 | 0.696 |
| ball | 0.772 | 0.664 | 0.686 | 0.346 |

Tốc độ suy luận khoảng 4–5 ms mỗi ảnh trên GPU. Mô hình nhận diện người (cầu thủ/trọng tài/thủ môn) rất tốt; **quả bóng là lớp khó nhất** do kích thước nhỏ, dễ bị che và chuyển động nhanh — đây cũng là nguyên nhân chính của hiện tượng bóng nhấp nháy mà pipeline inference phải xử lý thêm.

### 4.2. Mô hình keypoint sân (YOLOv11s-pose, trên tập kiểm định)

| Chỉ số | Box | Pose (keypoint) |
|---|---|---|
| mAP@50 | 0.995 | 0.995 |
| mAP@50–95 | 0.984 | 0.813 |
| Precision / Recall | 1.0 / 1.0 | 1.0 / 1.0 |

Mô hình keypoint đạt độ chính xác rất cao. Lưu ý tập kiểm định khá nhỏ (34 ảnh), nên các con số này nên được xác nhận thêm trên dữ liệu đa dạng hơn trước khi kết luận tổng quát.

### 4.3. Hệ thống inference

Hệ thống đã chạy hoàn chỉnh trên cả ảnh lẫn video, tạo ra khung hình chú giải kèm radar 2D và Voronoi hiển thị đồng thời, cùng tốc độ km/h dưới chân cầu thủ. Các vấn đề thực tế đã được xử lý: nhấp nháy nhãn đội, nhấp nháy/mất dấu bóng, rung bản đồ do keypoint, và quỹ đạo bóng bay bị vẽ cong trên bản đồ 2D (được làm thẳng bằng cách nối các thời điểm chạm bóng).

## 5. Hạn chế & hướng phát triển

- **Phát hiện bóng** còn yếu nhất trong bốn lớp; có thể cải thiện bằng dữ liệu bổ sung, tăng độ phân giải, hoặc mô hình chuyên cho vật thể nhỏ.
- **Ước lượng tốc độ và vị trí 2D** phụ thuộc chất lượng homography từ một camera — đây là ước lượng, không phải đo lường chính xác; độ cao của bóng khi bay không khôi phục được từ một góc nhìn.
- **Tập kiểm định keypoint nhỏ**, nên đánh giá lại trên nhiều trận/nhiều sân khác nhau.
- Hướng phát triển: chạy inference bằng trọng số tải về máy (thay vì gọi Roboflow API), đo thêm các chỉ số chiến thuật (cự ly đội hình, khoảng cách chuyền, kiểm soát khu vực), và tối ưu tốc độ để tiến gần thời gian thực.
