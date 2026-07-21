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
- **Ánh xạ 2D ổn định và bám vạch sân — nguyên tắc "làm mượt phép đo, không làm mượt mô hình":** mỗi khung hình, homography được fit **least-squares trên toàn bộ keypoint** rồi mới loại các điểm lệch quá xa so với fit toàn cục (ngưỡng thích nghi theo trung vị residual) và fit lại — cách này giữ được "baseline" trải rộng khắp sân. Đề tài từng dùng RANSAC ở bước này và gặp một chế độ hỏng kinh điển: khi nhiễu keypoint (~15–35 px) vượt ngưỡng RANSAC, thuật toán khoá vào một cụm 5–6 điểm sát nhau (ví dụ cùng một vòng cấm) tự-nhất-quán rồi loại hết các điểm trải rộng — homography ngoại suy lệch 4–23 m ở phần sân còn lại mà cổng kiểm tra reprojection không phát hiện được (vì đo trên chính cụm điểm đó), sau đó cơ chế giữ-H-gần-nhất đóng băng đúng homography hỏng này trong suốt hơn một giây model mất sân. Sau bước fit, **chuỗi thời gian của từng keypoint trong ảnh** được lọc trung vị (chặn outlier) và làm mượt **có tâm** bằng Savitzky-Golay bậc 2 (bám được chuyển động lia nhanh mà không trễ như trung bình trượt nhân quả — khả thi nhờ kiến trúc hai lượt); keypoint chỉ được lấp khoảng trống ngắn trong phạm vi nó thực sự được quan sát, **không bao giờ ngoại suy ra ngoài khung hình**. Cuối cùng homography được **fit lại từng khung hình** từ các keypoint đã làm mượt; khung hình mất sân (mờ do lia nhanh) được **bắc cầu bằng dead-reckoning quang học**: vì camera phát sóng quay/zoom quanh trục cố định, chuyển động giữa hai khung hình liên tiếp là *một homography toàn cục* ước lượng được từ optical flow của toàn khung hình (Shi–Tomasi + Lucas–Kanade + RANSAC — RANSAC hợp lệ ở đây vì điểm nền tĩnh chiếm đa số áp đảo, khác với trường hợp fit keypoint) mà **không cần nhìn thấy vạch sân**; xích dồn các chuyển động này qua quãng mù giúp bản đồ tiếp tục bám camera. Trước đó đề tài dùng cách giữ nguyên homography gần nhất (hành vi của notebook tham chiếu): đo đạc cho thấy quãng mù dài nhất của video mẫu kéo dài 3,6 giây trong lúc camera vẫn lia — homography đóng băng làm cụm điểm lệch/co nhiều mét giữa quãng; còn nội suy ma trận xuyên quãng trống (thử trước nữa) gây méo dần. Dead-reckoning giảm lệch tại điểm bắt lại sân từ 2,8–6,0 m xuống 1,3–3,6 m và giữ vạch sân chiếu ngược khớp hình ảnh thật ở giữa quãng mù. Vì homography fit trực tiếp từ quan sát, nó tự hấp thụ các sai lệch hệ thống của thiết bị thật (méo ống kính, điểm chính lệch tâm, template sân chuẩn 120×70 m khác kích thước sân thật ~105×68 m) — chấm trên minimap bám vạch sân đúng như chất lượng model keypoint cho phép. Trên video mẫu: đoạn tĩnh fit trực tiếp 150/150 khung hình, sai số so với vạch sân ~1,1 m (trung vị); đoạn lia khó nhất (model keypoint mất sân ở 114/180 khung hình) đạt ~1,3 m trên các khung hình fit được — tốt hơn mọi phương án dựa trên mô hình camera đã thử. Nút thắt còn lại nằm ở **độ phủ của model keypoint** khi camera lia nhanh (mờ chuyển động), không phải ở thuật toán hậu xử lý. Trước khi đi đến thiết kế này, đề tài đã thử và loại qua đo đạc: (1) EMA nhân quả trên keypoint — trễ gây "shift" khi lia; (2) làm mượt trực tiếp các phần tử ma trận homography — không bảo toàn cấu trúc xạ ảnh, cụm điểm co giãn kiểu "thở"; (3) phân rã homography thành tham số camera vật lý (kiểu Zhang) rồi làm mượt tham số, kể cả biến thể khoá vị trí camera và trường hiệu chỉnh — mô hình pinhole lý tưởng lệch hệ thống vài mét so với thực tế (nặng nhất ở vùng xa khung hình), không khắc phục triệt để được. Bài học này nhất quán với việc các pipeline GSR hàng đầu phải *huấn luyện* mạng ước lượng tham số camera trên dữ liệu lớn thay vì giải tích thuần.
- **Bản đồ radar & Voronoi:** chiếu vị trí cầu thủ/bóng lên sân chuẩn để dựng minimap, đồng thời vẽ biểu đồ Voronoi (có chuyển màu mượt giữa hai vùng kiểm soát). Để **chống nhấp nháy (flicker)** trên minimap, chuỗi vị trí sân của từng track (theo ID ByteTrack) được hậu xử lý ở lượt hai: lấp khoảng trống ngắn khi detection chớp tắt (nội suy ≤ 1 giây, chỉ trong khoảng track thực sự được quan sát — không vẽ "bóng ma" khi track mất hẳn) và làm mượt có tâm (lọc trung vị + Savitzky-Golay); khung hình gốc vẫn chú giải theo detection thật. Mỗi lớp overlay (radar, Voronoi, quả bóng) đều **bật/tắt được khi inference**. Ngoài ra hệ thống hỗ trợ **chế độ hybrid** chọn nguồn từng model riêng biệt (`--field`): phát hiện cầu thủ bằng weights tự huấn luyện chạy local, còn keypoint sân gọi model hosted huấn luyện trên dataset lớn — trên video mẫu, model sân hosted bám sân 750/750 khung hình (so với 588/750 của model tự huấn luyện với 34 ảnh kiểm định), cho bản đồ 2D ổn định hơn rõ rệt; đây cũng là minh chứng định lượng rằng nút thắt nằm ở dữ liệu huấn luyện model keypoint.
- **Tốc độ cầu thủ:** tính quãng đường di chuyển trên toạ độ sân giữa các khung hình theo từng ID, làm sạch nhiễu và làm mượt, quy ra km/h hiển thị dưới chân mỗi cầu thủ.
- **Chú giải hình ảnh:** ellipse dưới chân người, tam giác trên quả bóng, nhãn thông tin; ghép radar (giữa, sát đáy) và Voronoi (bên cạnh) vào khung hình. Có tuỳ chọn **điều chỉnh FPS video kết quả** (lấy/nhân mẫu khung hình theo thời gian, giữ nguyên thời lượng).

Pipeline video chạy theo cơ chế **hai lượt** (phân tích một lượt rồi làm mượt homography + hậu kỳ + vẽ ở lượt sau) để xử lý những vấn đề chỉ giải được khi có thông tin toàn cục, ví dụ làm mượt homography có tâm khi camera lia, làm thẳng quỹ đạo bóng bay trên bản đồ và ổn định số đo tốc độ.

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

Hệ thống đã chạy hoàn chỉnh trên cả ảnh lẫn video, tạo ra khung hình chú giải kèm radar 2D và Voronoi hiển thị đồng thời, cùng tốc độ km/h dưới chân cầu thủ. Các vấn đề thực tế đã được xử lý: **sót cầu thủ ở xa do lệch độ phân giải suy luận** (mô hình huấn luyện ở 1280 px nhưng thư viện mặc định suy luận ở 640 px — đo trên video mẫu chỉ thấy 13–19/~24 người mỗi khung hình; sau khi khớp lại 1280 px thấy 22–25 người, chi phí thêm ~20 ms/khung hình); nhấp nháy vị trí trên minimap (nội suy + làm mượt chuỗi vị trí theo từng ID tracking); nhấp nháy nhãn đội; nhấp nháy/mất dấu bóng (tracker tự bắt lại bóng sau khi mất dấu, và **ẩn bóng trên bản đồ khi mất dấu lâu** thay vì đóng băng ở vị trí cuối); **bản đồ giật/rung/lệch vạch sân** (fit toàn bộ keypoint + cắt outlier so với fit toàn cục, làm mượt có tâm chuỗi keypoint rồi fit lại homography từng khung hình); và quỹ đạo bóng bay bị vẽ cong trên bản đồ 2D (làm thẳng bằng cách nối các thời điểm chạm bóng). Inference cũng cho phép bật/tắt từng overlay (radar / Voronoi / quả bóng) và chỉnh FPS video kết quả.

## 5. Hạn chế & hướng phát triển

- **Phát hiện bóng** còn yếu nhất trong bốn lớp; có thể cải thiện bằng dữ liệu bổ sung, tăng độ phân giải, hoặc mô hình chuyên cho vật thể nhỏ.
- **Ước lượng tốc độ và vị trí 2D** phụ thuộc chất lượng homography từ một camera — đây là ước lượng, không phải đo lường chính xác; độ cao của bóng khi bay không khôi phục được từ một góc nhìn (homography chỉ đúng cho điểm nằm trên mặt sân). Đo đạc chi tiết cho thấy giới hạn hiện tại nằm ở **độ chính xác định vị của model keypoint**: trong một khung nhìn, các keypoint tự mâu thuẫn nhau 10–22 px một cách *có hệ thống* (ổn định qua các khung hình liền kề, không tương quan với confidence), tương đương lệch ~2–5 m ở vùng xa cụm điểm neo — không thuật toán hậu xử lý nào bù được vì sai lệch nằm ngay trong phép đo. Đã loại trừ bằng thực nghiệm các nguyên nhân khác: tỉ lệ template sân (fit với 105×68 m cho residual tệ hơn 120×70 m; tối ưu hoá kích thước template trên nhiều khung hình hội tụ về ≈ template hiện tại) và trọng số theo confidence (fit chỉ bằng điểm confidence cao khái quát hoá kém hơn).
- **Tập kiểm định keypoint nhỏ**, nên đánh giá lại trên nhiều trận/nhiều sân khác nhau.
- Hướng phát triển:
  - Tiến tới **ước lượng tham số camera bằng mạng học sâu** huấn luyện trên dữ liệu lớn (hướng của các pipeline GSR đoạt giải SoccerNet) — bền hơn khi chỉ thấy ít vạch sân lúc lia; phiên bản giải tích thuần đã được thử trong đề tài và cho thấy không đủ chính xác với ống kính/template thực tế.
  - Khi đã có hiệu chỉnh camera đầy đủ: khôi phục **quỹ đạo bóng 3D** bằng khớp parabol (vận tốc ngang đều + trọng lực) để vẽ đúng vị trí bóng trên sân kể cả khi bay, thay cho phép nội suy theo lần chạm hiện tại.
  - Đo thêm các chỉ số chiến thuật (cự ly đội hình, khoảng cách chuyền, kiểm soát khu vực) và tối ưu tốc độ để tiến gần thời gian thực.
