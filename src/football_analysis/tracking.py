"""Theo dõi bóng ở toạ độ ảnh: chống flicker và lấp khoảng trống khi mất bóng."""
import numpy as np


class BallTracker:
    """Chọn detection gần vị trí trước nhất (loại box nhiễu), coast khi mất bóng,
    và làm mượt EMA."""

    def __init__(self, max_coast=8, max_jump=250, ema=0.5):
        self.max_coast = max_coast   # số frame tối đa được giữ khi mất bóng
        self.max_jump = max_jump     # px: loại detection nhảy quá xa so với vị trí trước
        self.ema = ema
        self.xy = None               # tâm bóng đã mượt
        self.wh = None               # kích thước box gần nhất
        self.miss = 0

    def update(self, ball_detections):
        """Trả về xyxy dạng (1, 4) để vẽ/transform, hoặc None nếu mất bóng quá lâu."""
        cand = None
        if len(ball_detections):
            xyxy = ball_detections.xyxy
            cxcy = (xyxy[:, :2] + xyxy[:, 2:]) / 2
            if self.xy is None:
                idx = int(np.argmax(ball_detections.confidence))  # lần đầu: conf cao nhất
            else:
                d = np.linalg.norm(cxcy - self.xy, axis=1)
                idx = int(np.argmin(d))
                if d[idx] > self.max_jump:                        # nhảy quá xa -> nhiễu
                    idx = None
            if idx is not None:
                cand = xyxy[idx]

        if cand is not None:
            c = (cand[:2] + cand[2:]) / 2
            wh = cand[2:] - cand[:2]
            if self.xy is None:
                self.xy, self.wh = c, wh
            else:
                self.xy = self.ema * c + (1 - self.ema) * self.xy
                self.wh = self.ema * wh + (1 - self.ema) * self.wh
            self.miss = 0
        else:
            self.miss += 1
            if self.xy is None or self.miss > self.max_coast:
                return None

        x, y = self.xy
        w, h = self.wh
        return np.array([[x - w / 2, y - h / 2, x + w / 2, y + h / 2]])
