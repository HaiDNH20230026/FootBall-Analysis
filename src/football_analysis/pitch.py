"""Homography sân: làm mượt keypoint theo thời gian rồi dựng lại ViewTransformer."""
import numpy as np
import supervision as sv
from sports.common.view import ViewTransformer

from .config import CONF, KP_CONF, PITCH, HOMOGRAPHY_ALPHA


class HomographySmoother:
    """EMA trên vị trí keypoint sân (khớp theo chỉ số vertex) rồi dựng lại homography
    mỗi frame -> giảm rung, toạ độ cầu thủ/bóng trên sân ổn định hơn.
    """

    def __init__(self, field_model, alpha=HOMOGRAPHY_ALPHA):
        self.model = field_model
        self.alpha = alpha
        self.kp = None
        self.seen = None

    def update(self, frame):
        """Trả về ViewTransformer cho frame hiện tại, hoặc None nếu thiếu keypoint."""
        result = self.model.infer(frame, confidence=CONF)[0]
        if hasattr(result, "keypoints"):                  # Ultralytics Results
            kp = sv.KeyPoints.from_ultralytics(result)
        else:
            kp = sv.KeyPoints.from_inference(result)       # Roboflow inference
        xy = kp.xy[0]
        cur = kp.confidence[0] > KP_CONF

        if self.kp is None:
            self.kp = xy.astype(float).copy()
            self.seen = cur.copy()
        else:
            for i in range(len(xy)):
                if cur[i]:
                    if self.seen[i]:
                        self.kp[i] = self.alpha * xy[i] + (1 - self.alpha) * self.kp[i]
                    else:
                        self.kp[i] = xy[i]
                        self.seen[i] = True

        if cur.sum() < 4:
            return None
        return ViewTransformer(source=self.kp[cur], target=np.array(PITCH.vertices)[cur])
