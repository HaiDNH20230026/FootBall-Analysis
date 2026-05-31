"""Vẽ: ellipse/tam giác/label, radar 2D, Voronoi, và ghép overlay kiểu FIFA."""
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

from .config import PITCH, COLOR_TEAM_1, COLOR_TEAM_2, COLOR_REFEREE

# --- Annotators dùng cho khung hình gốc ---
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex([f"#{COLOR_TEAM_1}", f"#{COLOR_TEAM_2}", f"#{COLOR_REFEREE}"]),
    thickness=2,
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex([f"#{COLOR_TEAM_1}", f"#{COLOR_TEAM_2}", f"#{COLOR_REFEREE}"]),
    text_color=sv.Color.from_hex("#000000"),
    text_position=sv.Position.BOTTOM_CENTER,
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex(f"#{COLOR_REFEREE}"),
    base=25, height=21, outline_thickness=1,
)


def draw_pitch_voronoi_diagram_2(config, team_1_xy, team_2_xy,
                                 team_1_color=sv.Color.RED, team_2_color=sv.Color.WHITE,
                                 opacity=0.5, padding=50, scale=0.1,
                                 pitch: Optional[np.ndarray] = None):
    """Voronoi với chuyển màu mượt giữa 2 vùng kiểm soát."""
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    c1 = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    c2 = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    ys, xs = np.indices((scaled_width + 2 * padding, scaled_length + 2 * padding))
    ys -= padding
    xs -= padding

    def dist(xy):
        return np.sqrt((xy[:, 0][:, None, None] * scale - xs) ** 2 +
                       (xy[:, 1][:, None, None] * scale - ys) ** 2)

    d1 = np.min(dist(team_1_xy), axis=0)
    d2 = np.min(dist(team_2_xy), axis=0)

    steepness = 15
    ratio = d2 / np.clip(d1 + d2, a_min=1e-5, a_max=None)
    blend = np.tanh((ratio - 0.5) * steepness) * 0.5 + 0.5

    for c in range(3):
        voronoi[:, :, c] = (blend * c1[c] + (1 - blend) * c2[c]).astype(np.uint8)

    return cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)


def build_radar(ball_xy, team0_xy, team1_xy, ref_xy):
    """Radar 2D kiểu minimap."""
    pitch = draw_pitch(PITCH)
    if len(team0_xy):
        pitch = draw_points_on_pitch(PITCH, xy=team0_xy, face_color=sv.Color.from_hex(COLOR_TEAM_1),
                                     edge_color=sv.Color.BLACK, radius=16, pitch=pitch)
    if len(team1_xy):
        pitch = draw_points_on_pitch(PITCH, xy=team1_xy, face_color=sv.Color.from_hex(COLOR_TEAM_2),
                                     edge_color=sv.Color.BLACK, radius=16, pitch=pitch)
    if len(ref_xy):
        pitch = draw_points_on_pitch(PITCH, xy=ref_xy, face_color=sv.Color.from_hex(COLOR_REFEREE),
                                     edge_color=sv.Color.BLACK, radius=16, pitch=pitch)
    if len(ball_xy):
        pitch = draw_points_on_pitch(PITCH, xy=ball_xy, face_color=sv.Color.WHITE,
                                     edge_color=sv.Color.BLACK, radius=10, pitch=pitch)
    return pitch


def build_voronoi(team0_xy, team1_xy, ball_xy):
    """Panel Voronoi (nền trắng)."""
    pitch = draw_pitch(PITCH, background_color=sv.Color.WHITE, line_color=sv.Color.BLACK)
    pitch = draw_pitch_voronoi_diagram_2(
        PITCH, team_1_xy=team0_xy, team_2_xy=team1_xy,
        team_1_color=sv.Color.from_hex(COLOR_TEAM_1),
        team_2_color=sv.Color.from_hex(COLOR_TEAM_2), pitch=pitch)
    if len(team0_xy):
        pitch = draw_points_on_pitch(PITCH, xy=team0_xy, face_color=sv.Color.from_hex(COLOR_TEAM_1),
                                     edge_color=sv.Color.WHITE, radius=16, thickness=1, pitch=pitch)
    if len(team1_xy):
        pitch = draw_points_on_pitch(PITCH, xy=team1_xy, face_color=sv.Color.from_hex(COLOR_TEAM_2),
                                     edge_color=sv.Color.WHITE, radius=16, thickness=1, pitch=pitch)
    if len(ball_xy):
        pitch = draw_points_on_pitch(PITCH, xy=ball_xy, face_color=sv.Color.WHITE,
                                     edge_color=sv.Color.WHITE, radius=8, thickness=1, pitch=pitch)
    return pitch


def _resize_w(img, width):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def _overlay(scene, panel, x, y, alpha):
    H, W = scene.shape[:2]
    ph, pw = panel.shape[:2]
    if pw >= W or ph >= H:
        return scene
    x = max(0, min(x, W - pw))
    y = max(0, min(y, H - ph))
    roi = scene[y:y + ph, x:x + pw]
    scene[y:y + ph, x:x + pw] = cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0)
    return scene


def compose_frame(annotated, radar=None, voronoi=None,
                  radar_scale=0.40, voronoi_scale=0.26,
                  alpha=0.85, margin=20, voronoi_side="right"):
    """Bố cục kiểu FIFA: radar giữa-đáy, Voronoi bên cạnh."""
    out = annotated.copy()
    H, W = out.shape[:2]

    if voronoi is not None:
        v = _resize_w(voronoi, int(W * voronoi_scale))
        vh, vw = v.shape[:2]
        vx = (W - vw - margin) if voronoi_side == "right" else margin
        out = _overlay(out, v, vx, H - vh - margin, alpha)

    if radar is not None:
        r = _resize_w(radar, int(W * radar_scale))
        rh, rw = r.shape[:2]
        out = _overlay(out, r, (W - rw) // 2, H - rh - margin, alpha)

    return out
