"""CCTV 위에 히트맵·궤적·임시보행로를 그린다."""

from __future__ import annotations

from typing import Optional

import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from detector import EQUIPMENT, WORKER, interpolate_snapshot

_HERE = os.path.dirname(os.path.abspath(__file__))
FONT_CANDIDATES = [
    os.path.join(_HERE, "fonts", "NanumGothic.ttf"),
    r"C:\Windows\Fonts\malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
]


def _font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy, text, fill, bg, size=16):
    font = _font(size)
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
        fill=bg,
    )
    draw.text((x, y), text, fill=fill, font=font)


def heatmap_overlay(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.58, floor: float = 0.16) -> np.ndarray:
    heat = np.clip(heatmap.astype(np.float32), 0.0, None)
    mx = float(heat.max())
    if mx <= 1e-6:
        return frame_bgr
    heat = heat / mx
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = np.clip((heat - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0) * alpha
    mask3 = mask[:, :, None]
    blended = frame_bgr.astype(np.float32) * (1.0 - mask3) + color.astype(np.float32) * mask3
    return blended.astype(np.uint8)


def render_scene(
    frame_bgr: np.ndarray,
    trajectories: pd.DataFrame,
    heatmap: Optional[np.ndarray],
    t: int,
    show_heatmap: bool = True,
    show_traj: bool = True,
    walkways: Optional[list] = None,
    current_band=None,
    chosen: Optional[dict] = None,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if show_heatmap and heatmap is not None and heatmap.max() > 0:
        canvas = heatmap_overlay(canvas, heatmap)

    trail = trajectories
    if trail is not None and not trail.empty:
        trail = trail[trail["t"] <= t]

    if show_traj and trail is not None and not trail.empty:
        for obj_id, g in trail.groupby("id"):
            g = g.sort_values("t")
            pts = g[["x", "y"]].to_numpy().astype(int)
            if len(pts) < 2:
                continue
            color = (220, 90, 40) if g["class"].iloc[0] == WORKER else (40, 160, 255)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
            cv2.arrowedLine(
                canvas,
                tuple(pts[max(0, len(pts) - 4)]),
                tuple(pts[-1]),
                color,
                2,
                tipLength=0.25,
            )

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    if current_band is not None:
        x, y, w, h = current_band.x, current_band.y, current_band.w, current_band.h
        draw.rectangle([x, y, x + w, y + h], outline=(220, 70, 70), width=2)
        _draw_text(draw, (x + 4, y + 4), "현재 보행 구간", (255, 255, 255), (180, 50, 50), 15)

    if walkways:
        colors = [(60, 200, 90), (80, 220, 170), (140, 230, 90)]
        for i, w in enumerate(walkways):
            pts = [(int(p[0]), int(p[1])) for p in w["points"]]
            if len(pts) < 2:
                continue
            col = colors[i % 3]
            for a, b in zip(pts, pts[1:]):
                draw.line([a, b], fill=col, width=4)
            flag = pts[len(pts) // 2]
            _draw_text(
                draw,
                (flag[0] + 8, flag[1] - 18),
                f"{w['name']}",
                (255, 255, 255),
                col,
                15,
            )

    if chosen is not None:
        pts = [(int(p[0]), int(p[1])) for p in chosen["points"]]
        for a, b in zip(pts, pts[1:]):
            draw.line([a, b], fill=(255, 220, 60), width=6)

    boxes = (
        interpolate_snapshot(trajectories, t)
        if trajectories is not None and not trajectories.empty
        else None
    )
    if boxes is not None:
        for _, row in boxes.iterrows():
            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            kind = row["class"]
            col = (40, 90, 220) if kind == WORKER else (255, 140, 0)
            draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
            label = "작업자" if kind == WORKER else "중장비"
            _draw_text(draw, (x1, max(0, y1 - 22)), f"{label} {row['id']}", (255, 255, 255), col, 15)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
