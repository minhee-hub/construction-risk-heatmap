# -*- coding: utf-8 -*-
"""Detect only people and heavy equipment in a construction video."""
from __future__ import annotations

import os
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOWorld

SRC = r"C:\Users\Owner\Desktop\KakaoTalk_20261108_232135244.mp4"
OUT_DIR = r"C:\Users\Owner\construction-risk-heatmap"
OUT_VIDEO = os.path.join(OUT_DIR, "detected_사람_중장비.mp4")
PREVIEW_DIR = os.path.join(OUT_DIR, "detect_preview")
os.makedirs(PREVIEW_DIR, exist_ok=True)

PERSON_CLASSES = {"person"}
EQUIP_CLASSES = {
    "excavator",
    "forklift",
    "dump truck",
    "truck",
    "cement mixer",
    "wheel loader",
    "loader",
    "crane",
}

LABEL = {"person": "사람", "equipment": "중장비"}
COLOR = {"person": (40, 90, 220), "equipment": (0, 140, 255)}  # BGR later via RGB
CONF = 0.22
IOU = 0.5
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def nms(dets, iou_thr=0.55):
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep = []
    for d in dets:
        if all(iou(d["xyxy"], k["xyxy"]) < iou_thr or d["kind"] != k["kind"] for k in keep):
            keep.append(d)
    return keep


def draw_korean(frame_bgr, dets):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, 22)
        font_s = ImageFont.truetype(FONT_PATH, 16)
    except OSError:
        font = ImageFont.load_default()
        font_s = font
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        color = (220, 90, 40) if d["kind"] == "person" else (255, 140, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{LABEL[d['kind']]} {d['conf']:.2f}"
        tw, th = draw.textbbox((0, 0), text, font=font_s)[2:]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 4], fill=color)
        draw.text((x1 + 4, ty + 1), text, fill=(255, 255, 255), font=font_s)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    model = YOLOWorld("yolov8x-worldv2.pt")
    names = [
        "person",
        "excavator",
        "forklift",
        "dump truck",
        "truck",
        "cement mixer",
        "wheel loader",
    ]
    model.set_classes(names)

    cap = cv2.VideoCapture(SRC)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    totals = Counter()
    preview_at = {0, n // 3, n // 2, (2 * n) // 3, n - 1}

    print(f"frames={n} fps={fps:.2f} size={w}x{h}")
    for i, r in enumerate(
        model.predict(source=SRC, stream=True, conf=CONF, iou=IOU, verbose=False)
    ):
        dets = []
        if r.boxes is not None:
            for b in r.boxes:
                raw = r.names[int(b.cls)]
                if raw in PERSON_CLASSES:
                    kind = "person"
                elif raw in EQUIP_CLASSES:
                    kind = "equipment"
                else:
                    continue
                dets.append(
                    {
                        "kind": kind,
                        "conf": float(b.conf),
                        "xyxy": b.xyxy[0].tolist(),
                    }
                )
        dets = nms(dets)
        frame = r.orig_img.copy()
        annotated = draw_korean(frame, dets)
        n_p = sum(1 for d in dets if d["kind"] == "person")
        n_e = sum(1 for d in dets if d["kind"] == "equipment")
        totals["person"] += n_p
        totals["equipment"] += n_e
        cv2.putText(
            annotated,
            f"사람 {n_p}   중장비 {n_e}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(annotated)
        if i in preview_at:
            cv2.imwrite(os.path.join(PREVIEW_DIR, f"preview_{i:04d}.jpg"), annotated)
        if i % 25 == 0:
            print(f"{i}/{n} person={n_p} equipment={n_e}")

    writer.release()
    print("saved", OUT_VIDEO)
    print("sum_over_frames", dict(totals))


if __name__ == "__main__":
    main()
