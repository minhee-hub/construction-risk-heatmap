"""사람·중장비만 추적해 궤적을 만든다."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Optional

import cv2
import numpy as np
import pandas as pd

WORKER = "작업자"
EQUIPMENT = "중장비"
PERSON_NAMES = {"person"}
EQUIP_NAMES = {
    "excavator",
    "forklift",
    "dump truck",
    "truck",
    "cement mixer",
    "wheel loader",
    "loader",
    "car",
    "bus",
}


def _kind(name: str) -> Optional[str]:
    n = name.lower()
    if n in PERSON_NAMES:
        return WORKER
    if n in EQUIP_NAMES:
        return EQUIPMENT
    return None


def load_model(prefer_world: bool = False):
    weights_dir = os.path.dirname(os.path.abspath(__file__))
    world = os.path.join(weights_dir, "yolov8x-worldv2.pt")
    nano = os.path.join(weights_dir, "yolo26n.pt")
    if prefer_world and os.path.exists(world):
        from ultralytics import YOLOWorld

        model = YOLOWorld(world)
        model.set_classes(
            ["person", "excavator", "forklift", "dump truck", "truck", "cement mixer", "wheel loader"]
        )
        return model, "yolov8x-worldv2"
    from ultralytics import YOLO

    path = nano if os.path.exists(nano) else "yolo26n.pt"
    return YOLO(path), "yolo26n"


def analyze_video(
    source: str,
    stride: int = 2,
    conf: float = 0.25,
    prefer_world: bool = False,
    progress_cb=None,
) -> dict:
    model, model_name = load_model(prefer_world=prefer_world)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    classes = None
    if model_name == "yolo26n":
        classes = [0, 2, 5, 7]  # person, car, bus, truck

    rows = []
    key_frame = None
    key_t = 0
    frame_i = 0
    cache_dir = tempfile.mkdtemp(prefix="siteopt_frames_")
    frame_files: dict[int, str] = {}
    results = model.track(
        source=source,
        stream=True,
        persist=True,
        conf=conf,
        vid_stride=max(1, stride),
        classes=classes,
        verbose=False,
        tracker="bytetrack.yaml",
    )
    for r in results:
        if r.orig_img is None:
            frame_i += stride
            continue
        jpg = os.path.join(cache_dir, f"{int(frame_i):06d}.jpg")
        cv2.imwrite(jpg, r.orig_img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        frame_files[int(frame_i)] = jpg
        if key_frame is None or frame_i >= n_frames * 0.45:
            key_frame = r.orig_img.copy()
            key_t = max(0, frame_i)
        if r.boxes is not None:
            ids = r.boxes.id
            for j, b in enumerate(r.boxes):
                name = r.names[int(b.cls)]
                kind = _kind(name)
                if kind is None:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                tid = int(ids[j]) if ids is not None else j
                rows.append(
                    {
                        "id": f"{kind[0]}{tid}",
                        "class": kind,
                        "raw": name,
                        "t": int(frame_i),
                        "x": float((x1 + x2) / 2.0),
                        "y": float((y1 + y2) / 2.0),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "conf": float(b.conf),
                    }
                )
        if progress_cb and n_frames:
            progress_cb(min(1.0, frame_i / max(1, n_frames)))
        frame_i += stride

    traj = pd.DataFrame(rows)
    if traj.empty:
        n_workers = n_equip = 0
    else:
        n_workers = int(traj.loc[traj["class"] == WORKER, "id"].nunique())
        n_equip = int(traj.loc[traj["class"] == EQUIPMENT, "id"].nunique())

    return {
        "trajectories": traj,
        "frame": key_frame,
        "frame_files": frame_files,
        "cache_dir": cache_dir,
        "video_path": source,
        "t": key_t,
        "fps": float(fps),
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "n_workers": n_workers,
        "n_equip": n_equip,
        "model": model_name,
    }


def load_cached_frame(analysis: dict, t: int) -> Optional[np.ndarray]:
    files = analysis.get("frame_files") or {}
    if files:
        keys = np.array(sorted(int(k) for k in files.keys()), dtype=int)
        nearest = int(keys[np.abs(keys - int(t)).argmin()])
        img = cv2.imread(files[nearest])
        if img is not None:
            return img
    frame = analysis.get("frame")
    return None if frame is None else frame.copy()


def snapshot_boxes(traj: pd.DataFrame, t: int) -> pd.DataFrame:
    if traj.empty:
        return traj
    times = np.sort(traj["t"].unique())
    if t not in times:
        t = int(times[np.abs(times - t).argmin()])
    return traj[traj["t"] == t].copy()


def interpolate_snapshot(traj: pd.DataFrame, t: float) -> pd.DataFrame:
    """검출 프레임 사이에서는 객체 좌표를 보간해 히트맵이 따라가게 한다."""
    if traj is None or traj.empty:
        return traj
    times = np.sort(traj["t"].unique())
    t = float(t)
    if t <= float(times[0]):
        return traj[traj["t"] == times[0]].copy()
    if t >= float(times[-1]):
        return traj[traj["t"] == times[-1]].copy()
    t0 = float(times[times <= t].max())
    t1 = float(times[times >= t].min())
    a = traj[traj["t"] == t0]
    if abs(t1 - t0) < 1e-6:
        return a.copy()
    b = traj[traj["t"] == t1]
    u = (t - t0) / (t1 - t0)
    cols = [c for c in ("x", "y", "x1", "y1", "x2", "y2") if c in traj.columns]
    b_by_id = {str(r["id"]): r for _, r in b.iterrows()}
    rows = []
    seen = set()
    for _, ra in a.iterrows():
        oid = str(ra["id"])
        seen.add(oid)
        row = ra.copy()
        rb = b_by_id.get(oid)
        if rb is not None:
            for c in cols:
                row[c] = (1.0 - u) * float(ra[c]) + u * float(rb[c])
        row["t"] = t
        rows.append(row)
    for _, rb in b.iterrows():
        oid = str(rb["id"])
        if oid in seen:
            continue
        row = rb.copy()
        row["t"] = t
        rows.append(row)
    return pd.DataFrame(rows)
