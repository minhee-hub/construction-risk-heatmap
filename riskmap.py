"""픽셀 좌표계 충돌 위험 히트맵."""

from __future__ import annotations

import numpy as np
import pandas as pd

from detector import EQUIPMENT, WORKER, interpolate_snapshot


def compute_pixel_heatmap(
    trajectories: pd.DataFrame,
    height: int,
    width: int,
    threshold_px: float,
    sigma_px: float,
    use_dynamic: bool = True,
    predict_steps: int = 3,
    grid: int = 180,
    at_t: int | None = None,
    lookback: int | None = None,
) -> np.ndarray:
    """작업자-중장비가 임계값 이내면 중간점에 가우시안 위험을 누적한다.

    at_t가 있으면 그 시각(과 lookback 구간)만 써서 재생 중인 프레임에
    히트맵이 따라가게 한다.
    """
    gh = max(40, int(round(grid * height / max(width, 1))))
    gw = grid
    heatmap = np.zeros((gh, gw), dtype=np.float32)
    if trajectories is None or trajectories.empty:
        return cv2_resize(heatmap, width, height)

    sx = gw / float(width)
    sy = gh / float(height)
    thr = threshold_px * (sx + sy) / 2.0
    sig = max(1.0, sigma_px * (sx + sy) / 2.0)

    ys, xs = np.ogrid[0:gh, 0:gw]
    times = sorted(int(t) for t in trajectories["t"].unique())
    if at_t is not None:
        times = [t for t in times if t <= int(at_t)]
        if lookback is not None and times:
            times = [t for t in times if t >= int(at_t) - int(lookback)]
        if not times:
            return cv2_resize(heatmap, width, height)
    prev: dict[str, tuple[float, float]] = {}

    def add(wx, wy, ex, ey, weight: float) -> None:
        dist = float(np.hypot(wx - ex, wy - ey))
        if dist >= thr:
            return
        intensity = weight * (1.0 - dist / thr)
        mx = 0.5 * (wx + ex)
        my = 0.5 * (wy + ey)
        kernel = np.exp(-((xs - mx) ** 2 + (ys - my) ** 2) / (2.0 * sig**2))
        heatmap[:] += intensity * kernel

    for frame in times:
        snap = trajectories[trajectories["t"] == frame]
        workers = snap[snap["class"] == WORKER]
        equip = snap[snap["class"] == EQUIPMENT]
        curr = {
            str(row["id"]): (float(row["x"]) * sx, float(row["y"]) * sy)
            for _, row in snap.iterrows()
        }
        for _, w in workers.iterrows():
            wx, wy = float(w["x"]) * sx, float(w["y"]) * sy
            wid = str(w["id"])
            for _, e in equip.iterrows():
                ex, ey = float(e["x"]) * sx, float(e["y"]) * sy
                eid = str(e["id"])
                add(wx, wy, ex, ey, 1.0)
                if use_dynamic and predict_steps > 0:
                    w_prev = prev.get(wid)
                    e_prev = prev.get(eid)
                    wvx = 0.0 if w_prev is None else wx - w_prev[0]
                    wvy = 0.0 if w_prev is None else wy - w_prev[1]
                    evx = 0.0 if e_prev is None else ex - e_prev[0]
                    evy = 0.0 if e_prev is None else ey - e_prev[1]
                    for k in range(1, predict_steps + 1):
                        add(
                            wx + k * wvx,
                            wy + k * wvy,
                            ex + k * evx,
                            ey + k * evy,
                            weight=0.7**k,
                        )
        prev = curr

    return cv2_resize(heatmap, width, height)


def compute_object_heatmap(
    trajectories: pd.DataFrame,
    height: int,
    width: int,
    t: float,
    sigma_worker_px: float,
    sigma_equip_px: float,
    threshold_px: float,
    predict_steps: int = 3,
    grid: int = 180,
) -> np.ndarray:
    """현재 시각의 각 객체 중심에 가우시안을 올려 객체를 따라가는 히트맵."""
    gh = max(40, int(round(grid * height / max(width, 1))))
    gw = grid
    heatmap = np.zeros((gh, gw), dtype=np.float32)
    if trajectories is None or trajectories.empty:
        return cv2_resize(heatmap, width, height)

    sx = gw / float(width)
    sy = gh / float(height)
    sig_w = max(1.2, sigma_worker_px * (sx + sy) / 2.0)
    sig_e = max(1.8, sigma_equip_px * (sx + sy) / 2.0)
    thr = threshold_px * (sx + sy) / 2.0
    ys, xs = np.ogrid[0:gh, 0:gw]

    snap = interpolate_snapshot(trajectories, t)
    prev = interpolate_snapshot(trajectories, float(t) - 2.0)
    if snap is None or snap.empty:
        return cv2_resize(heatmap, width, height)

    prev_xy: dict[str, tuple[float, float]] = {}
    if prev is not None and not prev.empty:
        for _, row in prev.iterrows():
            prev_xy[str(row["id"])] = (float(row["x"]) * sx, float(row["y"]) * sy)

    def splat(px: float, py: float, sigma: float, weight: float) -> None:
        kernel = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2.0 * sigma**2))
        heatmap[:] += weight * kernel

    def trail(oid: str, px: float, py: float, sigma: float, weight: float) -> None:
        splat(px, py, sigma, weight)
        last = prev_xy.get(oid)
        if last is None or predict_steps <= 0:
            return
        vx, vy = px - last[0], py - last[1]
        for k in range(1, predict_steps + 1):
            splat(px + k * vx, py + k * vy, sigma * (1.0 + 0.08 * k), weight * (0.55**k))

    workers = []
    equip = []
    for _, row in snap.iterrows():
        oid = str(row["id"])
        px, py = float(row["x"]) * sx, float(row["y"]) * sy
        if row["class"] == EQUIPMENT:
            trail(oid, px, py, sig_e, 1.35)
            equip.append((oid, px, py))
        elif row["class"] == WORKER:
            trail(oid, px, py, sig_w, 0.85)
            workers.append((oid, px, py))

    for _, wx, wy in workers:
        for _, ex, ey in equip:
            dist = float(np.hypot(wx - ex, wy - ey))
            if dist >= thr:
                continue
            closeness = 1.0 - dist / thr
            splat(wx, wy, sig_w * 1.15, 1.4 * closeness)
            splat(ex, ey, sig_e * 1.05, 1.1 * closeness)
            splat(0.5 * (wx + ex), 0.5 * (wy + ey), (sig_w + sig_e) * 0.45, 1.8 * closeness)

    return cv2_resize(heatmap, width, height)


def cv2_resize(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)


def risk_scores(heatmap: np.ndarray) -> tuple[float, float]:
    if heatmap.size == 0 or float(heatmap.max()) <= 0:
        return 0.0, 0.0
    norm = heatmap / float(heatmap.max())
    return float(norm.mean() * 40.0), float(np.percentile(norm, 99) * 25.0)
