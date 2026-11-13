"""임시 보행로 후보: 고위험 격자를 피해 낮은 위험 경로를 찾는다."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from simulator import Rect, rect_risk_sum


def downsample_heatmap(heatmap: np.ndarray, max_side: int = 80) -> tuple[np.ndarray, float]:
    h, w = heatmap.shape
    scale = max(h, w) / float(max_side)
    if scale <= 1:
        return heatmap.copy(), 1.0
    nh, nw = max(8, int(round(h / scale))), max(8, int(round(w / scale)))
    ys = np.linspace(0, h - 1, nh).astype(int)
    xs = np.linspace(0, w - 1, nw).astype(int)
    small = heatmap[ys][:, xs]
    return small, h / nh


def _neighbors(y: int, x: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def _dijkstra(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = cost.shape
    inf = 1e18
    dist = np.full((h, w), inf)
    prev = np.full((h, w, 2), -1, dtype=int)
    sy, sx = start
    gy, gx = goal
    dist[sy, sx] = 0.0
    q: deque[tuple[int, int]] = deque([(sy, sx)])
    inq = np.zeros((h, w), dtype=bool)
    inq[sy, sx] = True
    while q:
        y, x = q.popleft()
        inq[y, x] = False
        for ny, nx in _neighbors(y, x, h, w):
            step = 1.414 if (ny != y and nx != x) else 1.0
            nd = dist[y, x] + step * (1.0 + float(cost[ny, nx]))
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx] = (y, x)
                if not inq[ny, nx]:
                    q.append((ny, nx))
                    inq[ny, nx] = True
    if dist[gy, gx] >= inf / 2:
        return []
    path = []
    y, x = gy, gx
    while y >= 0:
        path.append((y, x))
        y, x = int(prev[y, x, 0]), int(prev[y, x, 1])
    path.reverse()
    return path


def recommend_walkways(
    heatmap: np.ndarray,
    n_candidates: int = 3,
) -> list[dict]:
    """고위험 셀을 우회하는 임시 보행로 후보를 최대 3개 반환한다."""
    small, scale = downsample_heatmap(heatmap)
    h, w = small.shape
    peak = float(small.max()) if small.size else 0.0
    if peak <= 1e-9:
        peak = 1.0
    cost = small / peak
    # 상위 위험은 더 비싸게
    cost = np.clip(cost * 4.0, 0.05, 8.0)

    starts = [
        (h - 2, 2),
        (h // 2, 2),
        (h - 2, w // 3),
    ]
    goals = [
        (2, w - 3),
        (h // 3, w - 3),
        (2, w // 2),
    ]

    results = []
    seen = []
    for i, (start, goal) in enumerate(zip(starts, goals)):
        path = _dijkstra(cost, start, goal)
        if len(path) < 4:
            continue
        pts = [(int(x * scale + scale / 2), int(y * scale + scale / 2)) for y, x in path]
        # 너무 비슷한 경로는 건너뜀
        mid = pts[len(pts) // 2]
        if any(abs(mid[0] - s[0]) + abs(mid[1] - s[1]) < 40 for s in seen):
            continue
        seen.append(mid)
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        x0, x1 = max(0, min(xs) - 4), min(heatmap.shape[1], max(xs) + 4)
        y0, y1 = max(0, min(ys) - 4), min(heatmap.shape[0], max(ys) + 4)
        band = Rect(float(x0), float(y0), float(max(1, x1 - x0)), float(max(1, y1 - y0)))
        samples = []
        hh, ww = heatmap.shape[:2]
        for x, y in pts:
            xi, yi = int(np.clip(x, 0, ww - 1)), int(np.clip(y, 0, hh - 1))
            samples.append(float(heatmap[yi, xi]))
        risk = float(np.mean(samples)) if samples else 0.0
        length = 0.0
        for a, b in zip(pts, pts[1:]):
            length += float(np.hypot(a[0] - b[0], a[1] - b[1]))
        results.append(
            {
                "name": f"임시보행로 {i + 1}",
                "points": pts[:: max(1, len(pts) // 40)],
                "risk": float(risk),
                "length": float(length),
                "band": band,
            }
        )
        if len(results) >= n_candidates:
            break

    results.sort(key=lambda r: (r["risk"], r["length"]))
    for i, item in enumerate(results, start=1):
        item["rank"] = i
        item["name"] = f"임시보행로 {i}"
    return results


def mean_risk_along(heatmap: np.ndarray, points) -> float:
    if heatmap is None or heatmap.size == 0 or points is None:
        return 0.0
    hh, ww = heatmap.shape[:2]
    vals = []
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < ww and 0 <= y < hh:
            vals.append(float(heatmap[y, x]))
    return float(np.mean(vals)) if vals else 0.0


def current_walkway_band(worker_xy: np.ndarray, heatmap: np.ndarray, pad: int = 18) -> Rect:
    if worker_xy.size == 0:
        h, w = heatmap.shape
        return Rect(w * 0.35, h * 0.45, w * 0.2, h * 0.2)
    xs, ys = worker_xy[:, 0], worker_xy[:, 1]
    x0 = max(0, float(xs.min()) - pad)
    y0 = max(0, float(ys.min()) - pad)
    x1 = min(heatmap.shape[1], float(xs.max()) + pad)
    y1 = min(heatmap.shape[0], float(ys.max()) + pad)
    return Rect(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))
