"""가상 궤적 생성과 충돌 위험도 계산 (시연용 초안)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SITE_SIZE = 100
N_FRAMES = 80
WORKER_CLASS = "작업자"
EQUIPMENT_CLASS = "중장비"


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def clip_to_site(self, size: int = SITE_SIZE) -> "Rect":
        x = float(np.clip(self.x, 0, size))
        y = float(np.clip(self.y, 0, size))
        w = float(np.clip(self.w, 1, size - x))
        h = float(np.clip(self.h, 1, size - y))
        return Rect(x, y, w, h)


def generate_trajectories(seed: int = 42) -> pd.DataFrame:
    """작업자 3명 + 중장비 1대의 가상 이동 궤적을 생성한다.

    중장비는 좌하단에서 우상단으로 대각선 이동하고,
    작업자 경로 일부가 그 대각선과 교차하도록 구성한다.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_FRAMES, dtype=float)
    rows: list[dict] = []

    # 중장비: (12, 12) -> (88, 88)
    eq_x = np.linspace(12, 88, N_FRAMES) + rng.normal(0, 0.25, N_FRAMES)
    eq_y = np.linspace(12, 88, N_FRAMES) + rng.normal(0, 0.25, N_FRAMES)
    for i in range(N_FRAMES):
        rows.append(
            {
                "id": "E1",
                "class": EQUIPMENT_CLASS,
                "t": int(t[i]),
                "x": float(np.clip(eq_x[i], 0, SITE_SIZE)),
                "y": float(np.clip(eq_y[i], 0, SITE_SIZE)),
            }
        )

    # 작업자 W1: 반대 대각선 (12, 88) -> (88, 12) — 중앙에서 교차
    w1_x = np.linspace(12, 88, N_FRAMES) + rng.normal(0, 0.35, N_FRAMES)
    w1_y = np.linspace(88, 12, N_FRAMES) + rng.normal(0, 0.35, N_FRAMES)

    # 작업자 W2: y≈40 수평 이동 — 대각선과 한 번 교차
    w2_x = np.linspace(8, 92, N_FRAMES) + rng.normal(0, 0.4, N_FRAMES)
    w2_y = 40 + 4 * np.sin(t / 7.0) + rng.normal(0, 0.3, N_FRAMES)

    # 작업자 W3: 우상단 부근을 왕복 — 후반부에 중장비와 근접
    w3_x = 72 + 14 * np.sin(t / 9.0) + rng.normal(0, 0.3, N_FRAMES)
    w3_y = 68 + 12 * np.cos(t / 11.0) + rng.normal(0, 0.3, N_FRAMES)

    workers = {
        "W1": (w1_x, w1_y),
        "W2": (w2_x, w2_y),
        "W3": (w3_x, w3_y),
    }
    for worker_id, (xs, ys) in workers.items():
        for i in range(N_FRAMES):
            rows.append(
                {
                    "id": worker_id,
                    "class": WORKER_CLASS,
                    "t": int(t[i]),
                    "x": float(np.clip(xs[i], 0, SITE_SIZE)),
                    "y": float(np.clip(ys[i], 0, SITE_SIZE)),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["t", "class", "id"]).reset_index(drop=True)


def _positions_at(df: pd.DataFrame, frame: int) -> dict[str, tuple[float, float, str]]:
    snap = df[df["t"] == frame]
    return {
        str(row["id"]): (float(row["x"]), float(row["y"]), str(row["class"]))
        for _, row in snap.iterrows()
    }


def _velocity(prev: tuple[float, float] | None, curr: tuple[float, float]) -> tuple[float, float]:
    if prev is None:
        return (0.0, 0.0)
    return (curr[0] - prev[0], curr[1] - prev[1])


def compute_risk_heatmap(
    trajectories: pd.DataFrame,
    threshold: float = 10.0,
    sigma: float = 5.0,
    use_dynamic: bool = False,
    predict_steps: int = 3,
    grid_size: int = SITE_SIZE,
) -> np.ndarray:
    """같은 시각의 작업자-중장비 거리가 임계값 이내면 중간점에 가우시안 위험을 누적한다.

    use_dynamic이 True이면 속도 벡터로 predict_steps만큼 앞선 예상 위치도 반영한다.
    예상 위치의 기여는 단계가 멀수록 감쇠한다.
    """
    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    ys, xs = np.ogrid[0:grid_size, 0:grid_size]

    times = sorted(trajectories["t"].unique())
    prev_pos: dict[str, tuple[float, float]] = {}

    def add_pair_risk(
        wx: float,
        wy: float,
        ex: float,
        ey: float,
        weight: float,
    ) -> None:
        dist = float(np.hypot(wx - ex, wy - ey))
        if dist >= threshold:
            return
        intensity = weight * (1.0 - dist / threshold)
        mx = 0.5 * (wx + ex)
        my = 0.5 * (wy + ey)
        kernel = np.exp(-((xs - mx) ** 2 + (ys - my) ** 2) / (2.0 * sigma**2))
        heatmap[:] += intensity * kernel

    for frame in times:
        curr = _positions_at(trajectories, frame)
        equipment = [
            (oid, xy[0], xy[1])
            for oid, xy in curr.items()
            if xy[2] == EQUIPMENT_CLASS
        ]
        workers = [
            (oid, xy[0], xy[1])
            for oid, xy in curr.items()
            if xy[2] == WORKER_CLASS
        ]

        for wid, wx, wy in workers:
            for eid, ex, ey in equipment:
                add_pair_risk(wx, wy, ex, ey, weight=1.0)

                if use_dynamic and predict_steps > 0:
                    w_prev = prev_pos.get(wid)
                    e_prev = prev_pos.get(eid)
                    wvx, wvy = _velocity(w_prev, (wx, wy))
                    evx, evy = _velocity(e_prev, (ex, ey))
                    for k in range(1, predict_steps + 1):
                        decay = 0.7**k
                        add_pair_risk(
                            wx + k * wvx,
                            wy + k * wvy,
                            ex + k * evx,
                            ey + k * evy,
                            weight=decay,
                        )

        prev_pos = {oid: (xy[0], xy[1]) for oid, xy in curr.items()}

    return heatmap


def rect_risk_sum(heatmap: np.ndarray, rect: Rect) -> float:
    """사각형이 덮는 격자 칸의 위험도 합계."""
    hh, ww = heatmap.shape[:2]
    x = float(np.clip(rect.x, 0, ww))
    y = float(np.clip(rect.y, 0, hh))
    w = float(np.clip(rect.w, 1, max(1, ww - x)))
    h = float(np.clip(rect.h, 1, max(1, hh - y)))
    r = Rect(x, y, w, h)
    x0 = int(np.floor(r.x))
    y0 = int(np.floor(r.y))
    x1 = int(np.ceil(r.x + r.w))
    y1 = int(np.ceil(r.y + r.h))
    x1 = min(x1, ww)
    y1 = min(y1, hh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    # 히트맵 인덱스는 [y, x]
    return float(heatmap[y0:y1, x0:x1].sum())


def move_distance(a: Rect, b: Rect) -> float:
    ax, ay = a.center
    bx, by = b.center
    return float(np.hypot(ax - bx, ay - by))
