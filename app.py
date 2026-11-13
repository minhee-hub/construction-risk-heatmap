"""이동 궤적 및 동적 충돌 위험도 히트맵 분석기 기반 건설현장 임시 보행로 지정 및 개선."""

from __future__ import annotations

import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detector import EQUIPMENT, WORKER, analyze_video, interpolate_snapshot, load_cached_frame
from riskmap import compute_object_heatmap, compute_pixel_heatmap, risk_scores
from simulator import generate_trajectories
from thresholds import (
    DYNAMIC_THRESHOLD_M,
    PREDICT_STEPS,
    SIGMA_M,
    SITE_SPEED_LIMIT_KMH,
    STATIC_THRESHOLD_M,
    THRESHOLD_NOTE,
    meters_to_pixels,
)
from viz import render_scene
from walkways import current_walkway_band, mean_risk_along, recommend_walkways

APP_TITLE = "이동 궤적 및 동적 충돌 위험도 히트맵 분석기 기반 건설현장 임시 보행로 지정 및 개선"
SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample.mp4")

st.set_page_config(page_title=APP_TITLE, page_icon="🚧", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: "Noto Sans KR", sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 20% -10%, #243049 0%, #141820 42%, #0f1218 100%); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.1rem; padding-bottom: 0.4rem; max-width: 1500px; }
h1, h2, h3, p, label, span { color: #e8edf7 !important; }
.hero-title { font-size: 1.45rem; font-weight: 700; color: #f4f7ff; letter-spacing: -0.3px; margin: 0; }
.hero-sub { color: #9aa6bf !important; font-size: 0.88rem; margin: 0.15rem 0 0.85rem 0; }
.card {
  background: linear-gradient(180deg, #232a38 0%, #1b2130 100%);
  border: 1px solid #343c50;
  border-radius: 14px;
  padding: 12px 14px 10px 14px;
  margin-bottom: 10px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.card h4 { margin: 0 0 8px 0; font-size: 0.86rem; color: #d5def0 !important; letter-spacing: 0.2px; }
.kv { color: #9aa6bf; font-size: 0.8rem; }
.kv b { color: #eef3ff; font-weight: 600; }
.metric-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 0.92rem; }
.metric-row span { color: #a9b4c9; }
.metric-row b { color: #ffd27a; }
.view-wrap {
  background: #151a24;
  border: 1px solid #3a465c;
  border-radius: 16px;
  padding: 8px;
  box-shadow: inset 0 0 0 1px rgba(80,140,255,0.08);
}
.status {
  margin-top: 8px;
  color: #b7c2d6;
  font-size: 0.82rem;
  background: #1a2130;
  border-radius: 8px;
  padding: 7px 10px;
  border: 1px solid #2e3648;
}
div.stButton > button {
  width: 100%;
  background: #2a3142;
  color: #eaf0ff;
  border: 1px solid #44506a;
  border-radius: 10px;
  padding: 0.45rem 0.6rem;
  font-weight: 600;
}
div.stButton > button:hover { border-color: #f59a3d; color: #fff; }
div[data-testid="stFileUploader"] { background: transparent; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stImage"] { width: 100% !important; }
[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

defaults = {
    "analysis": None,
    "heatmap": None,
    "walkways": [],
    "show_heat": True,
    "show_traj": True,
    "chosen": None,
    "playing": False,
    "play_idx": 0,
    "video_path": SAMPLE if os.path.exists(SAMPLE) else None,
    "upload_name": None,
    "n_video_frames": 0,
    "video_fps": 10.0,
    "preview_files": {},
    "preview_path": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def synthetic_on_frame(frame: np.ndarray) -> pd.DataFrame:
    """영상이 없거나 인식이 비면 가상 궤적을 프레임 크기에 맞춰 쓴다."""
    h, w = frame.shape[:2]
    raw = generate_trajectories()
    df = raw.copy()
    df["x"] = df["x"] / 100.0 * w
    df["y"] = (1.0 - df["y"] / 100.0) * h
    df["x1"] = df["x"] - 28
    df["y1"] = df["y"] - 40
    df["x2"] = df["x"] + 28
    df["y2"] = df["y"] + 40
    df["conf"] = 0.7
    df["raw"] = df["class"]
    df["t"] = (df["t"] * 3).astype(int)
    return df


def blank_frame() -> np.ndarray:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (42, 48, 58)
    cv2.putText(img, "CCTV View", (48, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (180, 190, 210), 2)
    return img


def probe_video(path: str) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 10.0, 1280, 720
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()
    return n, max(fps, 1.0), w, h


def read_video_frame(path: str, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def ensure_video_meta() -> None:
    path = st.session_state.video_path
    if st.session_state.n_video_frames > 0 or not path or not os.path.exists(path):
        return
    n, fps, _, _ = probe_video(path)
    st.session_state.n_video_frames = n
    st.session_state.video_fps = fps


def load_preview(path: str) -> None:
    """Decode the video once into JPEG files so Play does not reopen the file every frame."""
    if not path or not os.path.exists(path):
        return
    if st.session_state.preview_path == path and st.session_state.preview_files:
        return
    n, fps, _, _ = probe_video(path)
    step = max(1, int(round(max(fps, 1.0) / 10.0)))
    cache_dir = tempfile.mkdtemp(prefix="siteopt_preview_")
    files: dict[int, str] = {}
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            jpg = os.path.join(cache_dir, f"{i:06d}.jpg")
            cv2.imwrite(jpg, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            files[i] = jpg
        i += 1
    cap.release()
    st.session_state.preview_files = files
    st.session_state.preview_path = path
    st.session_state.n_video_frames = n or i
    st.session_state.video_fps = fps


def play_times() -> list[int]:
    files = st.session_state.get("preview_files") or {}
    if files:
        return sorted(int(k) for k in files.keys())
    a = st.session_state.analysis
    cached = (a or {}).get("frame_files") or {}
    if cached:
        return sorted(int(k) for k in cached.keys())
    ensure_video_meta()
    n = int(st.session_state.n_video_frames or 0)
    if a is not None:
        n = max(n, int(a.get("n_frames") or 0))
    fps = float(st.session_state.video_fps or 10.0)
    step = max(1, int(round(max(fps, 1.0) / 10.0)))
    if n > 0:
        return list(range(0, n, step))
    if a is not None and a.get("trajectories") is not None and not a["trajectories"].empty:
        return [int(t) for t in sorted(a["trajectories"]["t"].unique())]
    return [0]


def current_t() -> int:
    times = play_times()
    idx = int(np.clip(st.session_state.play_idx, 0, len(times) - 1))
    st.session_state.play_idx = idx
    return times[idx]


def live_heatmap_for(t: int) -> np.ndarray | None:
    a = st.session_state.analysis
    if a is None or a.get("trajectories") is None or a["trajectories"].empty:
        return None
    if not st.session_state.show_heat:
        return None
    w, h = int(a["width"]), int(a["height"])
    return compute_object_heatmap(
        a["trajectories"],
        h,
        w,
        t=float(t),
        sigma_worker_px=meters_to_pixels(1.4, w),
        sigma_equip_px=meters_to_pixels(2.8, w),
        threshold_px=meters_to_pixels(DYNAMIC_THRESHOLD_M, w),
        predict_steps=PREDICT_STEPS,
    )


def frame_at(t: int) -> np.ndarray:
    preview = st.session_state.get("preview_files") or {}
    if preview:
        keys = np.array(sorted(int(k) for k in preview.keys()), dtype=int)
        nearest = int(keys[np.abs(keys - int(t)).argmin()])
        img = cv2.imread(preview[nearest])
        if img is not None:
            return img
    a = st.session_state.analysis
    files = (a or {}).get("frame_files") or {}
    if files and int(t) in files:
        img = cv2.imread(files[int(t)])
        if img is not None:
            return img
    path = (a or {}).get("video_path") or st.session_state.video_path
    if path and os.path.exists(str(path)):
        raw = read_video_frame(str(path), t)
        if raw is not None:
            return raw
    if a is not None:
        cached = load_cached_frame(a, t)
        if cached is not None:
            return cached
    return blank_frame()


def heat_for_frame(t: int) -> np.ndarray | None:
    """CCTV 재생용: 현재 프레임 객체 위치를 따라가는 히트맵만 올린다."""
    return live_heatmap_for(t)


def render_cctv(t: int) -> tuple[np.ndarray, np.ndarray | None]:
    a = st.session_state.analysis
    frame = frame_at(t)
    heat = heat_for_frame(t)
    if a is None:
        return frame, heat
    vis = render_scene(
        frame,
        a["trajectories"],
        heat,
        t=t,
        show_heatmap=heat is not None,
        show_traj=st.session_state.show_traj,
        walkways=st.session_state.walkways if st.session_state.heatmap is not None else None,
        current_band=st.session_state.get("current_band") if st.session_state.heatmap is not None else None,
        chosen=st.session_state.chosen,
    )
    return vis, heat


def start_play() -> None:
    path = st.session_state.video_path
    if path and os.path.exists(str(path)):
        load_preview(str(path))
    if st.session_state.analysis is not None and st.session_state.heatmap is None:
        build_heatmap()
    times = play_times()
    if st.session_state.play_idx >= len(times) - 1:
        st.session_state.play_idx = 0
    st.session_state.playing = True


def cctv_player():
    t_now = current_t()
    vis, heat = render_cctv(t_now)
    fps = float(st.session_state.video_fps or 10.0)
    clock = t_now / fps if fps else 0.0
    times = play_times()
    heat_note = "  ·  히트맵 ON" if heat is not None else ""
    st.image(
        cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
        width="stretch",
        caption=f"CCTV View  ·  t={clock:.1f}s  ·  frame {t_now} / {times[-1] if times else 0}{heat_note}",
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("▶  동영상 재생", key="play_view"):
            start_play()
            st.rerun()
    with b2:
        if st.button("⏸  Pause", key="pause_view"):
            st.session_state.playing = False
            st.rerun()
    with b3:
        if st.button("⏹  Stop", key="stop_view"):
            st.session_state.playing = False
            st.session_state.play_idx = 0
            st.rerun()
    status = "Playing" if st.session_state.playing else ("Active" if st.session_state.analysis else "Idle")
    rec_n = len(st.session_state.walkways)
    heat_state = "히트맵 표시 중" if heat is not None else "히트맵은 Analyze Video 후 재생에 겹칩니다"
    st.markdown(
        f"<div class='status'>Analysis Status: <b>{status}</b> &nbsp;|&nbsp; "
        f"{heat_state} &nbsp;|&nbsp; "
        f"모델 임계값 {DYNAMIC_THRESHOLD_M} m (동적) · {STATIC_THRESHOLD_M} m (정적) "
        f"&nbsp;|&nbsp; Recs: {rec_n} available</div>",
        unsafe_allow_html=True,
    )


def run_analysis(video_path: str, prefer_world: bool, progress):
    result = analyze_video(video_path, stride=3, prefer_world=prefer_world, progress_cb=progress.progress)
    frame = result["frame"]
    if frame is None:
        frame = blank_frame()
        result["frame"] = frame
        result["width"], result["height"] = 1280, 720
    if result["trajectories"].empty:
        result["trajectories"] = synthetic_on_frame(frame)
        result["n_workers"] = int(result["trajectories"].loc[result["trajectories"]["class"] == WORKER, "id"].nunique())
        result["n_equip"] = int(result["trajectories"].loc[result["trajectories"]["class"] == EQUIPMENT, "id"].nunique())
        result["model"] = result.get("model", "synthetic")
    result["video_path"] = video_path
    st.session_state.analysis = result
    st.session_state.heatmap = None
    st.session_state.walkways = []
    st.session_state.chosen = None
    st.session_state.play_idx = 0
    st.session_state.playing = False
    st.session_state.n_video_frames = int(result.get("n_frames") or 0)
    st.session_state.video_fps = float(result.get("fps") or 10.0)


def build_heatmap():
    a = st.session_state.analysis
    if a is None:
        return
    w, h = int(a["width"]), int(a["height"])
    thr = meters_to_pixels(DYNAMIC_THRESHOLD_M, w)
    sig = meters_to_pixels(SIGMA_M, w)
    heat = compute_pixel_heatmap(
        a["trajectories"],
        h,
        w,
        threshold_px=thr,
        sigma_px=sig,
        use_dynamic=True,
        predict_steps=PREDICT_STEPS,
    )
    st.session_state.heatmap = heat
    workers = a["trajectories"]
    worker_xy = workers.loc[workers["class"] == WORKER, ["x", "y"]].to_numpy() if not workers.empty else np.zeros((0, 2))
    current = current_walkway_band(worker_xy, heat)
    recs = recommend_walkways(heat, n_candidates=3)
    st.session_state.current_band = current
    st.session_state.walkways = recs
    if recs:
        st.session_state.chosen = recs[0]


st.markdown(f"<p class='hero-title'>SiteOpt AI Safety : {APP_TITLE}</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-sub'>충돌 위험도 히트맵 및 임시 보행로 지정 · 개선 시스템 "
    "(Collision Risk Heatmap &amp; Temporary Walkway Recommendation)</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([0.31, 0.69], gap="large")

with left:
    st.markdown(
        "<div class='card'><div class='kv'>Data Source · Cam 04 샘플 CCTV</div></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("다른 영상으로 바꾸기", type=["mp4", "avi", "mov"])
    if uploaded is not None and st.session_state.upload_name != uploaded.name:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1] or ".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state.video_path = tmp.name
        st.session_state.upload_name = uploaded.name
        n, fps, _, _ = probe_video(tmp.name)
        st.session_state.n_video_frames = n
        st.session_state.video_fps = fps
        st.session_state.play_idx = 0
        st.session_state.playing = False
    elif uploaded is None and (not st.session_state.video_path or not os.path.exists(str(st.session_state.video_path))):
        if os.path.exists(SAMPLE):
            st.session_state.video_path = SAMPLE
            n, fps, _, _ = probe_video(SAMPLE)
            st.session_state.n_video_frames = n
            st.session_state.video_fps = fps

    precise = st.checkbox("정밀 중장비 인식 (YOLO-World, 더 느림)", value=False)
    if st.button("▶  Analyze Video", use_container_width=True):
        path = st.session_state.video_path or SAMPLE
        if not os.path.exists(path):
            st.warning("샘플 영상이 없어 가상 궤적으로 시연합니다.")
        bar = st.progress(0.0, text="객체 인식 중…")
        try:
            if os.path.exists(path):
                run_analysis(path, prefer_world=precise, progress=bar)
            else:
                frame = blank_frame()
                st.session_state.analysis = {
                    "trajectories": synthetic_on_frame(frame),
                    "frame": frame,
                    "frame_files": {},
                    "t": 40,
                    "fps": 10,
                    "width": 1280,
                    "height": 720,
                    "n_frames": 80,
                    "n_workers": 3,
                    "n_equip": 1,
                    "model": "synthetic",
                }
                st.session_state.play_idx = 0
            st.success("영상 분석을 마쳤습니다. Play로 히트맵과 함께 재생하세요.")
        except Exception as exc:
            st.error(f"분석 실패: {exc}")
        finally:
            bar.empty()

    if st.button("▣  Generate Heatmap", use_container_width=True):
        if st.session_state.analysis is None:
            st.warning("먼저 Analyze Video를 실행하세요.")
        else:
            with st.spinner("위험도 히트맵과 보행로 후보를 계산하는 중…"):
                build_heatmap()
            st.success("전체 구간 히트맵과 임시 보행로 후보를 만들었습니다. 동영상 재생 시 화면 위에 겹쳐 보입니다.")

    if st.button("▶  동영상 재생", key="play_side", width="stretch"):
        with st.spinner("영상과 히트맵을 준비하고 재생합니다…"):
            start_play()
    pc2, pc3 = st.columns(2)
    with pc2:
        if st.button("⏸  일시정지", key="pause_side"):
            st.session_state.playing = False
    with pc3:
        if st.button("⏹  정지", key="stop_side"):
            st.session_state.playing = False
            st.session_state.play_idx = 0

    times = play_times()
    max_idx = max(0, len(times) - 1)
    if st.session_state.playing:
        frac = 0.0 if max_idx <= 0 else st.session_state.play_idx / max_idx
        st.progress(frac, text=f"재생 중  {st.session_state.play_idx + 1} / {max_idx + 1}")
    else:
        slider_idx = st.slider(
            "재생 위치",
            min_value=0,
            max_value=max_idx,
            value=min(int(st.session_state.play_idx), max_idx),
        )
        st.session_state.play_idx = int(slider_idx)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Toggle Heatmap"):
            st.session_state.show_heat = not st.session_state.show_heat
    with c2:
        if st.button("Trajectory Lines"):
            st.session_state.show_traj = not st.session_state.show_traj

    st.markdown("<div class='card'><h4>Layout Optimization · 임시 보행로</h4></div>", unsafe_allow_html=True)
    if st.button("Identify Current Walkways", use_container_width=True):
        if st.session_state.heatmap is None:
            st.warning("Generate Heatmap을 먼저 실행하세요.")
        else:
            st.info("현재 작업자 궤적 띠를 현재 보행 구간으로 표시합니다.")
    if st.button("Recommend New Walkways", use_container_width=True):
        if st.session_state.heatmap is None:
            st.warning("Generate Heatmap을 먼저 실행하세요.")
        else:
            recs = st.session_state.walkways
            if recs:
                st.session_state.chosen = recs[0]
                st.success(f"추천: {recs[0]['name']} (위험 합계 {recs[0]['risk']:.1f})")
            else:
                st.warning("후보 경로를 찾지 못했습니다.")

    names = [w["name"] for w in st.session_state.walkways] or ["후보 없음"]
    pick = st.selectbox("View Walkway Proposals", names)
    if st.session_state.walkways:
        st.session_state.chosen = next(w for w in st.session_state.walkways if w["name"] == pick)

    t_now = current_t()
    a = st.session_state.analysis
    live_heat = live_heatmap_for(t_now) if a is not None else None
    if a is not None and a.get("trajectories") is not None and not a["trajectories"].empty:
        snap = interpolate_snapshot(a["trajectories"], t_now)
        n_m = int(snap.loc[snap["class"] == EQUIPMENT, "id"].nunique()) if not snap.empty else 0
        n_w = int(snap.loc[snap["class"] == WORKER, "id"].nunique()) if not snap.empty else 0
    else:
        n_m = a["n_equip"] if a else 0
        n_w = a["n_workers"] if a else 0
    avg, peak = risk_scores(live_heat) if live_heat is not None else (0.0, 0.0)
    st.markdown(
        f"<div class='card'><h4>Object Class Detection</h4>"
        f"<div class='metric-row'><span>Machinery</span><b>{n_m}</b></div>"
        f"<div class='metric-row'><span>Workers</span><b>{n_w}</b></div>"
        f"<div class='metric-row'><span>Walkway recs</span><b>{len(st.session_state.walkways)}</b></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='card'><h4>Risk Score Summary</h4>"
        f"<div class='metric-row'><span>Current Average</span><b>{avg:.2f}</b></div>"
        f"<div class='metric-row'><span>Peak</span><b>{peak:.2f}</b></div></div>",
        unsafe_allow_html=True,
    )

    current_sum = 0.0
    best_sum = 0.0
    heat = st.session_state.heatmap
    if heat is not None and a is not None:
        workers = a["trajectories"]
        if workers is not None and not workers.empty:
            wxy = workers.loc[workers["class"] == WORKER, ["x", "y"]].to_numpy()
            current_sum = mean_risk_along(heat, wxy)
        if st.session_state.chosen is not None:
            best_sum = mean_risk_along(heat, st.session_state.chosen["points"])
    st.markdown(
        f"<div class='card'><h4>Walkway Risk Compare</h4>"
        f"<div class='metric-row'><span>현재 보행 구간</span><b>{current_sum:.1f}</b></div>"
        f"<div class='metric-row'><span>추천 임시보행로</span><b>{best_sum:.1f}</b></div></div>",
        unsafe_allow_html=True,
    )

    with st.expander("임계값 근거 (KOSHA · 연구용 참고치)", expanded=False):
        st.write(THRESHOLD_NOTE)
        st.caption(
            f"정적 {STATIC_THRESHOLD_M} m · 동적 {DYNAMIC_THRESHOLD_M} m · "
            f"제한속도 {SITE_SPEED_LIMIT_KMH:.0f} km/h · 시그마 {SIGMA_M} m"
        )

with right:
    cctv_player()

    if st.session_state.walkways:
        table = pd.DataFrame(
            [
                {
                    "순위": w["rank"],
                    "후보": w["name"],
                    "경로 위험 합계": round(w["risk"], 1),
                    "경로 길이(px)": round(w["length"], 0),
                }
                for w in st.session_state.walkways
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        best = st.session_state.walkways[0]
        drop = 0.0
        if current_sum > 0:
            drop = (current_sum - best["risk"]) / current_sum * 100.0
        st.success(
            f"추천 임시 보행로는 **{best['name']}** 입니다. "
            f"현재 작업자 이동 띠 대비 경로 위험 합계가 {drop:.1f}% 낮습니다. "
            "고위험(빨강) 교차 구간을 우회하도록 지정하세요."
        )

if st.session_state.playing:
    times = play_times()
    if len(times) <= 1 or st.session_state.play_idx >= len(times) - 1:
        st.session_state.playing = False
        st.session_state.play_idx = max(0, len(times) - 1)
    else:
        st.session_state.play_idx += 1
        time.sleep(0.08)
        st.rerun()


