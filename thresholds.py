"""KOSHA 근거를 반영한 연구용 충돌 거리 임계값.

법적 단일 거리 기준은 없다. 안전보건규칙 제98조는 현장 제한속도를
지정하도록 할 뿐 충돌 이격거리(m)를 정하지 않는다. 따라서 장비 상용
제원과 현장에서 지정된 제한속도로 보정한 연구용 근삿값을 쓴다.
"""

from __future__ import annotations

# --- 현장에서 확인되는 운행 조건 ---------------------------------
# KOSHA GUIDE D-C-4-2025 / 안전보건규칙 제98조:
# 차량계 건설기계 작업 시 지형·지반에 맞는 제한속도를 미리 정해야 한다.
# (최대제한속도가 시속 10 km 이하인 장비는 이 조항에서 제외)
# 샘플 CCTV에도 "현장내 제한속도 20" 표지가 있어 20 km/h를 현장 지정값으로 둔다.
SITE_SPEED_LIMIT_KMH = 20.0
SITE_SPEED_LIMIT_MPS = SITE_SPEED_LIMIT_KMH / 3.6  # 5.556 m/s

# 타이어식 굴착기 자체 주행능력은 30~40 km/h (KOSHA GUIDE 5.(1)(나)).
# 현장 제한속도가 더 낮으므로 임계값 보정에는 현장 지정값 20 km/h를 쓴다.
WHEEL_EXCAVATOR_TRAVEL_KMH = 35.0

# --- 장비 크기 (제조사 제원 근사, KOSHA 부록 제원표 계열) ----------
# 현장에서 흔한 6~14 t급 굴착기 차폭 약 2.5 m.
EXCAVATOR_WIDTH_M = 2.5
EXCAVATOR_HALF_WIDTH_M = EXCAVATOR_WIDTH_M / 2.0  # 1.25 m

# 안전보건규칙 제20조·제200조, KOSHA 작업반경 그림:
# 붐·암·버킷 선회에 근로자가 접근하면 안 된다. 중형 굴착기 작업반경
# 안쪽을 약 3.75 m로 근사한다 (법적 수치가 아니라 제원 기반 참고치).
SWING_CLEARANCE_M = 3.75

# --- 연구용 임계값 ------------------------------------------------
# 정적: 장비 반폭 + 선회 여유 ≈ 5.0 m
STATIC_THRESHOLD_M = round(EXCAVATOR_HALF_WIDTH_M + SWING_CLEARANCE_M, 1)

# 동적: 정적 임계 + 제한속도 × 1초 예지
# 20 km/h * 1.0 s = 5.56 m → 5.0 + 5.56 ≈ 10.6 → 10 m 근사
PREDICT_HORIZON_S = 1.0
DYNAMIC_THRESHOLD_M = round(STATIC_THRESHOLD_M + SITE_SPEED_LIMIT_MPS * PREDICT_HORIZON_S, 1)

# 히트맵 가우시안 퍼짐. 위치 불확실성 ≈ 정적 임계와 같게 둔다.
SIGMA_M = STATIC_THRESHOLD_M

# 영상 한 장의 가로가 약 45 m 현장을 담는다고 근사 (교정 전 참고).
VISIBLE_WIDTH_M = 45.0

PREDICT_STEPS = 3

THRESHOLD_NOTE = (
    "충돌 거리 임계값은 법령에 정해진 수치가 아닙니다. "
    "KOSHA GUIDE D-C-4-2025가 인용하는 안전보건규칙 제98조에 따라 "
    f"현장 제한속도 {SITE_SPEED_LIMIT_KMH:.0f} km/h와 "
    f"중형 굴착기 차폭 {EXCAVATOR_WIDTH_M} m·작업반경 여유를 반영한 "
    f"연구용 근삿값입니다. 정적 {STATIC_THRESHOLD_M} m, "
    f"동적(1초 예지) {DYNAMIC_THRESHOLD_M} m."
)


def meters_to_pixels(meters: float, image_width_px: int) -> float:
    return meters / VISIBLE_WIDTH_M * float(image_width_px)
