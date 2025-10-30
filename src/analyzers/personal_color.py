"""Personal color inference pipeline built on MediaPipe landmarks."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
from skimage import color as skcolor

from .face_detection import denormalise_landmarks, detect_landmarks
from .skin_roi import DEFAULT_PATCH_SIZE, extract_skin_rois

logger = logging.getLogger(__name__)


DEFAULT_PALETTES: Dict[str, list[str]] = {
    "Spring": ["#F8C2A3", "#F5E6A8", "#F7DAD9", "#FFB7B2"],
    "Summer": ["#A8C8E8", "#E9BFD1", "#9EC3B1", "#F5E6A8"],
    "Autumn": ["#B46A55", "#D39F6B", "#F2C572", "#865640"],
    "Winter": ["#5B6DCE", "#B6CAE3", "#6ED0D4", "#F5F5F5"],
}


THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ita": {"warm": 28.0, "cool": 10.0},
    "L": {"too_dark": 25.0, "too_bright": 90.0},
    "roi": {"min_pixels": 900.0, "min_brightness": 10.0, "min_variance": 5.0},
}


def classify_from_metrics(metrics: Mapping[str, float]) -> Tuple[str, str]:
    """Return season and tone labels derived from LAB metrics."""

    ita = metrics["ITA"]
    l_star = metrics["L"]
    b_star = metrics["b"]

    warm_threshold = THRESHOLDS["ita"]["warm"]
    cool_threshold = THRESHOLDS["ita"]["cool"]

    if ita >= warm_threshold:
        tone = "Warm"
    elif ita <= cool_threshold:
        tone = "Cool"
    else:
        tone = "Warm" if b_star >= 0 else "Cool"

    if tone == "Warm":
        season = "Spring" if l_star >= 55.0 else "Autumn"
    else:
        season = "Summer" if l_star >= 55.0 else "Winter"

    return season, tone


def classify_with_confidence(metrics: Mapping[str, float]) -> Dict[str, Any]:
    """Return detailed classification with confidence scores."""

    ita = metrics["ITA"]
    l_star = metrics["L"]
    b_star = metrics["b"]

    warm_threshold = THRESHOLDS["ita"]["warm"]
    cool_threshold = THRESHOLDS["ita"]["cool"]

    # Tone 신뢰도 계산
    if ita >= warm_threshold:
        tone = "Warm"
        tone_confidence = min(1.0, (ita - warm_threshold) / 20.0 + 0.7)
    elif ita <= cool_threshold:
        tone = "Cool"
        tone_confidence = min(1.0, (cool_threshold - ita) / 20.0 + 0.7)
    else:
        # 모호한 경우 b값으로 판단
        tone = "Warm" if b_star >= 0 else "Cool"
        tone_confidence = min(0.8, abs(b_star) / 10.0 + 0.5)

    # Season 판정 및 신뢰도
    brightness_threshold = 55.0
    if tone == "Warm":
        season = "Spring" if l_star >= brightness_threshold else "Autumn"
        season_confidence = min(1.0, abs(l_star - brightness_threshold) / 20.0 + 0.6)
    else:
        season = "Summer" if l_star >= brightness_threshold else "Winter"
        season_confidence = min(1.0, abs(l_star - brightness_threshold) / 20.0 + 0.6)

    # 전체 신뢰도는 tone과 season 신뢰도의 평균
    overall_confidence = (tone_confidence + season_confidence) / 2.0

    # 가능한 다른 분류들 (top2)
    all_seasons = ["Spring", "Summer", "Autumn", "Winter"]
    alternatives = [s for s in all_seasons if s != season]

    # 간단한 대안 점수 계산
    if tone == "Warm":
        alt_season = "Autumn" if season == "Spring" else "Spring"
    else:
        alt_season = "Winter" if season == "Summer" else "Summer"

    alt_confidence = 1.0 - overall_confidence

    return {
        "season": season,
        "tone": tone,
        "confidence": round(overall_confidence, 3),
        "tone_confidence": round(tone_confidence, 3),
        "season_confidence": round(season_confidence, 3),
        "top2": [
            {"label": season, "prob": round(overall_confidence, 3)},
            {"label": alt_season, "prob": round(alt_confidence, 3)}
        ],
        "is_mixed": overall_confidence < 0.75,  # 75% 미만이면 혼합형으로 간주
        "mixed_type": f"{season}-{alt_season}" if overall_confidence < 0.75 else None
    }


class PersonalColorAnalyzer:
    """Run personal color analysis over a BGR image."""

    def __init__(self, min_valid_rois: int = 1) -> None:
        self.min_valid_rois = min_valid_rois

    def analyze(
        self, image_bgr: np.ndarray, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        opts: Dict[str, Any] = dict(options or {})
        trace_id = opts.get("trace_id")

        landmarks_norm = detect_landmarks(image_bgr)
        if landmarks_norm is None:
            return self._build_response(
                status="guardrail", code="NO_FACE", trace_id=trace_id
            )

        height, width = image_bgr.shape[:2]
        landmarks_px = denormalise_landmarks(landmarks_norm, width, height)

        rois, roi_metrics = extract_skin_rois(
            image_bgr,
            landmarks_px,
            patch_size=DEFAULT_PATCH_SIZE,
            thresholds=THRESHOLDS["roi"],
        )

        if len(rois) < self.min_valid_rois:
            logger.info("ROI guardrail triggered", extra={"roi_metrics": roi_metrics})
            return self._build_response(
                status="guardrail",
                code="LOW_QUALITY",
                trace_id=trace_id,
                landmarks=landmarks_norm,
            )

        metrics = self._compute_color_metrics(rois)

        # 상세 분류 정보 계산
        detailed_classification = classify_with_confidence(metrics)
        season = detailed_classification["season"]
        tone = detailed_classification["tone"]
        palette = DEFAULT_PALETTES.get(season, DEFAULT_PALETTES["Summer"])

        return self._build_response(
            status="ok",
            season=season,
            tone=tone,
            confidence=detailed_classification["confidence"],
            tone_confidence=detailed_classification["tone_confidence"],
            season_confidence=detailed_classification["season_confidence"],
            top2=detailed_classification["top2"],
            is_mixed=detailed_classification["is_mixed"],
            mixed_type=detailed_classification["mixed_type"],
            metrics=metrics,
            palette=palette,
            landmarks=landmarks_norm,
            trace_id=trace_id,
        )

    def _compute_color_metrics(self, rois: Dict[str, np.ndarray]) -> Dict[str, float]:
        lab_values = []
        for name, patch_bgr in rois.items():
            rgb_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            lab_patch = skcolor.rgb2lab(rgb_patch.astype(np.float32) / 255.0)
            mean_lab = lab_patch.reshape(-1, 3).mean(axis=0)
            lab_values.append(mean_lab)
            logger.debug("ROI %s mean LAB: %s", name, mean_lab)

        lab_array = np.vstack(lab_values)
        avg_L, avg_a, avg_b = lab_array.mean(axis=0)
        ita = math.degrees(math.atan((avg_L - 50.0) / (avg_b + 1e-6)))

        metrics = {
            "L": round(float(avg_L), 2),
            "a": round(float(avg_a), 2),
            "b": round(float(avg_b), 2),
            "ITA": round(float(ita), 2),
        }
        return metrics

    def _build_response(
        self,
        *,
        status: str,
        code: Optional[str] = None,
        season: Optional[str] = None,
        tone: Optional[str] = None,
        confidence: Optional[float] = None,
        tone_confidence: Optional[float] = None,
        season_confidence: Optional[float] = None,
        top2: Optional[list] = None,
        is_mixed: Optional[bool] = None,
        mixed_type: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        palette: Optional[list[str]] = None,
        landmarks: Optional[np.ndarray] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"status": status}
        if code:
            payload["code"] = code
        if season:
            payload["season"] = season
        if tone:
            payload["tone"] = tone
        if confidence is not None:
            payload["confidence"] = confidence
        if tone_confidence is not None:
            payload["tone_confidence"] = tone_confidence
        if season_confidence is not None:
            payload["season_confidence"] = season_confidence
        if top2 is not None:
            payload["top2"] = top2
        if is_mixed is not None:
            payload["is_mixed"] = is_mixed
        if mixed_type is not None:
            payload["mixed_type"] = mixed_type
        if metrics:
            payload["metrics"] = metrics
        if palette:
            payload["palette"] = palette
        if landmarks is not None:
            payload["landmarks"] = landmarks.tolist()
        if trace_id:
            payload["traceId"] = trace_id
        return payload


__all__ = ["PersonalColorAnalyzer", "THRESHOLDS", "classify_from_metrics", "classify_with_confidence"]
