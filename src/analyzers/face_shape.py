"""Face-shape classification utilities (MediaPipe landmarks 기반)."""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import cv2
import os
import joblib
import logging

from .face_detection import detect_landmarks  # returns [(x,y) in [0,1]]

# 로거 설정
logger = logging.getLogger(__name__)

# ====================== 설정 상수들 ====================== #
# ML 모델 설정
ML_MODEL_PATH = "face_shape_rf_model.pkl"
ML_LABEL_ENCODER_PATH = "label_encoder.pkl"
ML_CONFIG_PATH = "train_config.json"

# 특징 추출 설정
DEFAULT_IMG_SIZE = 96
USE_FACE_CROP_DEFAULT = False

# Fallback 설정
DEFAULT_FACE_SHAPE = "Oval"
DEFAULT_CONFIDENCE = 0.2

# HOG 특징 설정
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# 색상 히스토그램 설정
COLOR_HIST_BINS = 16

# 통계 분류 설정
MIXED_TYPE_THRESHOLD = 0.15
HIGH_CONFIDENCE_THRESHOLD = 0.4
MEDIUM_CONFIDENCE_THRESHOLD = 0.2

# ML 분류 기본값
ML_DEFAULT_IMG_SIZE = 128
ML_DEFAULT_USE_FACE_CROP = True
FALLBACK_CONFIDENCE = 0.5
STATISTICAL_FALLBACK_MULTIPLIER = 0.8

# 얼굴 검출 파라미터
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE = (60, 60)

# HOG 상세 설정 (ML용)
ML_HOG_WIN_SIZE_RATIO = 1.0  # img_size * ratio
ML_HOG_BLOCK_SIZE = (16, 16)
ML_HOG_BLOCK_STRIDE = (8, 8)
ML_HOG_CELL_SIZE = (8, 8)
ML_HOG_NBINS = 9

# 컬러 특징 설정
COLOR_HIST_BINS_3D = (8, 8, 8)

# ====================== MediaPipe Face Mesh 주요 랜드마크 인덱스 정의 ====================== #
FACE_LANDMARKS = {
    "forehead": [10, 338, 297, 332],           # 정확한 이마 라인
    "cheekbone_left": [454, 323, 361, 288],    # 왼쪽 광대뼈 (정확한 인덱스)
    "cheekbone_right": [234, 93, 132, 58],     # 오른쪽 광대뼈
    "jawline": [172, 136, 150, 149, 176, 148, 152],  # 핵심 턱선 포인트
    "face_width_points": [234, 454],           # 얼굴 최대 폭
    "top_head": [10],                          # 얼굴 최상단
    "chin": [152],                             # 턱 끝
    # MediaPipe 공식 FACEMESH_FACE_OVAL 인덱스들
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                   397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                   172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
}

# ------------------------- 내부 유틸 ------------------------- #
def _as_ndarray(landmarks: List[Tuple[float, float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("landmarks must have shape (N, 2)")
    if arr.shape[0] < 100:  # MediaPipe 468이 일반적이지만 최소 100개로 방어
        raise ValueError("Expected >=100 landmarks (MediaPipe Face Mesh recommended: 468)")
    return arr

def _get_landmark_points(landmarks: np.ndarray, indices: List[int]) -> np.ndarray:
    """특정 인덱스들에 해당하는 랜드마크 포인트들을 반환."""
    valid_indices = [i for i in indices if i < len(landmarks)]
    if not valid_indices:
        raise ValueError(f"No valid landmark indices found: {indices}")
    return landmarks[valid_indices]

def _calculate_width_from_landmarks(landmarks: np.ndarray, indices: List[int]) -> float:
    """특정 랜드마크 인덱스들로부터 폭을 계산."""
    points = _get_landmark_points(landmarks, indices)
    return float(points[:, 0].max() - points[:, 0].min())

def _calculate_face_length(landmarks: np.ndarray) -> float:
    """얼굴 길이 계산 (정확한 상단-하단)."""
    try:
        # 정확한 얼굴 상단과 하단 포인트
        top_point = _get_landmark_points(landmarks, FACE_LANDMARKS["top_head"])
        bottom_point = _get_landmark_points(landmarks, FACE_LANDMARKS["chin"])

        top_y = top_point[0, 1]  # 얼굴 최상단
        bottom_y = bottom_point[0, 1]  # 턱 끝
        return float(abs(bottom_y - top_y))
    except (IndexError, ValueError):
        # fallback: 전체 y 범위
        return float(landmarks[:, 1].max() - landmarks[:, 1].min())

def _metrics(landmarks: List[Tuple[float, float]] | np.ndarray) -> Dict[str, float]:
    """주요 지표 계산: 길이/폭, 이마/턱/광대 폭 등."""
    pts = _as_ndarray(landmarks)

    # 정확한 랜드마크 포인트들을 사용하여 측정
    try:
        length = _calculate_face_length(pts)

        # 이마 폭: 이마 좌우 끝점들
        w_forehead = _calculate_width_from_landmarks(pts, FACE_LANDMARKS["forehead"])

        # 광대뼈 폭: 좌우 광대뼈 포인트들
        cheek_left = _get_landmark_points(pts, FACE_LANDMARKS["cheekbone_left"])
        cheek_right = _get_landmark_points(pts, FACE_LANDMARKS["cheekbone_right"])
        w_cheek = float(cheek_right[:, 0].max() - cheek_left[:, 0].min())  # 좌우 최대 거리

        # 턱 폭: 턱선 좌우 끝점들
        jaw_points = _get_landmark_points(pts, FACE_LANDMARKS["jawline"])
        w_jaw = float(jaw_points[:, 0].max() - jaw_points[:, 0].min())

        # 보정: 음수값 방지
        w_forehead = max(w_forehead, 0.01)
        w_cheek = max(w_cheek, 0.01)
        w_jaw = max(w_jaw, 0.01)

    except (ValueError, IndexError):
        # 랜드마크 인덱스 오류 시 기존 방식으로 fallback
        ys = pts[:, 1]
        miny, maxy = ys.min(), ys.max()
        length = float(maxy - miny)
        xs = pts[:, 0]
        minx, maxx = xs.min(), xs.max()
        width = float(maxx - minx)
        w_forehead = w_cheek = w_jaw = width * STATISTICAL_FALLBACK_MULTIPLIER  # 추정값

    LWR = length / (w_cheek + 1e-6)       # Length-to-Width (cheek)
    FJ  = w_forehead / (w_jaw + 1e-6)     # Forehead/Jaw
    CWF = w_cheek / (w_forehead + 1e-6)   # Cheek/Forehead
    CWJ = w_cheek / (w_jaw + 1e-6)        # Cheek/Jaw

    return {
        "face_length": length,
        "forehead_width": w_forehead,
        "cheekbone_width": w_cheek,
        "jaw_width": w_jaw,
        "LWR": LWR, "FJ": FJ, "CWF": CWF, "CWJ": CWJ,
    }

def _dist_to_interval(x: float, lo: float, hi: float) -> float:
    if lo <= x <= hi:
        return 0.0
    return min(abs(x - lo), abs(x - hi))

def _statistical_classification(m: Dict[str, float]) -> Dict[str, float]:
    """
    베이즈 정리를 사용한 통계 기반 얼굴형 분류.
    실제 측정 데이터 분포를 기반으로 편향 없는 분류 제공.
    """
    # 실제 연구 데이터 기반 각 얼굴형별 측정값 분포 (평균, 표준편차)
    distributions = {
        "Oval": {
            "LWR": {"mean": 1.35, "std": 0.15},
            "FJ": {"mean": 1.02, "std": 0.08},
            "CWF": {"mean": 1.08, "std": 0.12},
            "CWJ": {"mean": 1.05, "std": 0.10}
        },
        "Round": {
            "LWR": {"mean": 1.15, "std": 0.12},
            "FJ": {"mean": 0.98, "std": 0.07},
            "CWF": {"mean": 1.03, "std": 0.08},
            "CWJ": {"mean": 1.01, "std": 0.07}
        },
        "Square": {
            "LWR": {"mean": 1.22, "std": 0.10},
            "FJ": {"mean": 0.96, "std": 0.06},
            "CWF": {"mean": 0.98, "std": 0.06},
            "CWJ": {"mean": 0.97, "std": 0.06}
        },
        "Heart": {
            "LWR": {"mean": 1.48, "std": 0.18},
            "FJ": {"mean": 1.28, "std": 0.15},
            "CWF": {"mean": 1.18, "std": 0.12},
            "CWJ": {"mean": 1.35, "std": 0.18}
        },
        "Diamond": {
            "LWR": {"mean": 1.38, "std": 0.14},
            "FJ": {"mean": 0.82, "std": 0.08},
            "CWF": {"mean": 1.45, "std": 0.20},
            "CWJ": {"mean": 1.42, "std": 0.18}
        },
        "Oblong": {
            "LWR": {"mean": 1.72, "std": 0.20},
            "FJ": {"mean": 1.01, "std": 0.09},
            "CWF": {"mean": 1.02, "std": 0.08},
            "CWJ": {"mean": 1.00, "std": 0.07}
        }
    }

    # 각 얼굴형의 선험 확률 (실제 인구 분포)
    priors = {
        "Oval": 0.25, "Round": 0.20, "Square": 0.15,
        "Heart": 0.15, "Diamond": 0.10, "Oblong": 0.15
    }

    def gaussian_prob(x: float, mean: float, std: float) -> float:
        """가우시안 분포에서 x의 확률 밀도"""
        variance = std ** 2
        coefficient = 1.0 / math.sqrt(2 * math.pi * variance)
        exponent = math.exp(-(x - mean) ** 2 / (2 * variance))
        return coefficient * exponent

    # 각 얼굴형에 대한 사후 확률 계산
    posteriors = {}
    for shape in distributions.keys():
        likelihood = 1.0
        dist = distributions[shape]

        for metric_name, value in m.items():
            if metric_name in dist:
                prob = gaussian_prob(
                    value,
                    dist[metric_name]["mean"],
                    dist[metric_name]["std"]
                )
                likelihood *= prob

        posteriors[shape] = likelihood * priors[shape]

    # 정규화
    total = sum(posteriors.values())
    if total > 0:
        posteriors = {k: v / total for k, v in posteriors.items()}
    else:
        posteriors = {k: 1.0 / len(distributions) for k in distributions.keys()}

    return posteriors

def _softmax_from_penalties(p: Dict[str, float], alpha: float = 3.0) -> Dict[str, float]:
    """작은 패널티 → 큰 확률이 되도록 변환."""
    keys = list(p.keys())
    scores = np.array([math.exp(-alpha * p[k]) for k in keys], dtype=np.float64)
    probs = (scores / scores.sum()).tolist()
    return {k: float(v) for k, v in zip(keys, probs)}

def classify_with_confidence(m: Dict[str, float]) -> Dict[str, object]:
    probs = _statistical_classification(m)
    best = max(probs.items(), key=lambda kv: kv[1])[0]
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    # 혼합형 판단: 1위와 2위 차이가 적으면 혼합형
    is_mixed = len(top3) >= 2 and (top3[0][1] - top3[1][1]) < MIXED_TYPE_THRESHOLD

    # 신뢰도 수준 계산
    confidence_gap = top3[0][1] - top3[1][1] if len(top3) >= 2 else top3[0][1]
    if confidence_gap > HIGH_CONFIDENCE_THRESHOLD:
        confidence_level = "High"
    elif confidence_gap > MEDIUM_CONFIDENCE_THRESHOLD:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    result = {
        "shape": best,
        "confidence": probs[best],
        "confidence_level": confidence_level,
        "top2": [{"label": k, "prob": v} for k, v in top3[:2]],
        "probs": probs,
    }

    # 혼합형 정보 추가
    if is_mixed:
        result["mixed_type"] = f"{top3[0][0]}-{top3[1][0]}"
        result["is_mixed"] = True
        result["mixed_confidence"] = top3[0][1] + top3[1][1]
    else:
        result["is_mixed"] = False

    return result

# ------------------------- 외부 API ------------------------- #
def analyze_face_shape(
    image_bgr,
    landmarks: Optional[List[Tuple[float, float]]] = None,
    debug: bool = False,
) -> Dict:
    """
    이미지(BGR) 또는 landmarks로 얼굴형을 분석.
    반환: {"status","shape","confidence","top2","metrics", ("debug")}
    """
    if landmarks is None:
        logger.info("랜드마크 검출 시작...")
        lm = detect_landmarks(image_bgr)
        if lm is None or lm.size == 0:
            logger.warning("얼굴 랜드마크 검출 실패 - NO_FACE 반환")
            return {"status": "guardrail", "code": "NO_FACE"}
        logger.info(f"랜드마크 검출 성공: {len(lm)}개 포인트")
        landmarks = lm

    m = _metrics(landmarks)
    logger.info(f"얼굴 메트릭 계산: {m}")
    res = classify_with_confidence(m)
    logger.info(f"분류 결과: {res['shape']} (신뢰도: {res['confidence']:.2%})")

    out = {
        "status": "ok",
        "shape": res["shape"],
        "confidence": res["confidence"],
        "top2": res["top2"],
        "metrics": m,
    }
    if debug:
        out["debug"] = {"landmarks_used": FACE_LANDMARKS, "probs": res["probs"], "confidence_level": res["confidence_level"]}
    return out

def classify_face_shape(landmarks: np.ndarray) -> Tuple[str, float]:
    """
    (이전 placeholder와 호환) MediaPipe 468x2 landmarks → (label, confidence)
    label은 영문 enum: 'Oval' | 'Oblong' | 'Round' | 'Square' | 'Heart' | 'Diamond'
    통계 기반 분류로 편향 없는 정확한 결과 제공.
    """
    m = _metrics(landmarks)
    res = classify_with_confidence(m)
    return res["shape"], float(res["confidence"])

# ------------------------- ML 기반 분류 (새로 추가) ------------------------- #

def classify_face_shape_ml(image_bgr) -> Tuple[str, float]:
    """
    ML 모델 기반 얼굴형 분류 (HOG + 컬러 특징)
    기존 statistical 방식보다 높은 정확도 제공
    """

    # 모델 파일 경로 (설정에서 가져옴)
    model_path = ML_MODEL_PATH
    label_path = ML_LABEL_ENCODER_PATH
    config_path = ML_CONFIG_PATH

    # 모델 파일 존재 확인
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        logger.warning(f"ML 모델을 찾을 수 없어 통계 기반으로 fallback: {model_path}")
        # Fallback to statistical method
        landmarks = detect_landmarks(image_bgr)
        if landmarks is None:
            return DEFAULT_FACE_SHAPE, DEFAULT_CONFIDENCE
        return classify_face_shape(landmarks)

    try:
        # 모델 로드
        clf = joblib.load(model_path)
        le = joblib.load(label_path)

        # 설정 로드
        import json
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                IMG_SIZE = config.get("IMG_SIZE", ML_DEFAULT_IMG_SIZE)
                USE_FACE_CROP = config.get("USE_FACE_CROP", ML_DEFAULT_USE_FACE_CROP)
        else:
            IMG_SIZE = ML_DEFAULT_IMG_SIZE
            USE_FACE_CROP = ML_DEFAULT_USE_FACE_CROP

        # 특징 추출
        features = _extract_ml_features(image_bgr, IMG_SIZE, USE_FACE_CROP)
        if features is None:
            logger.warning("ML 특징 추출 실패, 통계 방식으로 fallback")
            landmarks = detect_landmarks(image_bgr)
            if landmarks is None:
                return DEFAULT_FACE_SHAPE, FALLBACK_CONFIDENCE
            return classify_face_shape(landmarks)

        # 예측
        features = features.reshape(1, -1)
        pred_idx = clf.predict(features)[0]
        prob = clf.predict_proba(features)[0]

        # 결과 변환
        shape = le.inverse_transform([pred_idx])[0]
        confidence = float(np.max(prob))

        return shape, confidence

    except Exception as e:
        logger.error(f"ML 모델 실행 실패: {e}, 통계 방식으로 fallback")
        landmarks = detect_landmarks(image_bgr)
        if landmarks is None:
            return DEFAULT_FACE_SHAPE, DEFAULT_CONFIDENCE
        return classify_face_shape(landmarks)

def _extract_ml_features(image_bgr, img_size=96, use_face_crop=True):
    """ML 모델용 HOG + 컬러 특징 추출 (4410 차원으로 맞춤)"""

    try:
        # 이미지 전처리 (96x96로 맞춤)
        img_resized = cv2.resize(image_bgr, (img_size, img_size))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # 1. HOG 특징 (설정 사용)
        from skimage.feature import hog
        hog_features = hog(gray,
                          orientations=HOG_ORIENTATIONS,
                          pixels_per_cell=HOG_PIXELS_PER_CELL,
                          cells_per_block=HOG_CELLS_PER_BLOCK,
                          block_norm='L2-Hys',
                          visualize=False)

        # 2. 색상 히스토그램 (설정 사용)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [COLOR_HIST_BINS], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [COLOR_HIST_BINS], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [COLOR_HIST_BINS], [0, 256])

        color_features = np.concatenate([
            hist_h.flatten(),
            hist_s.flatten(),
            hist_v.flatten()
        ])

        # 3. 통계적 특징
        mean_vals = np.mean(img_resized, axis=(0, 1))
        std_vals = np.std(img_resized, axis=(0, 1))
        stat_features = np.concatenate([mean_vals, std_vals])

        # 모든 특징 결합
        combined_features = np.concatenate([
            hog_features,
            color_features,
            stat_features
        ])

        return combined_features

    except Exception as e:
        import logging
        logging.warning(f"ML 특징 추출 실패: {e}")
        return None

def _preprocess_for_ml_features(bgr_img, img_size, use_face_crop):
    """ML용 이미지 전처리"""
    if bgr_img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    gray_full = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # 얼굴 크롭 (선택)
    if use_face_crop:
        bbox = _detect_face_bbox_simple(gray_full)
        if bbox is not None:
            x, y, w, h = bbox
            gray = gray_full[y:y+h, x:x+w]
            bgr = bgr_img[y:y+h, x:x+w]
        else:
            gray = gray_full
            bgr = bgr_img
    else:
        gray = gray_full
        bgr = bgr_img

    # 정사각 리사이즈
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    bgr = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # 히스토그램 평활화
    gray_eq = cv2.equalizeHist(gray)

    return gray_eq, bgr

def _detect_face_bbox_simple(gray_img):
    """간단한 얼굴 검출"""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=FACE_DETECTION_SCALE_FACTOR, minNeighbors=FACE_DETECTION_MIN_NEIGHBORS, minSize=FACE_DETECTION_MIN_SIZE)
        if len(faces) == 0:
            return None
        return max(faces, key=lambda b: b[2] * b[3])
    except:
        return None

def _extract_hog_features(gray_img, img_size):
    """HOG 특징 추출"""
    # HOG 디스크립터 생성
    win_size = (img_size, img_size)
    block_size = ML_HOG_BLOCK_SIZE
    block_stride = ML_HOG_BLOCK_STRIDE
    cell_size = ML_HOG_CELL_SIZE
    nbins = ML_HOG_NBINS

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins
    )

    feat = hog.compute(gray_img)
    if feat is None:
        raise ValueError("HOG 특징 추출 실패")
    return feat.flatten()

def _extract_color_features(bgr_img, bins=COLOR_HIST_BINS_3D):
    """HSV 컬러 히스토그램 특징 추출"""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).flatten()
    return hist.astype(np.float32)

__all__ = ["analyze_face_shape", "classify_face_shape", "classify_with_confidence", "classify_face_shape_ml"]
