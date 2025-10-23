# -*- coding: utf-8 -*-
"""
얼굴형 분류 (HOG / HSV 컬러 히스토그램 / HOG+Color)
- 데이터 구조: published_dataset/<heart|oblong|oval|round|square>/*.jpg
- 모델: RandomForestClassifier
- 저장: 모델(pkl), 라벨 인코더(pkl), 설정(json)
"""

import os
import json
from glob import glob

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# =======================
# CONFIG
# =======================
DATA_DIR = "faceshape-master/published_dataset"
CLASSES = ["heart", "oblong", "oval", "round", "square"]  # 폴더명(소문자)
IMG_SIZE = 128  # 정사각 리사이즈 크기
USE_FACE_CROP = True  # 얼굴 검출 후 영역 크롭 사용
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_PATH = "face_shape_rf_model.pkl"
LABEL_PATH = "label_encoder.pkl"
CONFIG_PATH = "train_config.json"

# 특성 추출 방식 ('hog' | 'color' | 'hog_color')
FEATURE_TYPE = "hog_color"

print("라이브러리 로드 완료!")

# =======================
# HOG 디스크립터
# =======================
def create_hog(win=IMG_SIZE):
    win_size = (win, win)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    return cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins
    )

HOG = create_hog()

# =======================
# 데이터 로드
# =======================
def load_dataset(data_dir=DATA_DIR, classes=CLASSES):
    """published_dataset에서 이미지 경로와 라벨 로드"""
    rows = []
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            print(f"[경고] 폴더 없음: {folder}")
            continue
        files = [f for f in glob(os.path.join(folder, "*")) if os.path.isfile(f)]
        print(f"{cls}: {len(files)}개 이미지 발견")
        for f in files:
            rows.append({"image_path": f, "shape": cls.capitalize()})
    df = pd.DataFrame(rows)
    print(f"\n총 {len(df)}개 이미지 로드!")
    if not df.empty:
        print("\n얼굴형별 분포:")
        print(df["shape"].value_counts())
    return df

# =======================
# 전처리 & 특징 추출
# =======================
def detect_face_bbox(gray_img):
    """얼굴 bbox 반환 (x, y, w, h). 없으면 None."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # 가장 큰 얼굴
    return max(faces, key=lambda b: b[2] * b[3])

def preprocess_for_features(bgr_img):
    """
    공통 전처리:
    - (선택) 얼굴 크롭
    - 그레이스케일(히스토그램 평활화)
    - 정사각 리사이즈
    - 컬러 특징용 BGR 정사각 리사이즈
    """
    if bgr_img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    gray_full = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    if USE_FACE_CROP:
        bbox = detect_face_bbox(gray_full)
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
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # HOG 성능 개선을 위한 평활화
    gray_eq = cv2.equalizeHist(gray)

    return gray_eq, bgr

def _extract_hog_from_gray(gray_resized_equalized):
    feat = HOG.compute(gray_resized_equalized)
    if feat is None:
        raise ValueError("HOG 특징 추출 실패.")
    return feat.flatten()

def compute_color_hist(bgr_img, bins=(8, 8, 8)):
    """HSV 3D 히스토그램 → L1 정규화 후 평탄화"""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).flatten()
    return hist.astype(np.float32)

def extract_features(image_path, feature_type=FEATURE_TYPE):
    """이미지에서 선택한 방식(HOG / Color / HOG+Color)으로 특징 추출"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    gray_eq, bgr_sq = preprocess_for_features(img)

    if feature_type == "hog":
        hog = _extract_hog_from_gray(gray_eq)
        return hog

    elif feature_type == "color":
        color_hist = compute_color_hist(bgr_sq)
        return color_hist

    elif feature_type == "hog_color":
        hog = _extract_hog_from_gray(gray_eq)
        color_hist = compute_color_hist(bgr_sq)
        return np.concatenate([hog, color_hist], axis=0)

    else:
        raise ValueError(f"알 수 없는 FEATURE_TYPE: {feature_type}")

def build_features(df):
    X, y = [], []
    skipped = 0
    for _, row in df.iterrows():
        path = row["image_path"]
        label = row["shape"]
        try:
            feat = extract_features(path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            skipped += 1
            # 필요 시 상세 로그
            # print(f"[스킵] {path}: {e}")
            continue

    if len(X) == 0:
        raise RuntimeError("특징을 추출한 이미지가 없습니다. 데이터/전처리를 확인하세요.")
    X = np.vstack(X)
    y = np.array(y)
    print(f"\n특징 추출 완료: 사용 {len(y)}개, 스킵 {skipped}개, 벡터 길이: {X.shape[1]}")
    return X, y

# =======================
# 학습 파이프라인
# =======================
def train():
    # 1) 데이터 로드
    df = load_dataset()
    if df.empty:
        raise RuntimeError("데이터가 비어 있습니다.")

    # 2) 특징/라벨 생성
    X, y_text = build_features(df)

    # 3) 라벨 인코딩
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    # 4) Stratified 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 5) 모델 학습 (클래스 불균형 대응)
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight='balanced_subsample'
    )
    clf.fit(X_train, y_train)

    # 6) 평가
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Accuracy] {acc:.4f}")
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred))

    # 7) 저장
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_PATH)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "IMG_SIZE": IMG_SIZE,
                "USE_FACE_CROP": USE_FACE_CROP,
                "FEATURE_TYPE": FEATURE_TYPE,
                "HOG": {
                    "win": IMG_SIZE,
                    "block_size": (16, 16),
                    "block_stride": (8, 8),
                    "cell_size": (8, 8),
                    "nbins": 9,
                },
                "TEST_SIZE": TEST_SIZE,
                "RANDOM_STATE": RANDOM_STATE,
                "CLASSES": CLASSES,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n모델 저장 완료: {MODEL_PATH}")
    print(f"라벨 인코더 저장 완료: {LABEL_PATH}")
    print(f"설정 저장 완료: {CONFIG_PATH}")

# =======================
# 단일 이미지 예측
# =======================
def predict_image(image_path):
    """단일 이미지 예측 (학습 시 설정된 FEATURE_TYPE을 로드하여 동일 파이프라인 보장)"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        raise FileNotFoundError("모델/라벨 파일이 없습니다. 먼저 train()을 실행하세요.")

    # 학습 시 사용한 FEATURE_TYPE 동기화
    global FEATURE_TYPE
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            FEATURE_TYPE = cfg.get("FEATURE_TYPE", FEATURE_TYPE)

    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_PATH)

    feat = extract_features(image_path).reshape(1, -1)
    pred_idx = clf.predict(feat)[0]
    prob = clf.predict_proba(feat)[0]
    label = le.inverse_transform([pred_idx])[0]
    return label, float(np.max(prob))

# =======================
# 실행부
# =======================
if __name__ == "__main__":
    print("얼굴형 머신러닝 시작!")
    train()

    # 사용 예시:
    # label, confidence = predict_image("some_test_image.jpg")
    # print(f"예측: {label} (신뢰도: {confidence:.3f})")