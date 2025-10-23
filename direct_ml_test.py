#!/usr/bin/env python3
"""
ML 모델 직접 테스트 (MediaPipe 없이)
"""

import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os

def extract_ml_features(img):
    """ML 특징 추출 (face_shape.py와 동일)"""
    if img is None:
        return None

    # 이미지 전처리
    img = cv2.resize(img, (96, 96))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. HOG 특징
    hog_features = hog(gray,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=False)

    # 2. 색상 히스토그램
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    color_features = np.concatenate([
        hist_h.flatten(),
        hist_s.flatten(),
        hist_v.flatten()
    ])

    # 3. 통계적 특징
    mean_vals = np.mean(img, axis=(0, 1))
    std_vals = np.std(img, axis=(0, 1))
    stat_features = np.concatenate([mean_vals, std_vals])

    # 모든 특징 결합
    combined_features = np.concatenate([
        hog_features,
        color_features,
        stat_features
    ])

    return combined_features

def test_improved_ml_model():
    """개선된 ML 모델 직접 테스트"""
    print("🧪 개선된 ML 모델 직접 테스트")

    # 모델 로드
    try:
        clf = joblib.load('face_shape_rf_model.pkl')
        le = joblib.load('label_encoder.pkl')
        print("✅ 개선된 모델 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 테스트 이미지들
    test_cases = [
        ("../faceshape-master/published_dataset/heart/img_no_1.jpg", "Heart"),
        ("../faceshape-master/published_dataset/oval/img_no_201.jpg", "Oval"),
        ("../faceshape-master/published_dataset/round/img_no_301.jpg", "Round"),
        ("../faceshape-master/published_dataset/square/img_no_401.jpg", "Square"),
        ("../faceshape-master/published_dataset/oblong/img_no_101.jpg", "Oblong")
    ]

    print(f"\n📊 테스트 결과 (개선된 74.1% 모델):")
    print(f"{'실제':<10} {'예측':<10} {'신뢰도':<8} {'상태'}")
    print("-" * 40)

    correct = 0
    total = 0

    for img_path, expected in test_cases:
        if not os.path.exists(img_path):
            print(f"{expected:<10} {'파일없음':<10} {'N/A':<8} ❌")
            continue

        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"{expected:<10} {'로드실패':<10} {'N/A':<8} ❌")
            continue

        try:
            # 특징 추출
            features = extract_ml_features(img)
            if features is None:
                print(f"{expected:<10} {'특징실패':<10} {'N/A':<8} ❌")
                continue

            # 예측
            features = features.reshape(1, -1)
            pred_idx = clf.predict(features)[0]
            prob = clf.predict_proba(features)[0]

            predicted = le.inverse_transform([pred_idx])[0]
            confidence = np.max(prob)

            # 결과 출력
            status = "✅" if predicted == expected else "❌"
            if predicted == expected:
                correct += 1

            print(f"{expected:<10} {predicted:<10} {confidence:<8.1%} {status}")
            total += 1

        except Exception as e:
            print(f"{expected:<10} {'오류':<10} {'N/A':<8} ❌")
            print(f"   오류: {e}")
            total += 1

    # 결과 요약
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n🎯 실제 테스트 정확도: {correct}/{total} = {accuracy:.1f}%")
        print(f"📈 학습 시 테스트 정확도: 74.1%")

        if accuracy >= 70:
            print("✅ 모델이 실제로도 우수한 성능을 보입니다!")
        elif accuracy >= 50:
            print("⚠️ 모델 성능이 보통입니다.")
        else:
            print("❌ 실제 성능이 예상보다 낮습니다.")

        # 개선도 계산
        print(f"\n📊 개선 현황:")
        print(f"   원본 모델: 47% 정확도")
        print(f"   개선된 모델: 74.1% 정확도 (+27.1%p)")
        print(f"   실제 테스트: {accuracy:.1f}% 정확도")

if __name__ == "__main__":
    test_improved_ml_model()