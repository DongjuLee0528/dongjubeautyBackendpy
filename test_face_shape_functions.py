#!/usr/bin/env python3
"""
face_shape.py 함수들 직접 테스트
"""

import sys
import os
sys.path.append('src')

import cv2
import numpy as np

def test_face_shape_functions():
    """얼굴형 분석 함수들 테스트"""
    print("🔍 얼굴형 분석 함수 테스트 시작")

    try:
        from analyzers.face_shape import classify_face_shape_ml
        print("✅ classify_face_shape_ml 함수 로드 성공")
    except Exception as e:
        print(f"❌ 함수 로드 실패: {e}")
        return

    # 테스트 이미지들
    test_images = [
        ("../faceshape-master/published_dataset/heart/img_no_1.jpg", "Heart"),
        ("../faceshape-master/published_dataset/oval/img_no_201.jpg", "Oval"),
        ("../faceshape-master/published_dataset/round/img_no_301.jpg", "Round"),
        ("../faceshape-master/published_dataset/square/img_no_401.jpg", "Square"),
        ("../faceshape-master/published_dataset/oblong/img_no_101.jpg", "Oblong")
    ]

    print(f"\n📊 ML 모델 테스트 결과:")
    print(f"{'실제':<8} {'ML 예측':<8} {'신뢰도':<8} {'상태'}")
    print("-" * 35)

    correct = 0
    total = 0

    for img_path, expected in test_images:
        if not os.path.exists(img_path):
            print(f"{expected:<8} {'파일없음':<8} {'N/A':<8} ❌")
            continue

        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"{expected:<8} {'로드실패':<8} {'N/A':<8} ❌")
            continue

        try:
            # ML 모델 테스트
            ml_result = classify_face_shape_ml(img)

            if ml_result and len(ml_result) >= 2:
                ml_shape = ml_result[0]
                ml_confidence = ml_result[1]
            else:
                ml_shape = 'Failed'
                ml_confidence = 0

            # 정확도 체크
            if ml_shape == expected:
                correct += 1

            # 상태 표시
            status = "✅" if ml_shape == expected else "❌"

            print(f"{expected:<8} {ml_shape:<8} {ml_confidence:<8.1%} {status}")

            total += 1

        except Exception as e:
            print(f"{expected:<8} {'오류':<8} {'N/A':<8} ❌")
            print(f"   오류 상세: {e}")
            total += 1

    # 결과 요약
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n🎯 ML 모델 정확도: {correct}/{total} = {accuracy:.1f}%")

        if accuracy >= 80:
            print("✅ ML 모델이 우수한 성능을 보입니다!")
        elif accuracy >= 60:
            print("⚠️ ML 모델 성능이 보통입니다.")
        else:
            print("❌ ML 모델 성능 개선이 필요합니다.")
    else:
        print("❌ 테스트할 수 있는 이미지가 없습니다.")

def test_single_image_detailed():
    """단일 이미지 상세 테스트"""
    print(f"\n🔬 단일 이미지 상세 분석")

    test_image = "../faceshape-master/published_dataset/round/img_no_301.jpg"

    if not os.path.exists(test_image):
        print("테스트 이미지를 찾을 수 없습니다.")
        return

    try:
        from analyzers.face_shape import classify_face_shape_ml

        img = cv2.imread(test_image)
        result = classify_face_shape_ml(img)

        print(f"이미지: {test_image}")
        print(f"ML 분석 결과: {result}")

        # 이미지 정보
        print(f"이미지 크기: {img.shape}")

    except Exception as e:
        print(f"상세 분석 실패: {e}")
        import traceback
        traceback.print_exc()

def check_model_files():
    """모델 파일 존재 여부 확인"""
    print(f"\n📁 모델 파일 확인")

    model_files = [
        "face_shape_rf_model.pkl",
        "label_encoder.pkl",
        "train_config.json"
    ]

    for file in model_files:
        if os.path.exists(file):
            print(f"✅ {file} 존재")
        else:
            print(f"❌ {file} 없음")

if __name__ == "__main__":
    check_model_files()
    test_face_shape_functions()
    test_single_image_detailed()