#!/usr/bin/env python3
"""
개선된 모델 테스트
"""

import sys
import os
sys.path.append('src')

from analyzers.face_shape import FaceShapeAnalyzer
import cv2
import numpy as np

def test_model():
    """개선된 모델 테스트"""
    print("🧪 개선된 얼굴형 분류 모델 테스트")

    analyzer = FaceShapeAnalyzer()

    # 테스트 이미지들 로드
    test_images = [
        "../faceshape-master/published_dataset/heart/img_no_1.jpg",
        "../faceshape-master/published_dataset/oval/img_no_201.jpg",
        "../faceshape-master/published_dataset/round/img_no_301.jpg",
        "../faceshape-master/published_dataset/square/img_no_401.jpg",
        "../faceshape-master/published_dataset/oblong/img_no_101.jpg"
    ]

    expected = ["Heart", "Oval", "Round", "Square", "Oblong"]

    print(f"\n📊 테스트 결과:")
    print(f"{'실제':<10} {'예측':<10} {'신뢰도':<8} {'상태'}")
    print("-" * 40)

    correct = 0
    total = 0

    for i, (img_path, expected_shape) in enumerate(zip(test_images, expected)):
        if not os.path.exists(img_path):
            print(f"{expected_shape:<10} {'파일없음':<10} {'N/A':<8} ❌")
            continue

        try:
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                print(f"{expected_shape:<10} {'로드실패':<10} {'N/A':<8} ❌")
                continue

            # 얼굴형 분류
            result = analyzer.analyze_face_shape(img)

            if result and 'face_shape' in result:
                predicted = result['face_shape']
                confidence = result.get('ml_confidence', 0)

                status = "✅" if predicted == expected_shape else "❌"
                if predicted == expected_shape:
                    correct += 1

                print(f"{expected_shape:<10} {predicted:<10} {confidence:<8.1%} {status}")
            else:
                print(f"{expected_shape:<10} {'실패':<10} {'N/A':<8} ❌")

            total += 1

        except Exception as e:
            print(f"{expected_shape:<10} {'오류':<10} {'N/A':<8} ❌")
            print(f"   오류: {e}")
            total += 1

    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n🎯 테스트 정확도: {correct}/{total} = {accuracy:.1f}%")

        if accuracy >= 70:
            print("✅ 모델 성능 우수!")
        elif accuracy >= 50:
            print("⚠️ 모델 성능 보통")
        else:
            print("❌ 모델 성능 개선 필요")
    else:
        print("❌ 테스트할 수 있는 이미지가 없습니다.")

def test_ml_vs_statistical():
    """ML 모델과 통계 모델 비교"""
    print(f"\n🔬 ML vs 통계 방법 성능 비교")

    analyzer = FaceShapeAnalyzer()

    test_image = "../faceshape-master/published_dataset/round/img_no_301.jpg"

    if os.path.exists(test_image):
        img = cv2.imread(test_image)

        # ML 결과
        ml_result = analyzer.classify_face_shape_ml(img)

        # 통계 결과
        stat_result = analyzer.classify_face_shape_statistical(img)

        print(f"ML 모델 결과: {ml_result}")
        print(f"통계 모델 결과: {stat_result}")

        # 최종 통합 결과
        final_result = analyzer.analyze_face_shape(img)
        print(f"최종 결과: {final_result}")

if __name__ == "__main__":
    test_model()
    test_ml_vs_statistical()