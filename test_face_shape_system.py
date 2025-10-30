#!/usr/bin/env python3
"""
현재 face_shape.py 시스템 테스트
"""

import sys
import os
sys.path.append('src')

import cv2
import numpy as np

def test_face_shape_analysis():
    """얼굴형 분석 시스템 테스트"""
    print("🔍 얼굴형 분석 시스템 테스트 시작")

    try:
        from analyzers.face_shape import FaceShapeAnalyzer
        print("✅ FaceShapeAnalyzer 모듈 로드 성공")
    except Exception as e:
        print(f"❌ 모듈 로드 실패: {e}")
        return

    # 분석기 초기화
    try:
        analyzer = FaceShapeAnalyzer()
        print("✅ FaceShapeAnalyzer 초기화 성공")
    except Exception as e:
        print(f"❌ 분석기 초기화 실패: {e}")
        return

    # 테스트 이미지들
    test_images = [
        ("../faceshape-master/published_dataset/heart/img_no_1.jpg", "Heart"),
        ("../faceshape-master/published_dataset/oval/img_no_201.jpg", "Oval"),
        ("../faceshape-master/published_dataset/round/img_no_301.jpg", "Round"),
        ("../faceshape-master/published_dataset/square/img_no_401.jpg", "Square"),
        ("../faceshape-master/published_dataset/oblong/img_no_101.jpg", "Oblong")
    ]

    print(f"\n📊 테스트 결과:")
    print(f"{'실제':<8} {'ML 예측':<8} {'통계 예측':<10} {'최종 결과':<8} {'ML 신뢰도':<8} {'상태'}")
    print("-" * 70)

    correct_ml = 0
    correct_stat = 0
    correct_final = 0
    total = 0

    for img_path, expected in test_images:
        if not os.path.exists(img_path):
            print(f"{expected:<8} {'파일없음':<8} {'파일없음':<10} {'파일없음':<8} {'N/A':<8} ❌")
            continue

        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"{expected:<8} {'로드실패':<8} {'로드실패':<10} {'로드실패':<8} {'N/A':<8} ❌")
            continue

        try:
            # ML 모델 테스트
            ml_result = analyzer.classify_face_shape_ml(img)
            ml_shape = ml_result.get('face_shape', 'Unknown') if ml_result else 'Failed'
            ml_confidence = ml_result.get('confidence', 0) if ml_result else 0

            # 통계 모델 테스트
            stat_result = analyzer.classify_face_shape_statistical(img)
            stat_shape = stat_result.get('face_shape', 'Unknown') if stat_result else 'Failed'

            # 최종 통합 분석
            final_result = analyzer.analyze_face_shape(img)
            final_shape = final_result.get('face_shape', 'Unknown') if final_result else 'Failed'

            # 정확도 체크
            if ml_shape == expected:
                correct_ml += 1
            if stat_shape == expected:
                correct_stat += 1
            if final_shape == expected:
                correct_final += 1

            # 상태 표시
            status = "✅" if final_shape == expected else "❌"

            print(f"{expected:<8} {ml_shape:<8} {stat_shape:<10} {final_shape:<8} {ml_confidence:<8.1%} {status}")

            total += 1

        except Exception as e:
            print(f"{expected:<8} {'오류':<8} {'오류':<10} {'오류':<8} {'N/A':<8} ❌")
            print(f"   오류 상세: {e}")
            total += 1

    # 결과 요약
    if total > 0:
        ml_accuracy = (correct_ml / total) * 100
        stat_accuracy = (correct_stat / total) * 100
        final_accuracy = (correct_final / total) * 100

        print(f"\n🎯 정확도 요약:")
        print(f"   ML 모델: {correct_ml}/{total} = {ml_accuracy:.1f}%")
        print(f"   통계 모델: {correct_stat}/{total} = {stat_accuracy:.1f}%")
        print(f"   최종 통합: {correct_final}/{total} = {final_accuracy:.1f}%")

        if final_accuracy >= 80:
            print("✅ 시스템이 우수한 성능을 보입니다!")
        elif final_accuracy >= 60:
            print("⚠️ 시스템 성능이 보통입니다.")
        else:
            print("❌ 시스템 성능 개선이 필요합니다.")
    else:
        print("❌ 테스트할 수 있는 이미지가 없습니다.")

def test_single_image():
    """단일 이미지 상세 테스트"""
    print(f"\n🔬 단일 이미지 상세 분석")

    test_image = "../faceshape-master/published_dataset/round/img_no_301.jpg"

    if not os.path.exists(test_image):
        print("테스트 이미지를 찾을 수 없습니다.")
        return

    try:
        from analyzers.face_shape import FaceShapeAnalyzer
        analyzer = FaceShapeAnalyzer()

        img = cv2.imread(test_image)
        result = analyzer.analyze_face_shape(img)

        print(f"이미지: {test_image}")
        print(f"분석 결과: {result}")

    except Exception as e:
        print(f"상세 분석 실패: {e}")

if __name__ == "__main__":
    test_face_shape_analysis()
    test_single_image()