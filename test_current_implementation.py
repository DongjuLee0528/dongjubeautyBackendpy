#!/usr/bin/env python3
"""현재 구현된 통계 기반 분류기 테스트"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.analyzers.face_shape import classify_with_confidence, _metrics

# 468개 랜드마크를 가진 더미 데이터 생성
def create_test_landmarks(lwr, fj, cwf, cwj):
    """주어진 메트릭에 맞는 랜드마크 생성"""
    landmarks = np.random.rand(468, 2).astype(np.float32)

    # 얼굴 길이 설정 (LWR 기반)
    face_height = 0.6
    face_width = face_height / lwr
    landmarks[:, 1] = landmarks[:, 1] * face_height + 0.2  # y: 0.2-0.8

    # 이마 설정 (FJ 기반)
    forehead_width = face_width * fj
    landmarks[10:15, 0] = np.linspace(0.5 - forehead_width/2, 0.5 + forehead_width/2, 5)

    # 광대뼈 설정 (CWF 기반)
    cheek_width = forehead_width * cwf
    landmarks[50:55, 0] = np.linspace(0.5 - cheek_width/2, 0.5 + cheek_width/2, 5)

    # 턱 설정 (CWJ 기반)
    jaw_width = cheek_width / cwj
    landmarks[150:155, 0] = np.linspace(0.5 - jaw_width/2, 0.5 + jaw_width/2, 5)

    return landmarks

# 다양한 얼굴형 테스트
test_cases = [
    ("Oval형 특성", {"LWR": 1.35, "FJ": 1.02, "CWF": 1.08, "CWJ": 1.05}),
    ("Round형 특성", {"LWR": 1.15, "FJ": 0.98, "CWF": 1.03, "CWJ": 1.01}),
    ("Square형 특성", {"LWR": 1.22, "FJ": 0.96, "CWF": 0.98, "CWJ": 0.97}),
    ("Heart형 특성", {"LWR": 1.48, "FJ": 1.28, "CWF": 1.18, "CWJ": 1.35}),
    ("Diamond형 특성", {"LWR": 1.38, "FJ": 0.82, "CWF": 1.45, "CWJ": 1.42}),
    ("Oblong형 특성", {"LWR": 1.72, "FJ": 1.01, "CWF": 1.02, "CWJ": 1.00}),
]

print("=== 현재 구현된 통계 기반 분류기 테스트 ===\n")

try:
    for name, target_metrics in test_cases:
        # 메트릭을 직접 사용해서 분류
        result = classify_with_confidence(target_metrics)

        print(f"{name}:")
        print(f"  입력 메트릭: {target_metrics}")
        print(f"  예측: {result['shape']} ({result['confidence']:.1%})")
        print(f"  신뢰도 수준: {result.get('confidence_level', 'N/A')}")

        # 상위 3개 결과
        top3 = sorted(result['probs'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  상위 3개: {[(k, f'{v:.1%}') for k, v in top3]}")
        print()

except Exception as e:
    print(f"에러 발생: {e}")
    print("스택트레이스:")
    import traceback
    traceback.print_exc()