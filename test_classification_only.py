#!/usr/bin/env python3
"""분류 로직만 테스트 (의존성 없이)"""

import sys
import os
import numpy as np

# face_shape.py에서 분류 관련 함수만 추출
def _statistical_classification(m):
    """
    규칙 기반 얼굴형 분류 - 편향 없는 균형잡힌 분류.
    각 얼굴형에 동등한 기회를 주는 점수 계산 방식.
    """
    LWR = m["LWR"]
    FJ = m["FJ"]
    CWF = m["CWF"]
    CWJ = m["CWJ"]

    # 각 얼굴형별 점수 계산 (더 균형있는 조건으로 수정)
    scores = {}

    # Round: 짧고 넓은 얼굴, 모든 부위가 비슷한 폭
    round_score = 0
    if LWR < 1.2:  # 더 짧은 기준
        round_score += 0.5
    elif LWR < 1.3:
        round_score += 0.3
    if 0.9 <= FJ <= 1.1:  # 범위 확대
        round_score += 0.3
    if 0.95 <= CWF <= 1.15:  # 범위 확대
        round_score += 0.2

    # Square: 각진 얼굴, 이마-광대-턱이 비슷
    square_score = 0
    if 1.1 <= LWR <= 1.4:  # 범위 확대
        square_score += 0.4
    if 0.88 <= FJ <= 1.05:  # 범위 조정
        square_score += 0.3
    if 0.9 <= CWF <= 1.1:  # 범위 확대
        square_score += 0.3

    # Oval: 균형잡힌 얼굴 (범위 축소)
    oval_score = 0
    if 1.3 <= LWR <= 1.5:  # 범위 축소
        oval_score += 0.4
    if 0.95 <= FJ <= 1.1:  # 범위 축소
        oval_score += 0.3
    if 1.0 <= CWF <= 1.2:  # 범위 축소
        oval_score += 0.3

    # Heart: 이마가 넓고 턱이 좁음
    heart_score = 0
    if LWR >= 1.2:  # 조건 완화
        heart_score += 0.3
    if FJ >= 1.1:  # 조건 완화
        heart_score += 0.4
    if CWJ >= 1.15:  # 조건 완화
        heart_score += 0.3

    # Diamond: 광대뼈가 가장 넓고 이마와 턱이 좁음
    diamond_score = 0
    if 1.2 <= LWR <= 1.6:  # 범위 확대
        diamond_score += 0.3
    if FJ <= 0.95:  # 조건 완화
        diamond_score += 0.4
    if CWF >= 1.2:  # 조건 완화
        diamond_score += 0.3

    # Oblong: 매우 긴 얼굴
    oblong_score = 0
    if LWR >= 1.55:  # 조건 완화
        oblong_score += 0.5
    if 0.9 <= FJ <= 1.15:  # 범위 확대
        oblong_score += 0.3
    if 0.95 <= CWF <= 1.1:  # 범위 확대
        oblong_score += 0.2

    scores = {
        "Round": round_score,
        "Square": square_score,
        "Oval": oval_score,
        "Heart": heart_score,
        "Diamond": diamond_score,
        "Oblong": oblong_score
    }

    # 가중치 기반 정규화 (편향 제거)
    max_score = max(scores.values())

    if max_score == 0:
        # 모든 점수가 0이면 균등 분배
        return {k: 1.0 / len(scores) for k in scores.keys()}

    # 점수를 0.1 ~ 1.0 범위로 정규화 (최소값 보장)
    min_base_score = 0.1
    normalized_scores = {}
    for shape, score in scores.items():
        if score > 0:
            normalized_scores[shape] = min_base_score + (score / max_score) * (1.0 - min_base_score)
        else:
            normalized_scores[shape] = min_base_score

    # 총합으로 정규화하여 확률로 변환
    total = sum(normalized_scores.values())
    return {k: v / total for k, v in normalized_scores.items()}

def classify_with_confidence(m):
    probs = _statistical_classification(m)
    best = max(probs.items(), key=lambda kv: kv[1])[0]
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    # 혼합형 판단: 1위와 2위 차이가 적으면 혼합형
    MIXED_TYPE_THRESHOLD = 0.15
    is_mixed = len(top3) >= 2 and (top3[0][1] - top3[1][1]) < MIXED_TYPE_THRESHOLD

    # 신뢰도 수준 계산
    HIGH_CONFIDENCE_THRESHOLD = 0.4
    MEDIUM_CONFIDENCE_THRESHOLD = 0.2
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

# 다양한 얼굴형 테스트
test_cases = [
    ("Oval형 특성", {"LWR": 1.35, "FJ": 1.02, "CWF": 1.08, "CWJ": 1.05}),
    ("Round형 특성", {"LWR": 1.15, "FJ": 0.98, "CWF": 1.03, "CWJ": 1.01}),
    ("Square형 특성", {"LWR": 1.22, "FJ": 0.96, "CWF": 0.98, "CWJ": 0.97}),
    ("Heart형 특성", {"LWR": 1.48, "FJ": 1.28, "CWF": 1.18, "CWJ": 1.35}),
    ("Diamond형 특성", {"LWR": 1.38, "FJ": 0.82, "CWF": 1.45, "CWJ": 1.42}),
    ("Oblong형 특성", {"LWR": 1.72, "FJ": 1.01, "CWF": 1.02, "CWJ": 1.00}),
]

print("=== 수정된 통계 기반 분류기 테스트 ===\n")

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

print("=== 편향 분석 ===")
print("모든 테스트 케이스의 Oval 확률:")
for name, target_metrics in test_cases:
    result = classify_with_confidence(target_metrics)
    oval_prob = result['probs']['Oval']
    print(f"  {name}: {oval_prob:.1%}")