#!/usr/bin/env python3
"""긴형 편향 문제 디버깅"""

import numpy as np

# 직접 분류 로직 복사 (의존성 없이)
def _statistical_classification(m):
    """규칙 기반 얼굴형 분류"""
    LWR = m["LWR"]
    FJ = m["FJ"]
    CWF = m["CWF"]
    CWJ = m["CWJ"]

    # 각 얼굴형별 점수 계산
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

    # Oblong: 매우 긴 얼굴 (더 엄격한 조건으로 수정)
    oblong_score = 0
    if LWR >= 1.7:  # 기준을 더 엄격하게 상향 조정
        oblong_score += 0.6  # 점수도 약간 증가
    elif LWR >= 1.6:  # 중간 단계 추가
        oblong_score += 0.3
    if 0.9 <= FJ <= 1.15:  # 범위 확대
        oblong_score += 0.2  # 점수 감소
    if 0.95 <= CWF <= 1.1:  # 범위 확대
        oblong_score += 0.1  # 점수 감소

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

def debug_metrics_classification():
    """다양한 메트릭으로 분류 결과 테스트"""

    print("=== 긴형 편향 문제 디버깅 ===\n")

    # 다양한 LWR 값으로 테스트
    test_cases = [
        # LWR이 점진적으로 증가하는 케이스들
        ("매우 짧은 얼굴", {"LWR": 1.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0}),
        ("짧은 얼굴", {"LWR": 1.2, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0}),
        ("보통 얼굴", {"LWR": 1.35, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0}),
        ("긴 얼굴", {"LWR": 1.5, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0}),
        ("매우 긴 얼굴", {"LWR": 1.65, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0}),
        ("극도로 긴 얼굴", {"LWR": 1.8, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0}),
    ]

    for name, metrics in test_cases:
        result = classify_with_confidence(metrics)
        print(f"{name} (LWR={metrics['LWR']}):")
        print(f"  예측: {result['shape']} ({result['confidence']:.1%})")

        # 모든 점수 확인
        probs = result['probs']
        print(f"  전체 점수: {[(k, f'{v:.1%}') for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)]}")
        print()

    print("=== Oblong 조건 자세히 분석 ===")

    # 수정된 Oblong 조건: LWR >= 1.6/1.7
    borderline_cases = [
        {"LWR": 1.55, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},  # 경계선 아래
        {"LWR": 1.60, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},  # 중간 점수
        {"LWR": 1.70, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},  # 높은 점수
        {"LWR": 1.80, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},  # 매우 높은 점수
    ]

    for i, metrics in enumerate(borderline_cases):
        print(f"LWR = {metrics['LWR']}:")

        # 원본 점수 계산 로직 재현 (수정된 조건)
        LWR = metrics["LWR"]
        oblong_raw_score = 0
        if LWR >= 1.7:
            oblong_raw_score += 0.6
        elif LWR >= 1.6:
            oblong_raw_score += 0.3
        if 0.9 <= metrics["FJ"] <= 1.15:
            oblong_raw_score += 0.2
        if 0.95 <= metrics["CWF"] <= 1.1:
            oblong_raw_score += 0.1

        print(f"  Oblong 원본 점수: {oblong_raw_score}")

        result = classify_with_confidence(metrics)
        print(f"  최종 예측: {result['shape']} ({result['confidence']:.1%})")
        print(f"  Oblong 확률: {result['probs']['Oblong']:.1%}")
        print()

def test_realistic_metrics():
    """실제적인 얼굴 메트릭으로 테스트"""
    print("=== 실제적인 얼굴 비율 테스트 ===")

    # 실제 얼굴에서 나올 법한 메트릭들
    realistic_cases = [
        ("평범한 타원형", {"LWR": 1.32, "FJ": 1.08, "CWF": 1.12, "CWJ": 1.05}),
        ("살짝 긴 타원형", {"LWR": 1.45, "FJ": 1.02, "CWF": 1.08, "CWJ": 1.03}),
        ("보통 사각형", {"LWR": 1.28, "FJ": 0.96, "CWF": 1.02, "CWJ": 0.98}),
        ("하트형", {"LWR": 1.38, "FJ": 1.22, "CWF": 1.15, "CWJ": 1.28}),
        ("다이아몬드형", {"LWR": 1.42, "FJ": 0.88, "CWF": 1.35, "CWJ": 1.25}),
    ]

    for name, metrics in realistic_cases:
        result = classify_with_confidence(metrics)
        print(f"{name}:")
        print(f"  메트릭: LWR={metrics['LWR']:.2f}, FJ={metrics['FJ']:.2f}")
        print(f"  예측: {result['shape']} ({result['confidence']:.1%})")

        # 상위 3개 결과
        top3 = sorted(result['probs'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  상위 3개: {[(k, f'{v:.1%}') for k, v in top3]}")
        print()

if __name__ == "__main__":
    debug_metrics_classification()
    test_realistic_metrics()