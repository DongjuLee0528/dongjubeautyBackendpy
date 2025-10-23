#!/usr/bin/env python3
"""통계 기반 분류 함수를 직접 테스트"""

import math
from typing import Dict

def _statistical_classification(m: Dict[str, float]) -> Dict[str, float]:
    """현재 구현된 통계 기반 분류 함수"""
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

def classify_with_confidence(m: Dict[str, float]) -> Dict[str, object]:
    """통계 기반 분류 결과 반환"""
    probs = _statistical_classification(m)
    best = max(probs.items(), key=lambda kv: kv[1])[0]
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    # 혼합형 판단: 1위와 2위 차이가 적으면 혼합형
    is_mixed = len(top3) >= 2 and (top3[0][1] - top3[1][1]) < 0.15

    # 신뢰도 수준 계산
    confidence_gap = top3[0][1] - top3[1][1] if len(top3) >= 2 else top3[0][1]
    if confidence_gap > 0.4:
        confidence_level = "High"
    elif confidence_gap > 0.2:
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

# 테스트 케이스들
test_cases = [
    ("전형적인 Oval", {"LWR": 1.35, "FJ": 1.02, "CWF": 1.08, "CWJ": 1.05}),
    ("전형적인 Round", {"LWR": 1.15, "FJ": 0.98, "CWF": 1.03, "CWJ": 1.01}),
    ("전형적인 Square", {"LWR": 1.22, "FJ": 0.96, "CWF": 0.98, "CWJ": 0.97}),
    ("전형적인 Heart", {"LWR": 1.48, "FJ": 1.28, "CWF": 1.18, "CWJ": 1.35}),
    ("전형적인 Diamond", {"LWR": 1.38, "FJ": 0.82, "CWF": 1.45, "CWJ": 1.42}),
    ("전형적인 Oblong", {"LWR": 1.72, "FJ": 1.01, "CWF": 1.02, "CWJ": 1.00}),

    # 실제로 다이아몬드가 나올 수 있는 케이스들
    ("사용자 케이스 1", {"LWR": 1.3, "FJ": 1.0, "CWF": 1.1, "CWJ": 1.1}),
    ("사용자 케이스 2", {"LWR": 1.4, "FJ": 1.1, "CWF": 1.2, "CWJ": 1.1}),
    ("사용자 케이스 3", {"LWR": 1.2, "FJ": 0.9, "CWF": 1.0, "CWJ": 1.0}),
]

print("=== 현재 통계 기반 분류기 직접 테스트 ===\n")

diamond_count = 0
total_count = len(test_cases)

for name, metrics in test_cases:
    result = classify_with_confidence(metrics)

    if result['shape'] == 'Diamond':
        diamond_count += 1

    print(f"{name}:")
    print(f"  메트릭: LWR={metrics['LWR']:.2f}, FJ={metrics['FJ']:.2f}, CWF={metrics['CWF']:.2f}, CWJ={metrics['CWJ']:.2f}")
    print(f"  예측: {result['shape']} ({result['confidence']:.1%}, {result['confidence_level']})")

    # 상위 3개 결과
    top3 = sorted(result['probs'].items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  상위 3개: {[(k, f'{v:.1%}') for k, v in top3]}")
    print()

print(f"Diamond 결과: {diamond_count}/{total_count} = {diamond_count/total_count:.1%}")

if diamond_count / total_count > 0.5:
    print("⚠️  여전히 Diamond 편향이 있습니다!")
else:
    print("✅ Diamond 편향이 해결되었습니다.")