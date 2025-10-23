#!/usr/bin/env python3
"""새로운 균형잡힌 얼굴형 분류 테스트"""

import math
from typing import Dict

def _rule_penalties(m: Dict[str, float]) -> Dict[str, float]:
    """
    각 얼굴형의 이상적인 값으로부터의 거리를 계산.
    더 균형잡힌 분류를 위해 절대적인 임계값 대신 상대적 거리 사용.
    """
    LWR, FJ, CWF, CWJ = m["LWR"], m["FJ"], m["CWF"], m["CWJ"]
    p: Dict[str, float] = {}

    # 각 얼굴형의 이상적인 특성값 정의
    ideal = {
        "Oval": {"LWR": 1.4, "FJ": 1.0, "CWF": 1.05, "CWJ": 1.0},
        "Round": {"LWR": 1.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
        "Square": {"LWR": 1.2, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
        "Heart": {"LWR": 1.5, "FJ": 1.3, "CWF": 1.2, "CWJ": 1.3},
        "Diamond": {"LWR": 1.4, "FJ": 0.8, "CWF": 1.6, "CWJ": 1.6},
        "Oblong": {"LWR": 1.8, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    }

    # 가중치: 각 특성의 중요도
    weights = {
        "Oval": {"LWR": 1.0, "FJ": 1.5, "CWF": 1.0, "CWJ": 1.0},
        "Round": {"LWR": 2.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
        "Square": {"LWR": 1.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
        "Heart": {"LWR": 1.0, "FJ": 2.0, "CWF": 1.5, "CWJ": 2.0},
        "Diamond": {"LWR": 1.0, "FJ": 2.0, "CWF": 2.0, "CWJ": 2.0},
        "Oblong": {"LWR": 2.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    }

    # 각 얼굴형에 대해 가중 유클리디안 거리 계산
    for shape in ideal.keys():
        penalty = 0.0
        penalty += weights[shape]["LWR"] * abs(LWR - ideal[shape]["LWR"])
        penalty += weights[shape]["FJ"] * abs(FJ - ideal[shape]["FJ"])
        penalty += weights[shape]["CWF"] * abs(CWF - ideal[shape]["CWF"])
        penalty += weights[shape]["CWJ"] * abs(CWJ - ideal[shape]["CWJ"])
        p[shape] = penalty

    return p

def _softmax_from_penalties(p: Dict[str, float], alpha: float = 2.0) -> Dict[str, float]:
    """작은 패널티 → 큰 확률이 되도록 변환. alpha를 낮춰서 더 균형잡힌 분포"""
    keys = list(p.keys())
    scores = [math.exp(-alpha * p[k]) for k in keys]
    total = sum(scores)
    probs = [s / total for s in scores]
    return {k: float(v) for k, v in zip(keys, probs)}

# 테스트 케이스들
test_cases = {
    "Oval (이상적 타원형)": {"LWR": 1.4, "FJ": 1.0, "CWF": 1.05, "CWJ": 1.0},
    "Round (둥근형)": {"LWR": 1.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Square (사각형)": {"LWR": 1.2, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Heart (하트형)": {"LWR": 1.5, "FJ": 1.3, "CWF": 1.2, "CWJ": 1.3},
    "Diamond (다이아몬드)": {"LWR": 1.4, "FJ": 0.8, "CWF": 1.6, "CWJ": 1.6},
    "Oblong (긴형)": {"LWR": 1.8, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},

    # 애매한 경우들도 테스트
    "Mixed case 1": {"LWR": 1.3, "FJ": 1.1, "CWF": 1.1, "CWJ": 1.0},
    "Mixed case 2": {"LWR": 1.6, "FJ": 0.9, "CWF": 1.3, "CWJ": 1.2},
}

print("=== 새로운 균형잡힌 얼굴형 분류 테스트 ===\n")

for name, metrics in test_cases.items():
    penalties = _rule_penalties(metrics)
    probs = _softmax_from_penalties(penalties)

    # 가장 높은 확률의 얼굴형
    best_shape = max(probs.items(), key=lambda x: x[1])

    print(f"{name}:")
    print(f"  메트릭: LWR={metrics['LWR']:.1f}, FJ={metrics['FJ']:.1f}, CWF={metrics['CWF']:.1f}, CWJ={metrics['CWJ']:.1f}")
    print(f"  예측: {best_shape[0]} ({best_shape[1]:.1%})")

    # 상위 3개 확률 출력
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  상위 3개: {[(k, f'{v:.1%}') for k, v in top3]}")
    print()