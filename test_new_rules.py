#!/usr/bin/env python3
"""새로운 규칙 기반 분류 테스트"""

import math
from typing import Dict

def _dist_to_interval(x: float, lo: float, hi: float) -> float:
    if lo <= x <= hi:
        return 0.0
    return min(abs(x - lo), abs(x - hi))

def _rule_penalties(m: Dict[str, float]) -> Dict[str, float]:
    """새로운 규칙 기반 패널티 계산"""
    LWR, FJ, CWF, CWJ = m["LWR"], m["FJ"], m["CWF"], m["CWJ"]
    p: Dict[str, float] = {}

    # Oval: 균형잡힌 비율 (모든 비율이 1.0 근처)
    p["Oval"] = (
        abs(LWR - 1.4) * 2.0 +  # 이상적인 길이/폭 비율
        abs(FJ - 1.0) * 3.0 +   # 이마와 턱이 균형
        abs(CWF - 1.0) * 2.0 +  # 광대뼈와 이마가 균형
        abs(CWJ - 1.0) * 2.0    # 광대뼈와 턱이 균형
    )

    # Round: 짧고 넓은 얼굴 (LWR이 낮고, 모든 폭이 비슷)
    p["Round"] = (
        max(0, LWR - 1.2) * 4.0 +      # LWR이 1.2 이상이면 패널티
        abs(FJ - 1.0) * 1.0 +          # 이마/턱 비율은 덜 중요
        abs(CWF - 1.0) * 2.0 +         # 광대뼈/이마 균형
        abs(CWJ - 1.0) * 2.0           # 광대뼈/턱 균형
    )

    # Square: 각진 얼굴 (모든 폭이 비슷하고 적당한 길이)
    p["Square"] = (
        _dist_to_interval(LWR, 1.1, 1.3) * 2.0 +  # 적당한 LWR
        abs(FJ - 1.0) * 2.0 +                      # 이마/턱 균형
        abs(CWF - 1.0) * 2.0 +                     # 광대뼈/이마 균형
        abs(CWJ - 1.0) * 2.0                       # 광대뼈/턱 균형
    )

    # Heart: 넓은 이마, 좁은 턱 (FJ > 1.0, CWJ > 1.0)
    p["Heart"] = (
        _dist_to_interval(LWR, 1.3, 1.6) * 1.0 +   # 적당한 길이
        max(0, 1.15 - FJ) * 4.0 +                   # 이마가 턱보다 넓어야 함
        max(0, 1.1 - CWF) * 2.0 +                   # 광대뼈가 이마보다 좁아야 함
        max(0, 1.15 - CWJ) * 4.0                    # 광대뼈가 턱보다 넓어야 함
    )

    # Diamond: 좁은 이마와 턱, 넓은 광대뼈 (CWF > 1.0, CWJ > 1.0, FJ < 1.0)
    p["Diamond"] = (
        _dist_to_interval(LWR, 1.3, 1.5) * 1.0 +   # 적당한 길이
        max(0, FJ - 0.9) * 3.0 +                    # 이마가 턱보다 작거나 비슷
        max(0, 1.2 - CWF) * 4.0 +                   # 광대뼈가 이마보다 넓어야 함
        max(0, 1.2 - CWJ) * 4.0                     # 광대뼈가 턱보다 넓어야 함
    )

    # Oblong: 긴 얼굴 (LWR > 1.6)
    p["Oblong"] = (
        max(0, 1.6 - LWR) * 4.0 +      # LWR이 1.6 이상이어야 함
        abs(FJ - 1.0) * 1.0 +          # 이마/턱 비율은 덜 중요
        abs(CWF - 1.0) * 1.0 +         # 폭 비율들은 덜 중요
        abs(CWJ - 1.0) * 1.0
    )

    return p

def _softmax_from_penalties(p: Dict[str, float], alpha: float = 3.0) -> Dict[str, float]:
    """패널티를 확률로 변환"""
    keys = list(p.keys())
    scores = [math.exp(-alpha * p[k]) for k in keys]
    total = sum(scores)
    probs = [s / total for s in scores]
    return {k: float(v) for k, v in zip(keys, probs)}

# 각 얼굴형의 특징적인 메트릭으로 테스트
test_cases = {
    "Round 특징": {"LWR": 1.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Square 특징": {"LWR": 1.2, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Oval 특징": {"LWR": 1.4, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Heart 특징": {"LWR": 1.5, "FJ": 1.3, "CWF": 0.9, "CWJ": 1.3},
    "Diamond 특징": {"LWR": 1.4, "FJ": 0.8, "CWF": 1.4, "CWJ": 1.4},
    "Oblong 특징": {"LWR": 1.8, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},

    # 애매한 케이스들
    "중간값 1": {"LWR": 1.3, "FJ": 1.0, "CWF": 1.1, "CWJ": 1.1},
    "중간값 2": {"LWR": 1.5, "FJ": 1.1, "CWF": 1.2, "CWJ": 1.1},
}

print("=== 새로운 규칙 기반 분류 테스트 ===\n")

for name, metrics in test_cases.items():
    penalties = _rule_penalties(metrics)
    probs = _softmax_from_penalties(penalties)

    best_shape = max(probs.items(), key=lambda x: x[1])
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

    print(f"{name}:")
    print(f"  메트릭: LWR={metrics['LWR']:.1f}, FJ={metrics['FJ']:.1f}, CWF={metrics['CWF']:.1f}, CWJ={metrics['CWJ']:.1f}")
    print(f"  예측: {best_shape[0]} ({best_shape[1]:.1%})")
    print(f"  상위 3개: {[(k, f'{v:.1%}') for k, v in top3]}")
    print(f"  패널티: {[(k, f'{v:.2f}') for k, v in sorted(penalties.items(), key=lambda x: x[1])]}")
    print()