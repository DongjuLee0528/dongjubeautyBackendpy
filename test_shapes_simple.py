#!/usr/bin/env python3
"""간단한 얼굴형 분류 테스트 (cv2 의존성 없이)"""

import math
from typing import Dict

def _dist_to_interval(x: float, lo: float, hi: float) -> float:
    if lo <= x <= hi:
        return 0.0
    return min(abs(x - lo), abs(x - hi))

def _rule_penalties(m: Dict[str, float]) -> Dict[str, float]:
    """규칙에서 벗어난 정도(패널티). 값이 작을수록 해당 클래스일 가능성이 높음."""
    LWR, FJ, CWF, CWJ = m["LWR"], m["FJ"], m["CWF"], m["CWJ"]
    p: Dict[str, float] = {}

    # Oval: 이상적인 비율 (1.3-1.6 범위)
    p["Oval"] = _dist_to_interval(LWR, 1.3, 1.6) + _dist_to_interval(FJ, 0.95, 1.05) + _dist_to_interval(CWF, 1.0, 1.1)

    # Round: 짧고 넓은 얼굴 (LWR이 낮아야 함)
    p["Round"] = max(0.0, LWR - 1.2) + _dist_to_interval(FJ, 0.9, 1.1) + _dist_to_interval(CWJ, 0.9, 1.1)

    # Square: 비슷한 비율, 각진 턱
    p["Square"] = _dist_to_interval(FJ, 0.9, 1.1) + _dist_to_interval(CWF, 0.9, 1.1) + _dist_to_interval(CWJ, 0.9, 1.1) + max(0.0, LWR - 1.3)

    # Heart: 넓은 이마, 좁은 턱
    p["Heart"] = max(0.0, 1.15 - FJ) + max(0.0, 1.1 - CWF) + max(0.0, 1.15 - CWJ) + _dist_to_interval(LWR, 1.3, 1.7)

    # Diamond: 좁은 이마와 턱, 넓은 광대뼈 (더 엄격한 조건)
    p["Diamond"] = max(0.0, 1.5 - CWF) + max(0.0, 1.5 - CWJ) + _dist_to_interval(FJ, 0.7, 0.9) + _dist_to_interval(LWR, 1.3, 1.6)

    # Oblong: 긴 얼굴 (LWR이 높아야 함)
    p["Oblong"] = max(0.0, 1.6 - LWR) + _dist_to_interval(FJ, 0.9, 1.1) + _dist_to_interval(CWF, 0.9, 1.1)

    return p

def _softmax_from_penalties(p: Dict[str, float], alpha: float = 4.0) -> Dict[str, float]:
    """작은 패널티 → 큰 확률이 되도록 변환."""
    import numpy as np
    keys = list(p.keys())
    scores = [math.exp(-alpha * p[k]) for k in keys]
    total = sum(scores)
    probs = [s / total for s in scores]
    return {k: float(v) for k, v in zip(keys, probs)}

# 다양한 얼굴형 특성을 가진 메트릭들
test_cases = {
    "Oval (이상적 타원형)": {"LWR": 1.4, "FJ": 1.0, "CWF": 1.05, "CWJ": 1.0},
    "Round (둥근형)": {"LWR": 1.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Square (사각형)": {"LWR": 1.2, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Heart (하트형)": {"LWR": 1.5, "FJ": 1.3, "CWF": 1.2, "CWJ": 1.3},
    "Diamond (다이아몬드)": {"LWR": 1.4, "FJ": 0.8, "CWF": 1.6, "CWJ": 1.6},
    "Oblong (긴형)": {"LWR": 1.8, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
}

print("=== 수정된 얼굴형 분류 테스트 ===\n")

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

    # 패널티 출력 (디버깅용)
    print(f"  패널티: {[(k, f'{v:.2f}') for k, v in sorted(penalties.items(), key=lambda x: x[1])]}")
    print()