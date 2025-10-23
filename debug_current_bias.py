#!/usr/bin/env python3
"""현재 Oval 편향 문제 디버깅"""

import math
from typing import Dict

def _rule_penalties(m: Dict[str, float]) -> Dict[str, float]:
    """현재 실제 코드와 동일한 패널티 계산"""
    LWR, FJ, CWF, CWJ = m["LWR"], m["FJ"], m["CWF"], m["CWJ"]
    p: Dict[str, float] = {}

    # 현재 코드의 이상값
    ideal = {
        "Oval": {"LWR": 1.4, "FJ": 1.0, "CWF": 1.05, "CWJ": 1.0},
        "Round": {"LWR": 1.1, "FJ": 0.95, "CWF": 0.98, "CWJ": 0.98},
        "Square": {"LWR": 1.25, "FJ": 0.98, "CWF": 0.97, "CWJ": 0.97},
        "Heart": {"LWR": 1.5, "FJ": 1.25, "CWF": 1.15, "CWJ": 1.25},
        "Diamond": {"LWR": 1.4, "FJ": 0.85, "CWF": 1.4, "CWJ": 1.4},
        "Oblong": {"LWR": 1.7, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    }

    # 현재 코드의 가중치
    weights = {
        "Oval": {"LWR": 1.5, "FJ": 2.0, "CWF": 1.0, "CWJ": 1.0},
        "Round": {"LWR": 3.0, "FJ": 1.0, "CWF": 1.5, "CWJ": 1.5},
        "Square": {"LWR": 1.0, "FJ": 1.5, "CWF": 1.5, "CWJ": 1.5},
        "Heart": {"LWR": 1.0, "FJ": 3.0, "CWF": 2.0, "CWJ": 3.0},
        "Diamond": {"LWR": 1.0, "FJ": 3.0, "CWF": 3.0, "CWJ": 3.0},
        "Oblong": {"LWR": 3.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    }

    for shape in ideal.keys():
        penalty = 0.0
        penalty += weights[shape]["LWR"] * abs(LWR - ideal[shape]["LWR"])
        penalty += weights[shape]["FJ"] * abs(FJ - ideal[shape]["FJ"])
        penalty += weights[shape]["CWF"] * abs(CWF - ideal[shape]["CWF"])
        penalty += weights[shape]["CWJ"] * abs(CWJ - ideal[shape]["CWJ"])
        p[shape] = penalty

    return p

def _softmax_from_penalties(p: Dict[str, float], alpha: float = 3.0) -> Dict[str, float]:
    """현재 코드와 동일한 softmax"""
    keys = list(p.keys())
    scores = [math.exp(-alpha * p[k]) for k in keys]
    total = sum(scores)
    probs = [s / total for s in scores]
    return {k: float(v) for k, v in zip(keys, probs)}

# 다양한 테스트 케이스
test_cases = {
    "극단적 Round": {"LWR": 0.9, "FJ": 0.9, "CWF": 0.9, "CWJ": 0.9},
    "극단적 Oblong": {"LWR": 2.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "극단적 Diamond": {"LWR": 1.4, "FJ": 0.7, "CWF": 1.8, "CWJ": 1.8},
    "극단적 Heart": {"LWR": 1.6, "FJ": 1.5, "CWF": 1.3, "CWJ": 1.5},
    "완전 평균값": {"LWR": 1.3, "FJ": 1.0, "CWF": 1.1, "CWJ": 1.1},
}

print("=== Oval 편향 문제 디버깅 ===\n")

for name, metrics in test_cases.items():
    penalties = _rule_penalties(metrics)
    probs = _softmax_from_penalties(penalties)

    best_shape = max(probs.items(), key=lambda x: x[1])

    print(f"{name}:")
    print(f"  메트릭: LWR={metrics['LWR']:.1f}, FJ={metrics['FJ']:.1f}, CWF={metrics['CWF']:.1f}, CWJ={metrics['CWJ']:.1f}")
    print(f"  예측: {best_shape[0]} ({best_shape[1]:.1%})")

    # 패널티 상세 분석
    print(f"  패널티 순위: {sorted(penalties.items(), key=lambda x: x[1])}")
    print()

# Oval의 패널티가 항상 낮은 이유 분석
print("=== Oval 패널티가 낮은 이유 분석 ===")
test_metric = {"LWR": 1.3, "FJ": 1.0, "CWF": 1.1, "CWJ": 1.1}

ideal = {
    "Oval": {"LWR": 1.4, "FJ": 1.0, "CWF": 1.05, "CWJ": 1.0},
    "Round": {"LWR": 1.1, "FJ": 0.95, "CWF": 0.98, "CWJ": 0.98},
}

weights = {
    "Oval": {"LWR": 1.5, "FJ": 2.0, "CWF": 1.0, "CWJ": 1.0},
    "Round": {"LWR": 3.0, "FJ": 1.0, "CWF": 1.5, "CWJ": 1.5},
}

for shape in ["Oval", "Round"]:
    print(f"\n{shape} 패널티 계산:")
    lwr_penalty = weights[shape]["LWR"] * abs(test_metric["LWR"] - ideal[shape]["LWR"])
    fj_penalty = weights[shape]["FJ"] * abs(test_metric["FJ"] - ideal[shape]["FJ"])
    cwf_penalty = weights[shape]["CWF"] * abs(test_metric["CWF"] - ideal[shape]["CWF"])
    cwj_penalty = weights[shape]["CWJ"] * abs(test_metric["CWJ"] - ideal[shape]["CWJ"])

    print(f"  LWR: {weights[shape]['LWR']} * |{test_metric['LWR']:.1f} - {ideal[shape]['LWR']:.2f}| = {lwr_penalty:.3f}")
    print(f"  FJ:  {weights[shape]['FJ']} * |{test_metric['FJ']:.1f} - {ideal[shape]['FJ']:.2f}| = {fj_penalty:.3f}")
    print(f"  CWF: {weights[shape]['CWF']} * |{test_metric['CWF']:.1f} - {ideal[shape]['CWF']:.2f}| = {cwf_penalty:.3f}")
    print(f"  CWJ: {weights[shape]['CWJ']} * |{test_metric['CWJ']:.1f} - {ideal[shape]['CWJ']:.2f}| = {cwj_penalty:.3f}")
    print(f"  총합: {lwr_penalty + fj_penalty + cwf_penalty + cwj_penalty:.3f}")