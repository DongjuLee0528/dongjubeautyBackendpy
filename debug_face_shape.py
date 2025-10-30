#!/usr/bin/env python3
"""다양한 얼굴형 분류 테스트"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analyzers.face_shape import _rule_penalties, _softmax_from_penalties

# 다양한 얼굴형 특성을 가진 메트릭들
test_cases = {
    "Oval (이상적)": {"LWR": 1.4, "FJ": 1.0, "CWF": 1.05, "CWJ": 1.0},
    "Round (둥근형)": {"LWR": 1.0, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Square (사각형)": {"LWR": 1.1, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
    "Heart (하트형)": {"LWR": 1.5, "FJ": 1.5, "CWF": 1.3, "CWJ": 1.4},
    "Diamond (다이아몬드)": {"LWR": 1.4, "FJ": 0.8, "CWF": 1.6, "CWJ": 1.6},
    "Oblong (긴형)": {"LWR": 1.8, "FJ": 1.0, "CWF": 1.0, "CWJ": 1.0},
}

print("=== 얼굴형 분류 테스트 ===\n")

for name, metrics in test_cases.items():
    penalties = _rule_penalties(metrics)
    probs = _softmax_from_penalties(penalties)

    # 가장 높은 확률의 얼굴형
    best_shape = max(probs.items(), key=lambda x: x[1])

    print(f"{name}:")
    print(f"  메트릭: {metrics}")
    print(f"  예측: {best_shape[0]} ({best_shape[1]:.2%})")
    print(f"  상위 3개: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
    print()