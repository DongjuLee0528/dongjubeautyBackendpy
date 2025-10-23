import unittest
from unittest.mock import patch
import numpy as np

from src.analyzers.face_shape import (
    analyze_face_shape,
    classify_face_shape,
    _metrics,
    _rule_penalties,
    classify_with_confidence,
    FACE_LANDMARKS
)


class FaceShapeAnalysisTests(unittest.TestCase):

    def setUp(self):
        # 468개의 기본 랜드마크 생성 (정규화된 좌표)
        self.landmarks_468 = np.random.rand(468, 2).astype(np.float32)

        # 각 얼굴형에 맞는 특징적인 랜드마크 생성
        self.oval_landmarks = self._create_oval_landmarks()
        self.round_landmarks = self._create_round_landmarks()
        self.square_landmarks = self._create_square_landmarks()
        self.heart_landmarks = self._create_heart_landmarks()
        self.diamond_landmarks = self._create_diamond_landmarks()
        self.oblong_landmarks = self._create_oblong_landmarks()

    def _create_oval_landmarks(self):
        """타원형 얼굴 랜드마크 생성 (LWR: 1.4, 균형잡힌 비율)"""
        landmarks = np.random.rand(468, 2).astype(np.float32)
        # 얼굴 길이 설정 (0.3 ~ 0.9)
        landmarks[:, 1] = landmarks[:, 1] * 0.6 + 0.2  # 0.2-0.8 범위
        # 균형잡힌 폭 설정
        landmarks[FACE_LANDMARKS["forehead"], 0] = np.linspace(0.3, 0.7, len(FACE_LANDMARKS["forehead"]))
        landmarks[FACE_LANDMARKS["cheekbone_left"], 0] = np.linspace(0.25, 0.45, len(FACE_LANDMARKS["cheekbone_left"]))
        landmarks[FACE_LANDMARKS["cheekbone_right"], 0] = np.linspace(0.55, 0.75, len(FACE_LANDMARKS["cheekbone_right"]))
        landmarks[FACE_LANDMARKS["jawline"][:8], 0] = np.linspace(0.35, 0.65, 8)  # 턱선 일부만
        return landmarks

    def _create_round_landmarks(self):
        """둥근형 얼굴 랜드마크 생성 (LWR: 1.0, 비슷한 길이와 폭)"""
        landmarks = np.random.rand(468, 2).astype(np.float32)
        # 짧은 얼굴 길이
        landmarks[:, 1] = landmarks[:, 1] * 0.4 + 0.3  # 0.3-0.7 범위
        # 넓은 폭
        landmarks[FACE_LANDMARKS["forehead"], 0] = np.linspace(0.2, 0.8, len(FACE_LANDMARKS["forehead"]))
        landmarks[FACE_LANDMARKS["cheekbone_left"], 0] = np.linspace(0.15, 0.45, len(FACE_LANDMARKS["cheekbone_left"]))
        landmarks[FACE_LANDMARKS["cheekbone_right"], 0] = np.linspace(0.55, 0.85, len(FACE_LANDMARKS["cheekbone_right"]))
        landmarks[FACE_LANDMARKS["jawline"][:8], 0] = np.linspace(0.25, 0.75, 8)
        return landmarks

    def _create_square_landmarks(self):
        """사각형 얼굴 랜드마크 생성 (균등한 비율, 각진 턱)"""
        landmarks = np.random.rand(468, 2).astype(np.float32)
        landmarks[:, 1] = landmarks[:, 1] * 0.5 + 0.25  # 0.25-0.75 범위
        # 이마, 광대, 턱 폭이 비슷하게
        width = 0.5
        landmarks[FACE_LANDMARKS["forehead"], 0] = np.linspace(0.25, 0.75, len(FACE_LANDMARKS["forehead"]))
        landmarks[FACE_LANDMARKS["cheekbone_left"], 0] = np.linspace(0.22, 0.47, len(FACE_LANDMARKS["cheekbone_left"]))
        landmarks[FACE_LANDMARKS["cheekbone_right"], 0] = np.linspace(0.53, 0.78, len(FACE_LANDMARKS["cheekbone_right"]))
        landmarks[FACE_LANDMARKS["jawline"][:8], 0] = np.linspace(0.25, 0.75, 8)
        return landmarks

    def _create_heart_landmarks(self):
        """하트형 얼굴 랜드마크 생성 (넓은 이마, 좁은 턱)"""
        landmarks = np.random.rand(468, 2).astype(np.float32)
        landmarks[:, 1] = landmarks[:, 1] * 0.6 + 0.2
        # 넓은 이마
        landmarks[FACE_LANDMARKS["forehead"], 0] = np.linspace(0.15, 0.85, len(FACE_LANDMARKS["forehead"]))
        # 보통 광대
        landmarks[FACE_LANDMARKS["cheekbone_left"], 0] = np.linspace(0.25, 0.45, len(FACE_LANDMARKS["cheekbone_left"]))
        landmarks[FACE_LANDMARKS["cheekbone_right"], 0] = np.linspace(0.55, 0.75, len(FACE_LANDMARKS["cheekbone_right"]))
        # 좁은 턱
        landmarks[FACE_LANDMARKS["jawline"][:8], 0] = np.linspace(0.4, 0.6, 8)
        return landmarks

    def _create_diamond_landmarks(self):
        """다이아몬드형 얼굴 랜드마크 생성 (좁은 이마와 턱, 넓은 광대)"""
        landmarks = np.random.rand(468, 2).astype(np.float32)
        landmarks[:, 1] = landmarks[:, 1] * 0.55 + 0.225
        # 좁은 이마
        landmarks[FACE_LANDMARKS["forehead"], 0] = np.linspace(0.35, 0.65, len(FACE_LANDMARKS["forehead"]))
        # 넓은 광대
        landmarks[FACE_LANDMARKS["cheekbone_left"], 0] = np.linspace(0.15, 0.4, len(FACE_LANDMARKS["cheekbone_left"]))
        landmarks[FACE_LANDMARKS["cheekbone_right"], 0] = np.linspace(0.6, 0.85, len(FACE_LANDMARKS["cheekbone_right"]))
        # 좁은 턱
        landmarks[FACE_LANDMARKS["jawline"][:8], 0] = np.linspace(0.38, 0.62, 8)
        return landmarks

    def _create_oblong_landmarks(self):
        """직사각형 얼굴 랜드마크 생성 (긴 얼굴)"""
        landmarks = np.random.rand(468, 2).astype(np.float32)
        # 긴 얼굴
        landmarks[:, 1] = landmarks[:, 1] * 0.8 + 0.1  # 0.1-0.9 범위
        # 보통 폭
        landmarks[FACE_LANDMARKS["forehead"], 0] = np.linspace(0.3, 0.7, len(FACE_LANDMARKS["forehead"]))
        landmarks[FACE_LANDMARKS["cheekbone_left"], 0] = np.linspace(0.28, 0.47, len(FACE_LANDMARKS["cheekbone_left"]))
        landmarks[FACE_LANDMARKS["cheekbone_right"], 0] = np.linspace(0.53, 0.72, len(FACE_LANDMARKS["cheekbone_right"]))
        landmarks[FACE_LANDMARKS["jawline"][:8], 0] = np.linspace(0.32, 0.68, 8)
        return landmarks

    def test_metrics_calculation(self):
        """메트릭 계산이 정상적으로 동작하는지 테스트"""
        metrics = _metrics(self.landmarks_468)

        required_keys = ["face_length", "forehead_width", "cheekbone_width", "jaw_width", "LWR", "FJ", "CWF", "CWJ"]
        for key in required_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
            self.assertGreater(metrics[key], 0)

    def test_oval_classification(self):
        """타원형 얼굴 분류 테스트"""
        shape, confidence = classify_face_shape(self.oval_landmarks)
        self.assertEqual(shape, "Oval")
        self.assertGreater(confidence, 0.3)  # 최소 30% 신뢰도

    def test_round_classification(self):
        """둥근형 얼굴 분류 테스트"""
        shape, confidence = classify_face_shape(self.round_landmarks)
        self.assertEqual(shape, "Round")
        self.assertGreater(confidence, 0.3)

    def test_square_classification(self):
        """사각형 얼굴 분류 테스트"""
        shape, confidence = classify_face_shape(self.square_landmarks)
        self.assertEqual(shape, "Square")
        self.assertGreater(confidence, 0.3)

    def test_heart_classification_not_dominant(self):
        """하트형이 더 이상 기본값이 아님을 확인"""
        # 다양한 랜드마크로 테스트
        test_cases = [self.oval_landmarks, self.round_landmarks, self.square_landmarks, self.diamond_landmarks, self.oblong_landmarks]
        heart_count = 0

        for landmarks in test_cases:
            shape, _ = classify_face_shape(landmarks)
            if shape == "Heart":
                heart_count += 1

        # Heart형이 전체의 50% 미만이어야 함 (이전에는 거의 100%였음)
        self.assertLess(heart_count, len(test_cases) * 0.5)

    def test_diamond_classification(self):
        """다이아몬드형 얼굴 분류 테스트"""
        shape, confidence = classify_face_shape(self.diamond_landmarks)
        self.assertEqual(shape, "Diamond")
        self.assertGreater(confidence, 0.3)

    def test_oblong_classification(self):
        """직사각형 얼굴 분류 테스트"""
        shape, confidence = classify_face_shape(self.oblong_landmarks)
        self.assertEqual(shape, "Oblong")
        self.assertGreater(confidence, 0.3)

    def test_analyze_face_shape_with_image(self):
        """이미지로 얼굴형 분석 테스트"""
        image_bgr = np.zeros((224, 224, 3), dtype=np.uint8)

        with patch('src.analyzers.face_shape.detect_landmarks', return_value=self.oval_landmarks):
            result = analyze_face_shape(image_bgr)

            self.assertEqual(result["status"], "ok")
            self.assertIn(result["shape"], ["Oval", "Round", "Square", "Heart", "Diamond", "Oblong"])
            self.assertIn("confidence", result)
            self.assertIn("metrics", result)

    def test_analyze_face_shape_no_face(self):
        """얼굴이 감지되지 않을 때 테스트"""
        image_bgr = np.zeros((224, 224, 3), dtype=np.uint8)

        with patch('src.analyzers.face_shape.detect_landmarks', return_value=None):
            result = analyze_face_shape(image_bgr)

            self.assertEqual(result["status"], "guardrail")
            self.assertEqual(result["code"], "NO_FACE")

    def test_penalties_calculation(self):
        """패널티 계산 함수 테스트"""
        # 타원형에 이상적인 메트릭
        oval_metrics = {"LWR": 1.4, "FJ": 1.0, "CWF": 1.05, "CWJ": 1.0}
        penalties = _rule_penalties(oval_metrics)

        # Oval의 패널티가 가장 낮아야 함
        min_penalty_shape = min(penalties.items(), key=lambda x: x[1])[0]
        self.assertEqual(min_penalty_shape, "Oval")

    def test_confidence_calculation(self):
        """신뢰도 계산 테스트"""
        metrics = {"LWR": 1.4, "FJ": 1.0, "CWF": 1.05, "CWJ": 1.0}
        result = classify_with_confidence(metrics)

        self.assertIn("shape", result)
        self.assertIn("confidence", result)
        self.assertIn("top2", result)
        self.assertBetween(result["confidence"], 0.0, 1.0)
        self.assertEqual(len(result["top2"]), 2)

    def assertBetween(self, value, min_val, max_val):
        """값이 범위 내에 있는지 확인하는 헬퍼 메서드"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)


if __name__ == "__main__":
    unittest.main()