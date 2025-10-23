import unittest
from unittest.mock import patch

import numpy as np

from src.analyzers import PersonalColorAnalyzer, classify_from_metrics, classify_with_confidence


class PersonalColorAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = PersonalColorAnalyzer()

    def test_classify_from_metrics_warm_vs_cool(self) -> None:
        warm_metrics = {"L": 60.0, "a": 25.0, "b": 30.0, "ITA": 35.0}
        cool_metrics = {"L": 40.0, "a": -5.0, "b": -10.0, "ITA": 5.0}

        season, tone = classify_from_metrics(warm_metrics)
        self.assertEqual(tone, "Warm")
        self.assertIn(season, {"Spring", "Autumn"})

        season_cool, tone_cool = classify_from_metrics(cool_metrics)
        self.assertEqual(tone_cool, "Cool")
        self.assertIn(season_cool, {"Summer", "Winter"})

    def test_analyze_returns_guardrail_when_no_face(self) -> None:
        image_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("src.analyzers.personal_color.detect_landmarks", return_value=None):
            result = self.analyzer.analyze(image_bgr)

        self.assertEqual(result.get("status"), "guardrail")
        self.assertEqual(result.get("code"), "NO_FACE")

    def test_analyze_guardrail_on_low_quality_roi(self) -> None:
        image_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_landmarks = np.zeros((468, 2), dtype=np.float32)
        roi_metrics = {
            "left_cheek": {"mean_l": 5.0, "variance": 1.0, "pixels": 500.0, "reason": "ROI_TOO_DARK"}
        }

        with patch("src.analyzers.personal_color.detect_landmarks", return_value=fake_landmarks), patch(
            "src.analyzers.personal_color.extract_skin_rois",
            return_value=({}, roi_metrics),
        ):
            result = self.analyzer.analyze(image_bgr)

        self.assertEqual(result.get("status"), "guardrail")
        self.assertEqual(result.get("code"), "LOW_QUALITY")
        self.assertIn("landmarks", result)
        self.assertIsNone(result.get("metrics"))

    def test_analyze_successful_flow(self) -> None:
        image_bgr = np.full((128, 128, 3), (180, 170, 160), dtype=np.uint8)
        fake_landmarks = np.zeros((468, 2), dtype=np.float32)
        roi_patch = np.full((32, 32, 3), (200, 180, 160), dtype=np.uint8)
        roi_metrics = {
            "left_cheek": {"mean_l": 70.0, "variance": 15.0, "pixels": 1024.0},
            "right_cheek": {"mean_l": 68.0, "variance": 14.0, "pixels": 1024.0},
        }
        rois = {"left_cheek": roi_patch, "right_cheek": roi_patch}

        with patch("src.analyzers.personal_color.detect_landmarks", return_value=fake_landmarks), patch(
            "src.analyzers.personal_color.extract_skin_rois",
            return_value=(rois, roi_metrics),
        ):
            result = self.analyzer.analyze(image_bgr)

        self.assertEqual(result.get("status"), "ok")
        self.assertIn(result.get("season"), DEFAULT_SEASONS)
        self.assertIn(result.get("tone"), {"Warm", "Cool"})
        self.assertIn("metrics", result)

    def test_classify_with_confidence_warm_spring(self) -> None:
        """밝은 warm 톤 테스트 (Spring 예상)"""
        warm_bright_metrics = {"L": 70.0, "a": 15.0, "b": 20.0, "ITA": 35.0}
        result = classify_with_confidence(warm_bright_metrics)

        self.assertEqual(result["season"], "Spring")
        self.assertEqual(result["tone"], "Warm")
        self.assertGreaterEqual(result["confidence"], 0.5)
        self.assertIn("tone_confidence", result)
        self.assertIn("season_confidence", result)
        self.assertEqual(len(result["top2"]), 2)
        self.assertIsInstance(result["is_mixed"], bool)

    def test_classify_with_confidence_cool_winter(self) -> None:
        """어두운 cool 톤 테스트 (Winter 예상)"""
        cool_dark_metrics = {"L": 40.0, "a": -10.0, "b": -15.0, "ITA": 5.0}
        result = classify_with_confidence(cool_dark_metrics)

        self.assertEqual(result["season"], "Winter")
        self.assertEqual(result["tone"], "Cool")
        self.assertGreaterEqual(result["confidence"], 0.5)

    def test_classify_with_confidence_mixed_type(self) -> None:
        """혼합형 테스트 (경계값)"""
        mixed_metrics = {"L": 55.0, "a": 2.0, "b": 5.0, "ITA": 20.0}
        result = classify_with_confidence(mixed_metrics)

        # 혼합형이거나 낮은 신뢰도일 것으로 예상
        if result["is_mixed"]:
            self.assertIsNotNone(result["mixed_type"])
            self.assertIn("-", result["mixed_type"])

        self.assertLessEqual(result["confidence"], 1.0)
        self.assertGreaterEqual(result["confidence"], 0.0)

    def test_enhanced_analyzer_response(self) -> None:
        """개선된 analyzer 응답 형식 테스트"""
        image_bgr = np.full((128, 128, 3), (180, 170, 160), dtype=np.uint8)
        fake_landmarks = np.zeros((468, 2), dtype=np.float32)
        roi_patch = np.full((32, 32, 3), (200, 180, 160), dtype=np.uint8)
        roi_metrics = {
            "left_cheek": {"mean_l": 70.0, "variance": 15.0, "pixels": 1024.0},
            "right_cheek": {"mean_l": 68.0, "variance": 14.0, "pixels": 1024.0},
        }
        rois = {"left_cheek": roi_patch, "right_cheek": roi_patch}

        with patch("src.analyzers.personal_color.detect_landmarks", return_value=fake_landmarks), patch(
            "src.analyzers.personal_color.extract_skin_rois",
            return_value=(rois, roi_metrics),
        ):
            result = self.analyzer.analyze(image_bgr)

        # 새로운 응답 필드들 확인
        self.assertEqual(result.get("status"), "ok")
        self.assertIn("confidence", result)
        self.assertIn("tone_confidence", result)
        self.assertIn("season_confidence", result)
        self.assertIn("top2", result)
        self.assertIn("is_mixed", result)

        # top2 구조 확인
        top2 = result["top2"]
        self.assertEqual(len(top2), 2)
        for item in top2:
            self.assertIn("label", item)
            self.assertIn("prob", item)
            self.assertIsInstance(item["prob"], float)

    def test_roi_landmarks_updated(self) -> None:
        """업데이트된 ROI 랜드마크 인덱스 테스트"""
        from src.analyzers.skin_roi import ROI_LANDMARKS

        # 366 인덱스가 제거되었는지 확인
        for region, indices in ROI_LANDMARKS.items():
            self.assertNotIn(366, indices, f"{region}에서 잘못된 인덱스 366 발견")

        # 모든 인덱스가 유효한 범위 내에 있는지 확인 (0-467)
        for region, indices in ROI_LANDMARKS.items():
            for idx in indices:
                self.assertGreaterEqual(idx, 0, f"{region}의 인덱스 {idx}가 음수")
                self.assertLess(idx, 468, f"{region}의 인덱스 {idx}가 467을 초과")


DEFAULT_SEASONS = {"Spring", "Summer", "Autumn", "Winter"}


if __name__ == "__main__":
    unittest.main()
