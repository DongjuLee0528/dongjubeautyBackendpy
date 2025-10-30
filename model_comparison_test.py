#!/usr/bin/env python3
"""
RandomForest vs CNN 모델 성능 비교 테스트
"""

import cv2
import numpy as np
import joblib
import time
from glob import glob
import os

# RandomForest 모델 테스트
def test_random_forest():
    """RandomForest 모델 테스트"""
    print("🌲 RandomForest 모델 테스트")

    try:
        # 모델 로드
        start_time = time.time()
        clf = joblib.load('face_shape_rf_model.pkl')
        le = joblib.load('label_encoder.pkl')
        load_time = time.time() - start_time

        print(f"   로드 시간: {load_time:.3f}초")

        # 테스트 이미지들
        test_images = [
            ("../faceshape-master/published_dataset/heart/img_no_1.jpg", "Heart"),
            ("../faceshape-master/published_dataset/oval/img_no_201.jpg", "Oval"),
            ("../faceshape-master/published_dataset/round/img_no_301.jpg", "Round"),
            ("../faceshape-master/published_dataset/square/img_no_401.jpg", "Square"),
            ("../faceshape-master/published_dataset/oblong/img_no_101.jpg", "Oblong")
        ]

        correct = 0
        total_pred_time = 0

        for img_path, expected in test_images:
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # 특징 추출 및 예측
            start_time = time.time()

            # HOG + Color features (train_large_model.py와 동일)
            img_resized = cv2.resize(img, (96, 96))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            from skimage.feature import hog
            hog_features = hog(gray,
                              orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              block_norm='L2-Hys',
                              visualize=False)

            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

            color_features = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(),
                hist_v.flatten()
            ])

            mean_vals = np.mean(img_resized, axis=(0, 1))
            std_vals = np.std(img_resized, axis=(0, 1))
            stat_features = np.concatenate([mean_vals, std_vals])

            combined_features = np.concatenate([
                hog_features,
                color_features,
                stat_features
            ]).reshape(1, -1)

            # 예측
            pred_idx = clf.predict(combined_features)[0]
            prob = clf.predict_proba(combined_features)[0]
            predicted = le.inverse_transform([pred_idx])[0]
            confidence = np.max(prob)

            pred_time = time.time() - start_time
            total_pred_time += pred_time

            # 결과
            status = "✅" if predicted == expected else "❌"
            if predicted == expected:
                correct += 1

            print(f"   {expected:<7} → {predicted:<7} ({confidence:.1%}) {status} [{pred_time:.3f}초]")

        accuracy = (correct / len(test_images)) * 100
        avg_pred_time = total_pred_time / len(test_images)

        print(f"\n📊 RandomForest 결과:")
        print(f"   정확도: {accuracy:.1f}%")
        print(f"   평균 예측 시간: {avg_pred_time:.3f}초")

        return {
            'accuracy': accuracy,
            'avg_prediction_time': avg_pred_time,
            'model_load_time': load_time
        }

    except Exception as e:
        print(f"❌ RandomForest 테스트 실패: {e}")
        return None

def test_cnn_if_available():
    """CNN 모델 테스트 (가능한 경우)"""
    print("\n🔥 CNN 모델 테스트")

    model_path = "simple_cnn_face_shape_model.h5"
    if not os.path.exists(model_path):
        print("   CNN 모델이 아직 훈련되지 않았습니다.")
        return None

    try:
        import tensorflow as tf

        # 모델 로드
        start_time = time.time()
        model = tf.keras.models.load_model(model_path)
        le = joblib.load("simple_cnn_face_shape_model_label_encoder.pkl")
        load_time = time.time() - start_time

        print(f"   로드 시간: {load_time:.3f}초")

        # 테스트 이미지들
        test_images = [
            ("../faceshape-master/published_dataset/heart/img_no_1.jpg", "Heart"),
            ("../faceshape-master/published_dataset/oval/img_no_201.jpg", "Oval"),
            ("../faceshape-master/published_dataset/round/img_no_301.jpg", "Round"),
            ("../faceshape-master/published_dataset/square/img_no_401.jpg", "Square"),
            ("../faceshape-master/published_dataset/oblong/img_no_101.jpg", "Oblong")
        ]

        correct = 0
        total_pred_time = 0

        for img_path, expected in test_images:
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # 전처리 및 예측
            start_time = time.time()

            # CNN 전처리
            img_resized = cv2.resize(img, (128, 128))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            # 예측
            predictions = model.predict(img_batch, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted = le.inverse_transform([pred_idx])[0]

            pred_time = time.time() - start_time
            total_pred_time += pred_time

            # 결과
            status = "✅" if predicted == expected else "❌"
            if predicted == expected:
                correct += 1

            print(f"   {expected:<7} → {predicted:<7} ({confidence:.1%}) {status} [{pred_time:.3f}초]")

        accuracy = (correct / len(test_images)) * 100
        avg_pred_time = total_pred_time / len(test_images)

        print(f"\n📊 CNN 결과:")
        print(f"   정확도: {accuracy:.1f}%")
        print(f"   평균 예측 시간: {avg_pred_time:.3f}초")

        return {
            'accuracy': accuracy,
            'avg_prediction_time': avg_pred_time,
            'model_load_time': load_time
        }

    except Exception as e:
        print(f"❌ CNN 테스트 실패: {e}")
        return None

def main():
    """메인 비교 함수"""
    print("🔍 얼굴형 분류 모델 성능 비교")
    print("=" * 50)

    # RandomForest 테스트
    rf_results = test_random_forest()

    # CNN 테스트
    cnn_results = test_cnn_if_available()

    # 결과 비교
    print("\n" + "=" * 50)
    print("📈 최종 비교 결과")
    print("=" * 50)

    if rf_results:
        print(f"🌲 RandomForest:")
        print(f"   정확도: {rf_results['accuracy']:.1f}%")
        print(f"   예측 속도: {rf_results['avg_prediction_time']:.3f}초")
        print(f"   로드 시간: {rf_results['model_load_time']:.3f}초")

    if cnn_results:
        print(f"\n🔥 CNN:")
        print(f"   정확도: {cnn_results['accuracy']:.1f}%")
        print(f"   예측 속도: {cnn_results['avg_prediction_time']:.3f}초")
        print(f"   로드 시간: {cnn_results['model_load_time']:.3f}초")

        # 승자 결정
        if rf_results and cnn_results:
            print(f"\n🏆 승자:")
            if cnn_results['accuracy'] > rf_results['accuracy']:
                print(f"   정확도: CNN (+ {cnn_results['accuracy'] - rf_results['accuracy']:.1f}%p)")
            elif rf_results['accuracy'] > cnn_results['accuracy']:
                print(f"   정확도: RandomForest (+ {rf_results['accuracy'] - cnn_results['accuracy']:.1f}%p)")
            else:
                print(f"   정확도: 동점")

            if rf_results['avg_prediction_time'] < cnn_results['avg_prediction_time']:
                print(f"   속도: RandomForest ({rf_results['avg_prediction_time']:.3f}초 vs {cnn_results['avg_prediction_time']:.3f}초)")
            else:
                print(f"   속도: CNN ({cnn_results['avg_prediction_time']:.3f}초 vs {rf_results['avg_prediction_time']:.3f}초)")
    else:
        print(f"\n⏳ CNN 모델이 아직 훈련 중입니다.")
        print(f"   현재 RandomForest 모델만 사용 가능합니다.")

if __name__ == "__main__":
    main()