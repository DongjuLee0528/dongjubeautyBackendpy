#!/usr/bin/env python3
"""
정리된 데이터셋으로 ML 모델 재학습
"""

import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog
import joblib
import json
from glob import glob

class ImprovedFaceShapeTrainer:
    def __init__(self, data_dir="../faceshape-master/cleaned_dataset"):
        self.data_dir = data_dir
        self.classes = ["heart", "oblong", "oval", "round", "square"]
        self.features = []
        self.labels = []

    def extract_features(self, img_path):
        """향상된 특징 추출"""
        img = cv2.imread(img_path)
        if img is None:
            return None

        # 이미지 전처리
        img = cv2.resize(img, (128, 128))  # 더 큰 크기로
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. HOG 특징 (더 세밀한 파라미터)
        hog_features = hog(gray,
                          orientations=12,  # 증가
                          pixels_per_cell=(8, 8),  # 감소
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys',
                          visualize=False)

        # 2. 색상 히스토그램 (더 세밀하게)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])  # 증가
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])  # 증가
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])  # 증가

        color_features = np.concatenate([
            hist_h.flatten(),
            hist_s.flatten(),
            hist_v.flatten()
        ])

        # 3. 텍스처 특징 (LBP 추가)
        def lbp_features(image, radius=1, n_points=8):
            """간단한 LBP 특징"""
            h, w = image.shape
            lbp = np.zeros_like(image)

            for i in range(radius, h-radius):
                for j in range(radius, w-radius):
                    center = image[i, j]
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if 0 <= x < h and 0 <= y < w:
                            if image[x, y] >= center:
                                code |= (1 << k)
                    lbp[i, j] = code

            hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2))
            return hist

        lbp_hist = lbp_features(gray)

        # 모든 특징 결합
        combined_features = np.concatenate([
            hog_features,
            color_features,
            lbp_hist
        ])

        return combined_features

    def load_data(self):
        """정리된 데이터 로드"""
        print("📂 정리된 데이터셋 로딩 중...")

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            images = glob(os.path.join(class_dir, "*.jpg"))
            print(f"   {class_name}: {len(images)}개")

            for img_path in images:
                features = self.extract_features(img_path)
                if features is not None:
                    self.features.append(features)
                    self.labels.append(class_name.capitalize())

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        print(f"✅ 총 로드된 데이터: {len(self.features)}개")
        print(f"   특징 차원: {self.features.shape[1]}")

    def train_model(self):
        """모델 학습 및 평가"""
        print("\n🤖 모델 학습 시작...")

        # 레이블 인코딩
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.labels)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"   훈련 데이터: {len(X_train)}개")
        print(f"   테스트 데이터: {len(X_test)}개")

        # 모델 학습 (하이퍼파라미터 튜닝)
        clf = RandomForestClassifier(
            n_estimators=200,  # 증가
            max_depth=20,      # 증가
            min_samples_split=3,  # 감소
            min_samples_leaf=1,   # 감소
            random_state=42,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)

        # 교차 검증
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        print(f"   교차 검증 점수: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # 테스트 성능
        test_score = clf.score(X_test, y_test)
        print(f"   테스트 정확도: {test_score:.3f}")

        # 상세 분류 리포트
        y_pred = clf.predict(X_test)
        print(f"\n📊 상세 성능 리포트:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # 모델 저장
        joblib.dump(clf, 'face_shape_rf_model_v2.pkl')
        joblib.dump(le, 'label_encoder_v2.pkl')

        # 설정 저장
        config = {
            'model_version': '2.0',
            'training_samples': len(X_train),
            'test_accuracy': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'feature_dimension': self.features.shape[1],
            'classes': self.classes
        }

        with open('train_config_v2.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ 개선된 모델 저장 완료!")
        print(f"   파일: face_shape_rf_model_v2.pkl")
        print(f"   정확도: {test_score:.1%}")

        return clf, le, test_score

def main():
    trainer = ImprovedFaceShapeTrainer()
    trainer.load_data()
    clf, le, accuracy = trainer.train_model()

    print(f"\n🎉 모델 학습 완료!")
    print(f"   정확도 개선: 47% → {accuracy:.1%}")

if __name__ == "__main__":
    main()