#!/usr/bin/env python3
"""
확장된 데이터셋(3000개)으로 ML 모델 학습
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

class LargeFaceShapeTrainer:
    def __init__(self, data_dir="../faceshape-master/augmented_dataset"):
        self.data_dir = data_dir
        self.classes = ["heart", "oblong", "oval", "round", "square"]
        self.features = []
        self.labels = []

    def extract_features(self, img_path):
        """효율적인 특징 추출 (3000개 이미지 처리 고려)"""
        img = cv2.imread(img_path)
        if img is None:
            return None

        # 이미지 전처리
        img = cv2.resize(img, (96, 96))  # 적당한 크기로
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. HOG 특징 (효율적인 파라미터)
        hog_features = hog(gray,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys',
                          visualize=False)

        # 2. 색상 히스토그램
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

        color_features = np.concatenate([
            hist_h.flatten(),
            hist_s.flatten(),
            hist_v.flatten()
        ])

        # 3. 간단한 통계적 특징
        mean_vals = np.mean(img, axis=(0, 1))
        std_vals = np.std(img, axis=(0, 1))
        stat_features = np.concatenate([mean_vals, std_vals])

        # 모든 특징 결합
        combined_features = np.concatenate([
            hog_features,
            color_features,
            stat_features
        ])

        return combined_features

    def load_data(self):
        """확장된 데이터 로드"""
        print("📂 확장된 데이터셋 로딩 중...")

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
        """대용량 데이터로 모델 학습"""
        print("\n🤖 대용량 데이터셋 모델 학습 시작...")

        # 레이블 인코딩
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.labels)

        # 데이터 분할 (더 많은 테스트 데이터)
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )

        print(f"   훈련 데이터: {len(X_train)}개")
        print(f"   테스트 데이터: {len(X_test)}개")

        # 모델 학습 (대용량 데이터에 최적화)
        clf = RandomForestClassifier(
            n_estimators=100,  # 효율성을 위해 적당히
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1  # 진행 상황 표시
        )

        print("   모델 학습 중...")
        clf.fit(X_train, y_train)

        # 교차 검증 (3-fold로 시간 절약)
        print("   교차 검증 중...")
        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, verbose=1)
        print(f"   교차 검증 점수: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # 테스트 성능
        test_score = clf.score(X_test, y_test)
        print(f"   테스트 정확도: {test_score:.3f}")

        # 상세 분류 리포트
        y_pred = clf.predict(X_test)
        print(f"\n📊 상세 성능 리포트:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📈 혼동 행렬:")
        print("    ", " ".join(f"{cls[:4]:>5}" for cls in le.classes_))
        for i, row in enumerate(cm):
            print(f"{le.classes_[i][:4]:>4} {' '.join(f'{val:>5}' for val in row)}")

        # 모델 저장
        joblib.dump(clf, 'face_shape_large_model.pkl')
        joblib.dump(le, 'label_encoder_large.pkl')

        # 설정 저장
        config = {
            'model_version': 'large_3000',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'test_accuracy': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'feature_dimension': self.features.shape[1],
            'classes': self.classes,
            'data_augmentation': True
        }

        with open('train_config_large.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✅ 대용량 모델 저장 완료!")
        print(f"   파일: face_shape_large_model.pkl")
        print(f"   정확도: {test_score:.1%}")

        return clf, le, test_score

def main():
    trainer = LargeFaceShapeTrainer()
    trainer.load_data()
    clf, le, accuracy = trainer.train_model()

    print(f"\n🎉 대용량 모델 학습 완료!")
    print(f"   데이터 크기: 500개 → 3000개 (6배 증가)")
    print(f"   최종 정확도: {accuracy:.1%}")

if __name__ == "__main__":
    main()