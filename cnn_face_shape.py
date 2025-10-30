#!/usr/bin/env python3
"""
VGG-16 + VGGFace 기반 CNN 얼굴형 분류 모델
92.7% 정확도를 목표로 함
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from glob import glob

class CNNFaceShapeClassifier:
    def __init__(self, model_path=None):
        self.classes = ["Heart", "Oblong", "Oval", "Round", "Square"]
        self.model = None
        self.label_encoder = None
        self.img_size = (224, 224)  # VGG-16 표준 입력 크기

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def preprocess_image(self, img):
        """이미지 전처리 (VGG-16 형식에 맞게)"""
        if img is None:
            return None

        # 224x224로 리사이즈
        img_resized = cv2.resize(img, self.img_size)

        # BGR to RGB 변환
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 정규화 (0-1 범위)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        return img_normalized

    def load_data(self, data_dir="../faceshape-master/augmented_dataset"):
        """확장된 데이터셋 로드"""
        print("📂 CNN용 데이터셋 로딩 중...")

        images = []
        labels = []

        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name.lower())
            if not os.path.exists(class_dir):
                continue

            class_images = glob(os.path.join(class_dir, "*.jpg"))
            print(f"   {class_name}: {len(class_images)}개")

            for img_path in class_images:
                img = cv2.imread(img_path)
                if img is not None:
                    processed_img = self.preprocess_image(img)
                    if processed_img is not None:
                        images.append(processed_img)
                        labels.append(class_name)

        X = np.array(images)
        y = np.array(labels)

        print(f"✅ 총 로드된 데이터: {len(X)}개")
        print(f"   이미지 크기: {X.shape}")

        return X, y

    def create_vgg_model(self):
        """VGG-16 + VGGFace 기반 모델 생성"""
        print("🏗️ VGG-16 + VGGFace CNN 모델 생성 중...")

        # VGG-16 베이스 모델 (ImageNet weights 사용, VGGFace weights는 별도 다운로드 필요)
        base_model = VGG16(
            input_shape=(224, 224, 3),
            include_top=False,  # 마지막 분류 레이어 제외
            weights='imagenet'  # ImageNet 사전 훈련 가중치 사용
        )

        # 베이스 모델 레이어 고정 (Transfer Learning)
        for layer in base_model.layers:
            layer.trainable = False

        # 새로운 분류 헤드 추가
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(5, activation='softmax')(x)  # 5개 클래스

        # 모델 생성
        model = Model(inputs=base_model.input, outputs=predictions)

        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"📊 모델 구조:")
        print(f"   총 파라미터: {model.count_params():,}")
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        print(f"   훈련 가능 파라미터: {trainable_params:,}")

        return model

    def train_model(self, X, y, epochs=30, batch_size=32):
        """CNN 모델 훈련"""
        print("\n🤖 CNN 모델 훈련 시작...")

        # 레이블 인코딩
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=5)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"   훈련 데이터: {len(X_train)}개")
        print(f"   테스트 데이터: {len(X_test)}개")

        # 모델 생성
        self.model = self.create_vgg_model()

        # 데이터 증강
        datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        datagen.fit(X_train)

        # 조기 종료 및 체크포인트 콜백
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]

        # 모델 훈련
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # 최종 평가
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n🎯 최종 테스트 정확도: {test_accuracy:.1%}")

        # 상세 분류 리포트
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        print(f"\n📊 상세 성능 리포트:")
        print(classification_report(
            y_test_classes, y_pred_classes,
            target_names=self.label_encoder.classes_
        ))

        return history, test_accuracy

    def save_model(self, model_path="cnn_face_shape_model"):
        """모델 저장"""
        if self.model is None:
            print("❌ 저장할 모델이 없습니다.")
            return

        # TensorFlow 모델 저장
        self.model.save(f"{model_path}.h5")

        # 라벨 인코더 저장
        joblib.dump(self.label_encoder, f"{model_path}_label_encoder.pkl")

        print(f"✅ CNN 모델 저장 완료: {model_path}.h5")

    def load_model(self, model_path):
        """저장된 모델 로드"""
        try:
            self.model = tf.keras.models.load_model(f"{model_path}.h5")
            self.label_encoder = joblib.load(f"{model_path}_label_encoder.pkl")
            print(f"✅ CNN 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")

    def predict(self, img):
        """단일 이미지 예측"""
        if self.model is None or self.label_encoder is None:
            return None

        processed_img = self.preprocess_image(img)
        if processed_img is None:
            return None

        # 배치 차원 추가
        img_batch = np.expand_dims(processed_img, axis=0)

        # 예측
        predictions = self.model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        predicted_class = self.label_encoder.inverse_transform([pred_idx])[0]

        return {
            'face_shape': predicted_class,
            'confidence': float(confidence),
            'all_predictions': {
                cls: float(prob) for cls, prob in zip(
                    self.label_encoder.classes_, predictions[0]
                )
            }
        }

def main():
    """메인 실행 함수"""
    print("🚀 VGG-16 + VGGFace CNN 얼굴형 분류 모델 훈련")

    # 분류기 생성
    classifier = CNNFaceShapeClassifier()

    # 데이터 로드
    X, y = classifier.load_data()

    # 모델 훈련
    history, accuracy = classifier.train_model(X, y, epochs=20)

    # 모델 저장
    classifier.save_model("cnn_face_shape_model")

    print(f"\n🎉 CNN 모델 훈련 완료!")
    print(f"   최종 정확도: {accuracy:.1%}")
    print(f"   목표 달성: {'✅' if accuracy >= 0.85 else '⚠️'} (목표: 85%+)")

if __name__ == "__main__":
    main()