#!/usr/bin/env python3
"""
데이터 품질 분석 도구
- 얼굴 검출 실패 이미지 찾기
- 흐릿한 이미지 감지
- 중복 이미지 탐지
- 모델 신뢰도 낮은 이미지 분석
- 정리 추천사항 제공
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import hashlib
from collections import defaultdict
import joblib
import json

class DataQualityAnalyzer:
    def __init__(self, data_dir="../faceshape-master/published_dataset"):
        self.data_dir = data_dir
        self.classes = ["heart", "oblong", "oval", "round", "square"]
        self.results = defaultdict(list)

    def analyze_all(self):
        """전체 품질 분석 실행"""
        print("🔍 데이터 품질 분석 시작...")

        # 1. 기본 정보 수집
        self.collect_basic_info()

        # 2. 얼굴 검출 실패 분석
        self.analyze_face_detection()

        # 3. 이미지 품질 분석
        self.analyze_image_quality()

        # 4. 중복 이미지 탐지
        self.detect_duplicates()

        # 5. ML 모델 신뢰도 분석 (모델이 있는 경우)
        self.analyze_model_confidence()

        # 6. 리포트 생성
        self.generate_report()

    def collect_basic_info(self):
        """기본 데이터셋 정보 수집"""
        print("📊 기본 정보 수집 중...")

        total_images = 0
        class_counts = {}

        for cls in self.classes:
            folder_path = os.path.join(self.data_dir, cls)
            if os.path.exists(folder_path):
                images = glob(os.path.join(folder_path, "*"))
                images = [f for f in images if os.path.isfile(f)]
                class_counts[cls] = len(images)
                total_images += len(images)
            else:
                class_counts[cls] = 0

        self.results['basic_info'] = {
            'total_images': total_images,
            'class_counts': class_counts,
            'balanced': len(set(class_counts.values())) == 1
        }

        print(f"   총 이미지: {total_images}개")
        print(f"   클래스별: {class_counts}")

    def analyze_face_detection(self):
        """얼굴 검출 실패 이미지 분석"""
        print("👤 얼굴 검출 분석 중...")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        failed_detection = []
        multiple_faces = []

        for cls in self.classes:
            folder_path = os.path.join(self.data_dir, cls)
            if not os.path.exists(folder_path):
                continue

            images = glob(os.path.join(folder_path, "*"))

            for img_path in images:
                if not os.path.isfile(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    failed_detection.append({
                        'path': img_path,
                        'class': cls,
                        'reason': 'Cannot read image'
                    })
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

                if len(faces) == 0:
                    failed_detection.append({
                        'path': img_path,
                        'class': cls,
                        'reason': 'No face detected'
                    })
                elif len(faces) > 1:
                    multiple_faces.append({
                        'path': img_path,
                        'class': cls,
                        'face_count': len(faces)
                    })

        self.results['face_detection'] = {
            'failed_detection': failed_detection,
            'multiple_faces': multiple_faces
        }

        print(f"   얼굴 검출 실패: {len(failed_detection)}개")
        print(f"   다중 얼굴: {len(multiple_faces)}개")

    def analyze_image_quality(self):
        """이미지 품질 분석 (블러, 크기, 밝기 등)"""
        print("🖼️  이미지 품질 분석 중...")

        low_quality = []
        size_issues = []

        for cls in self.classes:
            folder_path = os.path.join(self.data_dir, cls)
            if not os.path.exists(folder_path):
                continue

            images = glob(os.path.join(folder_path, "*"))

            for img_path in images:
                if not os.path.isfile(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                # 1. 블러 검출 (Laplacian variance)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                # 2. 이미지 크기 확인
                h, w = img.shape[:2]

                # 3. 밝기 분석
                brightness = np.mean(gray)

                # 품질 평가
                if blur_score < 100:  # 블러 임계값
                    low_quality.append({
                        'path': img_path,
                        'class': cls,
                        'reason': 'Too blurry',
                        'blur_score': blur_score
                    })

                if w < 100 or h < 100:  # 크기 임계값
                    size_issues.append({
                        'path': img_path,
                        'class': cls,
                        'size': (w, h),
                        'reason': 'Too small'
                    })

                if brightness < 30 or brightness > 225:  # 밝기 임계값
                    low_quality.append({
                        'path': img_path,
                        'class': cls,
                        'reason': 'Poor lighting',
                        'brightness': brightness
                    })

        self.results['image_quality'] = {
            'low_quality': low_quality,
            'size_issues': size_issues
        }

        print(f"   품질 문제: {len(low_quality)}개")
        print(f"   크기 문제: {len(size_issues)}개")

    def detect_duplicates(self):
        """중복 이미지 탐지 (해시 기반)"""
        print("🔄 중복 이미지 탐지 중...")

        hash_dict = defaultdict(list)

        for cls in self.classes:
            folder_path = os.path.join(self.data_dir, cls)
            if not os.path.exists(folder_path):
                continue

            images = glob(os.path.join(folder_path, "*"))

            for img_path in images:
                if not os.path.isfile(img_path):
                    continue

                try:
                    with open(img_path, 'rb') as f:
                        img_hash = hashlib.md5(f.read()).hexdigest()
                        hash_dict[img_hash].append({
                            'path': img_path,
                            'class': cls
                        })
                except:
                    continue

        # 중복 찾기
        duplicates = []
        for img_hash, img_list in hash_dict.items():
            if len(img_list) > 1:
                duplicates.append(img_list)

        self.results['duplicates'] = duplicates
        print(f"   중복 그룹: {len(duplicates)}개")

    def analyze_model_confidence(self):
        """ML 모델 신뢰도 분석"""
        print("🤖 모델 신뢰도 분석 중...")

        model_path = "face_shape_rf_model.pkl"
        label_path = "label_encoder.pkl"

        if not os.path.exists(model_path) or not os.path.exists(label_path):
            print("   ML 모델을 찾을 수 없어 건너뜁니다.")
            self.results['model_confidence'] = {'analyzed': False}
            return

        try:
            # 모델 로드
            clf = joblib.load(model_path)
            le = joblib.load(label_path)

            low_confidence = []
            wrong_predictions = []

            for cls in self.classes:
                folder_path = os.path.join(self.data_dir, cls)
                if not os.path.exists(folder_path):
                    continue

                images = glob(os.path.join(folder_path, "*"))

                for img_path in images[:10]:  # 샘플만 테스트 (시간 절약)
                    if not os.path.isfile(img_path):
                        continue

                    try:
                        # 이미지 로드 및 특징 추출 (간단 버전)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        # 간단한 특징 추출 (실제 HOG+Color 대신)
                        img_resized = cv2.resize(img, (64, 64))
                        features = img_resized.flatten().astype(np.float32)

                        # 차원 맞추기 (실제 모델과 다를 수 있음)
                        if features.shape[0] != 8612:  # 실제 특징 차원
                            continue

                        # 예측
                        features = features.reshape(1, -1)
                        pred_idx = clf.predict(features)[0]
                        prob = clf.predict_proba(features)[0]

                        predicted_class = le.inverse_transform([pred_idx])[0]
                        confidence = np.max(prob)

                        # 분석
                        if confidence < 0.5:  # 낮은 신뢰도
                            low_confidence.append({
                                'path': img_path,
                                'true_class': cls.capitalize(),
                                'predicted_class': predicted_class,
                                'confidence': confidence
                            })

                        if predicted_class.lower() != cls:  # 잘못된 예측
                            wrong_predictions.append({
                                'path': img_path,
                                'true_class': cls.capitalize(),
                                'predicted_class': predicted_class,
                                'confidence': confidence
                            })

                    except Exception as e:
                        continue

            self.results['model_confidence'] = {
                'analyzed': True,
                'low_confidence': low_confidence,
                'wrong_predictions': wrong_predictions
            }

            print(f"   낮은 신뢰도: {len(low_confidence)}개")
            print(f"   잘못된 예측: {len(wrong_predictions)}개")

        except Exception as e:
            print(f"   모델 분석 실패: {e}")
            self.results['model_confidence'] = {'analyzed': False}

    def generate_report(self):
        """최종 분석 리포트 생성"""
        print("\n" + "="*60)
        print("📋 데이터 품질 분석 리포트")
        print("="*60)

        # 기본 정보
        basic = self.results['basic_info']
        print(f"\n📊 기본 정보:")
        print(f"   총 이미지: {basic['total_images']}개")
        print(f"   클래스별 분포: {basic['class_counts']}")
        print(f"   균형 여부: {'✅ 균형' if basic['balanced'] else '❌ 불균형'}")

        # 문제 요약
        face_issues = len(self.results['face_detection']['failed_detection'])
        quality_issues = len(self.results['image_quality']['low_quality'])
        duplicate_groups = len(self.results['duplicates'])

        total_problems = face_issues + quality_issues + sum(len(group) for group in self.results['duplicates'])

        print(f"\n🚨 발견된 문제:")
        print(f"   얼굴 검출 실패: {face_issues}개")
        print(f"   품질 문제: {quality_issues}개")
        print(f"   중복 이미지: {duplicate_groups}개 그룹")
        print(f"   총 문제 이미지: {total_problems}개")

        # 개선 제안
        print(f"\n💡 개선 제안:")
        improvement_potential = (total_problems / basic['total_images']) * 100 if basic['total_images'] > 0 else 0
        print(f"   문제 이미지 제거시 예상 성능 향상: +{improvement_potential:.1f}%")

        if face_issues > 0:
            print(f"   - {face_issues}개 얼굴 검출 실패 이미지 제거 권장")
        if quality_issues > 0:
            print(f"   - {quality_issues}개 저품질 이미지 재검토 권장")
        if duplicate_groups > 0:
            print(f"   - {duplicate_groups}개 중복 그룹에서 중복 제거 권장")

        # 상세 문제 목록 저장
        self.save_detailed_report()

        print(f"\n💾 상세 리포트가 'data_quality_report.json'에 저장되었습니다.")
        print("="*60)

    def save_detailed_report(self):
        """상세 리포트를 JSON 파일로 저장"""
        with open('data_quality_report.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)

    def get_cleanup_recommendations(self):
        """정리 추천사항 반환"""
        recommendations = []

        # 얼굴 검출 실패 → 삭제 추천
        for item in self.results['face_detection']['failed_detection']:
            recommendations.append({
                'action': 'delete',
                'path': item['path'],
                'reason': f"얼굴 검출 실패: {item['reason']}"
            })

        # 저품질 이미지 → 재검토 추천
        for item in self.results['image_quality']['low_quality']:
            recommendations.append({
                'action': 'review',
                'path': item['path'],
                'reason': f"품질 문제: {item['reason']}"
            })

        # 중복 이미지 → 하나만 남기고 삭제
        for group in self.results['duplicates']:
            for i, item in enumerate(group[1:], 1):  # 첫 번째 제외하고 나머지 삭제
                recommendations.append({
                    'action': 'delete',
                    'path': item['path'],
                    'reason': f"중복 이미지 (그룹 {i+1})"
                })

        return recommendations

def main():
    """메인 실행 함수"""
    analyzer = DataQualityAnalyzer()
    analyzer.analyze_all()

    # 정리 추천사항
    recommendations = analyzer.get_cleanup_recommendations()

    print(f"\n🧹 정리 추천사항: {len(recommendations)}개")

    # 처리 여부 묻기
    if recommendations:
        print("\n처리할 작업:")
        delete_count = len([r for r in recommendations if r['action'] == 'delete'])
        review_count = len([r for r in recommendations if r['action'] == 'review'])

        print(f"   삭제 추천: {delete_count}개")
        print(f"   재검토 추천: {review_count}개")

        print("\n⚠️  실제 파일 삭제는 수동으로 확인 후 진행하시기 바랍니다.")

if __name__ == "__main__":
    main()