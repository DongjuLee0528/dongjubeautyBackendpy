#!/usr/bin/env python3
"""
데이터 증강을 통한 데이터셋 확장
- 기존 500개 이미지를 2000-3000개로 확장
- 회전, 플립, 밝기 조정, 노이즈 추가 등
"""

import os
import cv2
import numpy as np
from glob import glob
import random

class DataAugmentor:
    def __init__(self, source_dir="../faceshape-master/published_dataset",
                 target_dir="../faceshape-master/augmented_dataset"):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.classes = ["heart", "oblong", "oval", "round", "square"]

    def augment_image(self, img, augment_type):
        """이미지 증강 함수"""
        h, w = img.shape[:2]

        if augment_type == 'flip':
            # 좌우 반전
            return cv2.flip(img, 1)

        elif augment_type == 'rotate_small':
            # 작은 각도 회전 (-10도 ~ +10도)
            angle = random.uniform(-10, 10)
            center = (w//2, h//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, matrix, (w, h))

        elif augment_type == 'brightness':
            # 밝기 조정
            factor = random.uniform(0.7, 1.3)
            return cv2.convertScaleAbs(img, alpha=factor, beta=0)

        elif augment_type == 'contrast':
            # 대비 조정
            factor = random.uniform(0.8, 1.2)
            return cv2.convertScaleAbs(img, alpha=factor, beta=0)

        elif augment_type == 'noise':
            # 노이즈 추가
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            return cv2.add(img, noise)

        elif augment_type == 'blur':
            # 약간의 블러
            return cv2.GaussianBlur(img, (3, 3), 0)

        elif augment_type == 'zoom':
            # 줌 인/아웃
            factor = random.uniform(0.9, 1.1)
            center_x, center_y = w//2, h//2
            new_w, new_h = int(w * factor), int(h * factor)

            if factor > 1:  # 줌 인
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                resized = cv2.resize(img, (new_w, new_h))
                return resized[start_y:start_y+h, start_x:start_x+w]
            else:  # 줌 아웃
                resized = cv2.resize(img, (new_w, new_h))
                result = np.zeros_like(img)
                start_x = (w - new_w) // 2
                start_y = (h - new_h) // 2
                result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
                return result

        return img

    def augment_dataset(self, target_per_class=600):
        """데이터셋 증강 실행"""
        print(f"🔄 데이터 증강 시작...")
        print(f"   목표: 클래스당 {target_per_class}개 이미지")

        # 타겟 디렉토리 생성
        if os.path.exists(self.target_dir):
            import shutil
            shutil.rmtree(self.target_dir)

        os.makedirs(self.target_dir)
        for cls in self.classes:
            os.makedirs(os.path.join(self.target_dir, cls))

        augment_types = ['flip', 'rotate_small', 'brightness', 'contrast', 'noise', 'blur', 'zoom']

        total_created = 0

        for cls in self.classes:
            source_class_dir = os.path.join(self.source_dir, cls)
            target_class_dir = os.path.join(self.target_dir, cls)

            if not os.path.exists(source_class_dir):
                continue

            # 원본 이미지 로드
            original_images = glob(os.path.join(source_class_dir, "*.jpg"))
            print(f"\n📂 {cls}: 원본 {len(original_images)}개")

            # 원본 이미지 복사
            for i, img_path in enumerate(original_images):
                img = cv2.imread(img_path)
                if img is not None:
                    target_path = os.path.join(target_class_dir, f"original_{i:03d}.jpg")
                    cv2.imwrite(target_path, img)

            # 증강 이미지 생성
            augmented_count = 0
            needed = target_per_class - len(original_images)

            while augmented_count < needed:
                # 랜덤한 원본 이미지 선택
                source_img_path = random.choice(original_images)
                img = cv2.imread(source_img_path)

                if img is None:
                    continue

                # 랜덤한 증강 기법 선택 (여러 개 조합 가능)
                num_augments = random.randint(1, 2)
                aug_types = random.sample(augment_types, num_augments)

                augmented_img = img.copy()
                aug_name = ""

                for aug_type in aug_types:
                    augmented_img = self.augment_image(augmented_img, aug_type)
                    aug_name += f"_{aug_type}"

                # 증강된 이미지 저장
                target_path = os.path.join(target_class_dir,
                                         f"aug_{augmented_count:03d}{aug_name}.jpg")
                cv2.imwrite(target_path, augmented_img)

                augmented_count += 1
                total_created += 1

            print(f"   생성: {augmented_count}개 증강 이미지")
            print(f"   총합: {len(original_images) + augmented_count}개")

        print(f"\n✅ 데이터 증강 완료!")
        print(f"   총 생성된 이미지: {total_created}개")
        print(f"   전체 데이터셋 크기: {total_created + 500}개")
        print(f"   저장 위치: {self.target_dir}")

        return self.target_dir

def main():
    augmentor = DataAugmentor()
    augmented_dir = augmentor.augment_dataset(target_per_class=600)

    # 결과 확인
    print(f"\n📊 최종 데이터셋 구성:")
    for cls in ["heart", "oblong", "oval", "round", "square"]:
        class_dir = os.path.join(augmented_dir, cls)
        if os.path.exists(class_dir):
            count = len(glob(os.path.join(class_dir, "*.jpg")))
            print(f"   {cls}: {count}개")

if __name__ == "__main__":
    main()