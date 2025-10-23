#!/usr/bin/env python3
"""
데이터셋 정리 스크립트
- 품질 분석 결과를 바탕으로 문제 이미지 제거
- 정리된 데이터셋을 새 폴더에 복사
"""

import os
import json
import shutil
from pathlib import Path

def cleanup_dataset():
    """데이터 품질 분석 결과를 바탕으로 데이터셋 정리"""

    # 분석 결과 로드
    with open('data_quality_report.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("🧹 데이터셋 정리 시작...")

    # 원본 데이터셋 경로
    original_dir = "../faceshape-master/published_dataset"
    cleaned_dir = "../faceshape-master/cleaned_dataset"

    # 정리된 데이터셋 폴더 생성
    if os.path.exists(cleaned_dir):
        shutil.rmtree(cleaned_dir)
    os.makedirs(cleaned_dir)

    # 클래스별 폴더 생성
    classes = ["heart", "oblong", "oval", "round", "square"]
    for cls in classes:
        os.makedirs(os.path.join(cleaned_dir, cls))

    # 제거할 파일 목록 수집
    files_to_remove = set()

    # 1. 얼굴 검출 실패 이미지
    for item in results['face_detection']['failed_detection']:
        files_to_remove.add(item['path'])

    # 2. 다중 얼굴 이미지 (2개 이상)
    for item in results['face_detection']['multiple_faces']:
        files_to_remove.add(item['path'])

    # 3. 저품질 이미지 (블러 점수가 매우 낮은 것만)
    for item in results['image_quality']['low_quality']:
        if item['reason'] == 'Too blurry' and item.get('blur_score', 100) < 50:
            files_to_remove.add(item['path'])

    # 4. 중복 이미지 (각 그룹에서 첫 번째만 남기고 나머지 제거)
    for group in results['duplicates']:
        for i in range(1, len(group)):  # 첫 번째 제외하고 나머지 제거
            files_to_remove.add(group[i]['path'])

    print(f"📝 제거할 파일: {len(files_to_remove)}개")

    # 파일 복사 (문제 파일 제외)
    copied_counts = {cls: 0 for cls in classes}
    total_copied = 0

    for cls in classes:
        original_class_dir = os.path.join(original_dir, cls)
        cleaned_class_dir = os.path.join(cleaned_dir, cls)

        if not os.path.exists(original_class_dir):
            continue

        for filename in os.listdir(original_class_dir):
            original_path = os.path.join(original_class_dir, filename)
            relative_path = os.path.join("..", "faceshape-master", "published_dataset", cls, filename)

            # 문제 파일이 아닌 경우만 복사
            if relative_path not in files_to_remove and os.path.isfile(original_path):
                cleaned_path = os.path.join(cleaned_class_dir, filename)
                shutil.copy2(original_path, cleaned_path)
                copied_counts[cls] += 1
                total_copied += 1

    print(f"✅ 정리 완료!")
    print(f"📊 정리 결과:")
    print(f"   원본: 500개 → 정리된 데이터셋: {total_copied}개")
    print(f"   제거된 이미지: {500 - total_copied}개")
    print(f"   클래스별 분포:")

    for cls in classes:
        print(f"     {cls}: {copied_counts[cls]}개")

    # 균형도 체크
    min_count = min(copied_counts.values())
    max_count = max(copied_counts.values())
    balance_ratio = min_count / max_count if max_count > 0 else 0

    print(f"   균형도: {balance_ratio:.2f} ({'✅ 균형' if balance_ratio > 0.8 else '⚠️ 불균형'})")

    print(f"\n💾 정리된 데이터셋 저장 위치: {cleaned_dir}")

    return cleaned_dir, copied_counts

if __name__ == "__main__":
    cleanup_dataset()