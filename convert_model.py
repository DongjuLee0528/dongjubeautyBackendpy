#!/usr/bin/env python3
"""
기존 모델을 현재 scikit-learn 버전으로 변환
"""

import joblib
import warnings
warnings.filterwarnings('ignore')

def convert_model():
    """기존 모델을 현재 버전으로 재저장"""
    try:
        print("🔄 기존 모델 로딩 중...")

        # 기존 모델 로드 (경고 무시하고)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = joblib.load('face_shape_rf_model.pkl')
            label_encoder = joblib.load('label_encoder.pkl')

        print("✅ 모델 로드 완료")

        # 현재 scikit-learn 버전으로 재저장
        joblib.dump(model, 'face_shape_rf_model_fixed.pkl')
        joblib.dump(label_encoder, 'label_encoder_fixed.pkl')

        print("✅ 변환된 모델 저장 완료:")
        print("   - face_shape_rf_model_fixed.pkl")
        print("   - label_encoder_fixed.pkl")

        return True

    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return False

if __name__ == "__main__":
    convert_model()