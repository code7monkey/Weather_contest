"""
train.py
========

안개 발생 예측 모델을 학습하는 스크립트입니다. `configs/train.yaml` 파일을
읽어 학습 파라미터와 오버샘플링 전략 등을 설정한 뒤, 데이터 전처리 → 오버샘플링 →
훈련/검증 분할 → XGBoost 학습 → CSI 계산 및 모델 저장 과정을 수행합니다.

사용 예::

    python train.py --config configs/train.yaml

학습된 모델은 `assets/` 디렉터리에 저장되고, 특징 중요도는 CSV 파일로 저장됩니다.
"""

from __future__ import annotations

import argparse
import os
import yaml
import pickle

import pandas as pd

from src.dataset import load_data, preprocess_data
from src.trainer import train_model
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost model for fog classification")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to the training config YAML file')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 설정 읽기
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 시드 고정
    set_seed(cfg.get('random_seed', 42))
    # 데이터 로드 및 전처리
    train_df, test_df = load_data(cfg['train_path'], cfg['test_path'])
    X_train, y_train, X_test, test_meta = preprocess_data(
        train_df,
        test_df,
        target_col=cfg.get('target_col', 'class'),
        random_seed=cfg.get('random_seed', 42),
    )
    # 오버샘플링 전략 계산
    oversample_strategy = None
    if 'oversample_strategy' in cfg and cfg['oversample_strategy']:
        # oversample_strategy 가 multiplier 형태({class: multiplier})인 경우 실제 샘플 수 계산
        # y_train 값 기반으로 샘플 개수 계산
        class_counts = y_train.value_counts().to_dict()
        strategy = {}
        for cls, multiplier in cfg['oversample_strategy'].items():
            cls_int = int(cls)
            if cls_int in class_counts:
                strategy[cls_int] = int(class_counts[cls_int] * float(multiplier))
        oversample_strategy = strategy

    # 모델 파라미터 설정
    model_params = cfg.get('model_params', {})
    num_boost_round = cfg.get('num_boost_round', 1000)
    early_stopping_rounds = cfg.get('early_stopping_rounds', 10)

    # 모델 학습
    result = train_model(
        X_train,
        y_train,
        params=model_params,
        oversample_strategy=oversample_strategy,
        test_size=cfg.get('val_size', 0.2),
        random_seed=cfg.get('random_seed', 42),
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    print(f"Validation CSI: {result['csi']:.4f}")

    # 모델 저장
    model_dir = cfg.get('model_dir', 'assets')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, cfg.get('model_filename', 'xgboost_model.pkl'))
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    print(f"Model saved to {model_path}")

    # feature importance 저장
    importance_path = os.path.join(model_dir, cfg.get('importance_filename', 'feature_importance.csv'))
    result['importance'].to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

    # 검증 예측 결과 저장 (선택적)
    if cfg.get('save_val_predictions', False):
        val_pred_path = os.path.join(model_dir, cfg.get('val_pred_filename', 'val_predictions.csv'))
        val_df = pd.DataFrame({
            'y_true': result['y_val'],
            'y_pred': result['y_pred'],
        })
        val_df.to_csv(val_pred_path, index=False)
        print(f"Validation predictions saved to {val_pred_path}")


if __name__ == '__main__':
    main()