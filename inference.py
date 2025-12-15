"""
inference.py
============

저장된 XGBoost 모델을 사용해 테스트 데이터에 대한 예측을 수행하고, 제출 파일을 생성합니다.
`configs/submit.yaml` 에서 입력/출력 경로와 설정을 읽습니다.

사용 예::

    python inference.py --config configs/submit.yaml

"""

from __future__ import annotations

import argparse
import os
import yaml
import pickle

import pandas as pd
import xgboost as xgb

from src.dataset import load_data, preprocess_data
from src.model import get_dmatrix
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for fog classification using saved model")
    parser.add_argument('--config', type=str, default='configs/submit.yaml', help='Path to the inference config YAML file')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 설정 읽기
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # 시드 고정
    set_seed(cfg.get('random_seed', 42))
    # 데이터 로드 및 전처리. train 데이터는 전처리 일관성을 위해 필요하지만 사용하지 않음
    train_df, test_df = load_data(cfg['train_path'], cfg['test_path'])
    _, _, X_test, test_meta = preprocess_data(
        train_df,
        test_df,
        target_col=cfg.get('target_col', 'class'),
        random_seed=cfg.get('random_seed', 42),
    )
    # 모델 로드
    model_path = cfg.get('model_path')
    if model_path is None:
        model_dir = cfg.get('model_dir', 'assets')
        model_filename = cfg.get('model_filename', 'xgboost_model.pkl')
        model_path = os.path.join(model_dir, model_filename)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # 테스트 예측
    dtest = get_dmatrix(X_test)
    preds_adj = model.predict(dtest)
    preds = preds_adj.astype(int) + 1  # 클래스 라벨 복원

    # 제출 데이터프레임 생성: 원본 테스트 메타데이터에 예측 컬럼 추가
    submission_label_col = cfg.get('submission_label_col', 'fog_test.class')
    # 원본 test_meta 에서 타겟 컬럼이 존재하면 제거 후 새로 추가
    if submission_label_col in test_meta.columns:
        test_meta = test_meta.drop(columns=[submission_label_col])
    test_meta[submission_label_col] = preds

    # 저장
    output_dir = cfg.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    submission_filename = cfg.get('submission_filename', 'submission.csv')
    submission_path = os.path.join(output_dir, submission_filename)
    test_meta.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == '__main__':
    main()