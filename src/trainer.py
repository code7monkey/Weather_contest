"""
trainer.py
==========

학습 루프를 정의합니다. 데이터를 오버샘플링한 뒤 XGBoost 모델을 학습하고,
검증 데이터를 통해 평가 지표(CSI 및 혼동 행렬)를 출력합니다.

주요 단계:

1. SMOTE를 이용한 클래스 불균형 해결
2. 학습/검증 데이터 분할
3. 클래스 가중치 계산 및 DMatrix 생성
4. XGBoost 학습 (early stopping 포함)
5. 혼동 행렬 및 CSI 계산

"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from .metrics import compute_csi
from .model import get_dmatrix


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    oversample_strategy: Optional[Dict[int, int]] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 10,
) -> Dict[str, Any]:
    """XGBoost 모델을 학습하고 검증 결과를 반환합니다.

    Args:
        X_train: 학습 특징
        y_train: 학습 타겟 (1부터 시작하는 클래스 레이블)
        params: XGBoost 학습 파라미터
        oversample_strategy: SMOTE 오버샘플링 전략. None이면 오버샘플링을 적용하지 않음.
        test_size: 검증 데이터 비율
        random_seed: 난수 시드
        num_boost_round: 최대 부스팅 라운드 수
        early_stopping_rounds: 조기 종료 라운드 수

    Returns:
        dict: 모델과 평가 결과를 담은 딕셔너리
    """
    # 클래스 라벨 확인
    classes = np.unique(y_train)
    num_classes = len(classes)
    # 오버샘플링 적용
    X_res, y_res = X_train, y_train
    if oversample_strategy:
        smote = SMOTE(sampling_strategy=oversample_strategy, random_state=random_seed)
        X_res, y_res = smote.fit_resample(X_train, y_train)

    # 학습/검증 데이터 분리
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_res, y_res, test_size=test_size, random_state=random_seed, stratify=y_res
    )

    # 클래스 라벨을 0부터 시작하도록 변환
    y_tr_adj = y_tr - 1
    y_val_adj = y_val - 1

    # 클래스 가중치 계산 (balanced)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_tr_adj), y=y_tr_adj)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_tr_adj), class_weights)}
    # 각 샘플에 대한 가중치 할당
    sample_weights = np.array([class_weight_dict[label] for label in y_tr_adj])

    # DMatrix 생성
    dtrain = get_dmatrix(X_tr, y_tr_adj, weight=sample_weights)
    dval = get_dmatrix(X_val, y_val_adj)

    # 학습
    evals = [(dval, 'eval'), (dtrain, 'train')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    # 예측
    y_pred_adj = model.predict(dval)
    # 클래스 라벨 복원 (1부터 시작)
    y_val_orig = y_val_adj + 1
    y_pred_orig = y_pred_adj.astype(int) + 1

    # 혼동 행렬 및 CSI 계산
    csi_score = compute_csi(y_val_orig, y_pred_orig, num_classes=num_classes)

    # feature importance
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return {
        'model': model,
        'csi': csi_score,
        'importance': importance_df,
        'y_val': y_val_orig,
        'y_pred': y_pred_orig,
    }