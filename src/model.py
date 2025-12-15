"""
model.py
========

XGBoost 모델 생성을 위한 도우미 함수를 정의합니다. 다중 클래스 분류에 특화된 파라미터를
전달하여 모델을 생성하거나, 데이터프레임을 DMatrix 형태로 변환할 수 있습니다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import xgboost as xgb


def get_dmatrix(X, y: Optional[Any] = None, weight: Optional[Any] = None) -> xgb.DMatrix:
    """XGBoost DMatrix 객체를 생성합니다.

    Args:
        X: 특징 데이터 (pandas DataFrame 또는 numpy.ndarray)
        y: 타겟 값 (None 일 경우 label 없이 생성)
        weight: 각 샘플에 대한 가중치 (옵션)

    Returns:
        xgb.DMatrix: 변환된 DMatrix 객체
    """
    return xgb.DMatrix(X, label=y, weight=weight)


def get_default_params(num_class: int = 4, seed: int = 42) -> Dict[str, Any]:
    """다중 클래스 분류를 위한 기본 XGBoost 파라미터를 반환합니다.

    Args:
        num_class: 클래스 개수 (기본값 4)
        seed: 랜덤 시드

    Returns:
        dict: 기본 파라미터 딕셔너리
    """
    return {
        'objective': 'multi:softmax',
        'num_class': num_class,
        'eval_metric': 'mlogloss',
        'eta': 0.3,
        'max_depth': 6,
        'seed': seed,
    }