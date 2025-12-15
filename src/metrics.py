"""
metrics.py
==========

모델 평가를 위한 지표를 제공합니다. 특히 다중 클래스 CSI (Critical Success Index)를 계산하는
함수를 구현했습니다. CSI는 특정 클래스(예: 안개 발생 등) 예측의 정확도를 측정하기 위해
사용될 수 있으며, 혼동 행렬을 기반으로 계산합니다.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Iterable


def compute_csi(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int = 4) -> float:
    """다중 클래스 CSI(Critical Success Index)를 계산합니다.

    원본 노트북에서는 특정 클래스(1~3)에 대한 정확도를 강조하기 위해 CSI를 다음과 같이
    정의했습니다::

        H = cm[0,0] + cm[1,1] + cm[2,2]
        F = 오답 중 클래스 4를 제외한 모든 잘못된 예측의 합
        M = 실제 클래스가 1~3인데 예측을 4로 한 경우의 합
        CSI = H / (H + F + M)

    Args:
        y_true: 실제 라벨 (1부터 num_classes 까지의 값이어야 함)
        y_pred: 예측 라벨 (1부터 num_classes 까지의 값이어야 함)
        num_classes: 클래스 개수. 기본값 4.

    Returns:
        float: 계산된 CSI 값 (0~1 사이)
    """
    # 혼동 행렬 계산. labels 파라미터를 명시하여 순서 고정
    labels = list(range(1, num_classes + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # H: 클래스 1~3을 정확히 맞춘 경우의 합 (대각선 요소)
    H = cm[0, 0] + cm[1, 1] + cm[2, 2] if num_classes >= 3 else cm.trace()
    # F: 클래스 4가 아닌 클래스들 간의 오예측. 즉, 예측이 잘못됐지만 4가 아닌 경우
    F = (
        cm[0, 1] + cm[0, 2] +  # 실제 1을 2,3 으로 예측
        cm[1, 0] + cm[1, 2] +  # 실제 2를 1,3 으로 예측
        cm[2, 0] + cm[2, 1] +  # 실제 3을 1,2 로 예측
        cm[3, 0] + cm[3, 1] + cm[3, 2]  # 실제 4를 1,2,3 으로 예측
    ) if num_classes >= 4 else 0
    # M: 실제 1~3인데 4로 예측한 경우
    M = cm[0, 3] + cm[1, 3] + cm[2, 3] if num_classes >= 4 else 0
    denominator = H + F + M
    return (H / denominator) if denominator != 0 else 0.0