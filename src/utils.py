"""
utils.py
========

재현성을 위한 시드 고정 함수를 제공합니다. 프로젝트 전반에서 동일한 시드를 사용할 때
import 하여 호출할 수 있습니다.
"""

from __future__ import annotations

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """난수 시드를 고정합니다.

    Args:
        seed: 고정할 시드 값 (기본값 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)