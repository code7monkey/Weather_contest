"""
dataset.py
==========

안개 발생 분류 데이터셋을 로드하고 전처리하는 함수를 제공합니다. 원본 노트북에서 수행된 데이터
정제, 결측치 보간, 파생변수 생성, 범주형 인코딩 등의 로직을 함수화하여 재사용할 수 있도록
구성했습니다.

사용 예::

    from src.dataset import load_data, preprocess_data
    train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    X_train, y_train, X_test, test_meta = preprocess_data(train_df, test_df)

`test_meta`는 추론 후 예측 라벨을 붙여 제출 파일을 생성할 때 사용됩니다.
"""

from __future__ import annotations

from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """학습 및 테스트 데이터를 CSV에서 로드합니다.

    Args:
        train_path: 학습 CSV 경로
        test_path: 테스트 CSV 경로

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'class',
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """주어진 안개 발생 데이터셋을 전처리합니다.

    전처리 과정은 다음과 같습니다:

    1. 열 이름에서 `fog_train.` 및 `fog_test.` 접두사 제거.
    2. `Unnamed: 0` 컬럼 제거.
    3. 테스트 데이터에서는 타겟 컬럼을 제거합니다.
    4. 사용하지 않는 열(`year`, `re`) 제거.
    5. `stn_id` 컬럼을 첫 글자로 축약하여 지역 구분.
    6. -99, -99.9 값을 결측치로 치환.
    7. `vis1` 컬럼 결측치 보간 (IterativeImputer).
    8. `vis1` 값 기반으로 결측 `class` 채우기.
    9. `vis1` 및 `class`를 제외한 결측치가 있는 컬럼에 IterativeImputer 적용.
    10. 파생 변수 생성 및 필요 없는 특성 제거.
    11. 범주형 변수 원-핫 인코딩.
    12. 잔여 결측값 제거.

    Args:
        train_df: 원본 학습 데이터프레임
        test_df: 원본 테스트 데이터프레임
        target_col: 타겟 컬럼명 (기본값 'class')
        random_seed: 임퓨터의 시드 값

    Returns:
        X_train: 전처리된 학습 특징
        y_train: 전처리된 학습 타겟 (pd.Series)
        X_test: 전처리된 테스트 특징
        test_meta: 전처리 전의 테스트 메타데이터 (제출을 위해 사용)
    """

    # 원본 테스트 데이터 보존 (타겟 제거 전) - 예측 결과 합치기 위해 사용
    test_meta = test_df.copy()

    # 열 이름 접두사 제거
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df.columns = train_df.columns.str.replace('fog_train.', '')
    test_df.columns = test_df.columns.str.replace('fog_test.', '')

    # Unnamed: 0 제거
    train_df = train_df.drop(columns=['Unnamed: 0'], errors='ignore')
    test_df = test_df.drop(columns=['Unnamed: 0'], errors='ignore')

    # 테스트 데이터에서 타겟 컬럼 제거
    test_df = test_df.drop(columns=[target_col], errors='ignore')

    # 사용하지 않는 열 제거
    for col in ['year', 're']:
        train_df = train_df.drop(columns=[col], errors='ignore')
        test_df = test_df.drop(columns=[col], errors='ignore')

    # stn_id 첫 글자로 축약
    if 'stn_id' in train_df.columns:
        train_df['stn_id'] = train_df['stn_id'].astype(str).str.slice(0, 1)
    if 'stn_id' in test_df.columns:
        test_df['stn_id'] = test_df['stn_id'].astype(str).str.slice(0, 1)

    # -99, -99.9 결측치 치환
    train_df = train_df.replace([-99, -99.9], np.nan)
    test_df = test_df.replace([-99, -99.9], np.nan)

    # vis1 결측치 보간 (train 에만 존재할 수 있음)
    if 'vis1' in train_df.columns:
        vis1_imputer = IterativeImputer(random_state=random_seed)
        train_df[['vis1']] = vis1_imputer.fit_transform(train_df[['vis1']])

    # vis1 값 기반으로 class 결측치 채우기
    if target_col in train_df.columns and 'vis1' in train_df.columns:
        def fill_class(row):
            if pd.isna(row[target_col]):
                if row['vis1'] < 200:
                    return 1
                elif 200 <= row['vis1'] < 500:
                    return 2
                elif 500 <= row['vis1'] < 1000:
                    return 3
                else:
                    return 4
            else:
                return row[target_col]

        train_df[target_col] = train_df.apply(fill_class, axis=1)

    # 결측치 있는 컬럼 식별 (train 기준). vis1과 target은 제외
    cols_with_na = train_df.columns[train_df.isna().any()].tolist()
    # 제외 목록
    exclude_cols = [target_col, 'vis1']
    cols_with_na_no_target = [c for c in cols_with_na if c not in exclude_cols]

    # IterativeImputer 적용: train 에서 fit 후 train/test transform
    if cols_with_na_no_target:
        imputer = IterativeImputer(random_state=random_seed)
        train_na = pd.DataFrame(
            imputer.fit_transform(train_df[cols_with_na_no_target]),
            columns=cols_with_na_no_target,
        )
        test_na = pd.DataFrame(
            imputer.transform(test_df[cols_with_na_no_target]),
            columns=cols_with_na_no_target,
        )
        # imputed columns + 원본에서 해당 컬럼 제거 후 합치기
        train_df = pd.concat([train_na, train_df.drop(columns=cols_with_na_no_target)], axis=1)
        test_df = pd.concat([test_na, test_df.drop(columns=cols_with_na_no_target)], axis=1)

    # 파생 변수 생성 (존재할 경우에만)
    # Temp_Diff: ta - ts
    if 'ta' in train_df.columns and 'ts' in train_df.columns:
        train_df['Temp_Diff'] = train_df['ta'] - train_df['ts']
    if 'ta' in test_df.columns and 'ts' in test_df.columns:
        test_df['Temp_Diff'] = test_df['ta'] - test_df['ts']
    # Humidity_Wind_Interaction: hm / (ws10_ms + 1)
    if 'hm' in train_df.columns and 'ws10_ms' in train_df.columns:
        train_df['Humidity_Wind_Interaction'] = train_df['hm'] / (train_df['ws10_ms'] + 1)
    if 'hm' in test_df.columns and 'ws10_ms' in test_df.columns:
        test_df['Humidity_Wind_Interaction'] = test_df['hm'] / (test_df['ws10_ms'] + 1)
    # Fog_Likelihood_Index: Temp_Diff * hm / (ws10_ms + 1)
    if all(col in train_df.columns for col in ['Temp_Diff', 'hm', 'ws10_ms']):
        train_df['Fog_Likelihood_Index'] = (train_df['Temp_Diff'] * train_df['hm']) / (train_df['ws10_ms'] + 1)
    if all(col in test_df.columns for col in ['Temp_Diff', 'hm', 'ws10_ms']):
        test_df['Fog_Likelihood_Index'] = (test_df['Temp_Diff'] * test_df['hm']) / (test_df['ws10_ms'] + 1)
    # Dew_Point 계산
    def compute_dew_point(ta, hm):
        # hm: 상대습도 (%), ta: 온도 (°C)
        return 243.04 * (np.log(hm/100) + ((17.625*ta)/(243.04+ta))) / (17.625 - np.log(hm/100) - ((17.625*ta)/(243.04+ta)))

    if 'ta' in train_df.columns and 'hm' in train_df.columns:
        train_df['Dew_Point'] = compute_dew_point(train_df['ta'], train_df['hm'])
    if 'ta' in test_df.columns and 'hm' in test_df.columns:
        test_df['Dew_Point'] = compute_dew_point(test_df['ta'], test_df['hm'])

    # 필요없는 feature 제거: 'ts'
    train_df = train_df.drop(columns=['ts'], errors='ignore')
    test_df = test_df.drop(columns=['ts'], errors='ignore')

    # 범주형 변수 추출 및 원-핫 인코딩
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    # target_col 은 범주형일 수 있으나 인코딩 대상에서 제외
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    # 테스트 데이터도 동일한 컬럼을 갖도록 보장
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if cat_cols:
        # OneHotEncoder fit
        train_cat = pd.DataFrame(encoder.fit_transform(train_df[cat_cols]))
        test_cat = pd.DataFrame(encoder.transform(test_df[cat_cols]))
        # 컬럼명 부여
        train_cat.columns = encoder.get_feature_names_out(cat_cols)
        test_cat.columns = encoder.get_feature_names_out(cat_cols)
        # drop cat cols and concat encoded
        train_df = pd.concat([train_df.drop(columns=cat_cols), train_cat], axis=1)
        test_df = pd.concat([test_df.drop(columns=cat_cols), test_cat], axis=1)

    # 잔여 결측치 제거
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # 타겟과 특징 분리
    if target_col in train_df.columns:
        y_train = train_df[target_col].astype(int)
        X_train = train_df.drop(columns=[target_col])
    else:
        y_train = pd.Series(dtype=int)
        X_train = train_df.copy()

    X_test = test_df.copy()

    return X_train, y_train, X_test, test_meta