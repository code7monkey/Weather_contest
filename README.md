# Fog-XGBoost 프로젝트

이 저장소는 안개 발생 예측 문제를 해결하기 위해 XGBoost 모델을 적용한 파이프라인을 담고 있습니다. 원본 노트북 코드를 구조화하여 데이터 전처리, 오버샘플링, 모델 학습, 검증, 추론을 모듈화하였습니다.

## 디렉터리 구조

```
├── src/
│   ├── __init__.py           # 패키지 초기화 파일
│   ├── dataset.py            # 데이터 로드 및 전처리 함수
│   ├── trainer.py            # 학습 루프 및 평가 지표 계산
│   ├── metrics.py            # CSI 지표 계산 함수
│   ├── model.py              # DMatrix 생성 및 기본 파라미터 함수
│   └── utils.py              # 공통 유틸리티(시드 설정)
├── train.py                  # 학습 실행 스크립트 (config 기반)
├── inference.py              # 추론 실행 스크립트 (config 기반)
├── configs/
│   ├── train.yaml            # 학습 설정 파일
│   └── submit.yaml           # 추론/제출 설정 파일
├── data/                     # 예시 데이터 (실제 데이터로 교체 필요)
│   ├── fog_train.csv
│   └── fog_test.csv
├── assets/                   # 학습된 모델과 중요도 파일 저장 디렉터리
├── outputs/                  # 제출 파일 저장 디렉터리
├── requirements.txt          # 필요한 라이브러리 버전
└── README.md                 # 프로젝트 설명서
```

## 실행 방법

1. **필수 라이브러리 설치**

   ```bash
   pip install -r requirements.txt
   ```

2. **데이터 준비**

   `data/fog_train.csv`와 `data/fog_test.csv` 파일을 실제 대회 데이터로 교체합니다. 예시 데이터는 구조 확인을 위한 것입니다.

3. **모델 학습**

   ```bash
   python train.py --config configs/train.yaml
   ```

   - `configs/train.yaml` 파일에서 모델 하이퍼파라미터와 오버샘플링 비율을 수정할 수 있습니다.
   - 학습이 완료되면 검증 CSI 점수가 출력되고, 모델 파일(`assets/xgboost_model.pkl`)과 특징 중요도 파일(`feature_importance.csv`)이 생성됩니다.

4. **추론 및 제출 파일 생성**

   ```bash
   python inference.py --config configs/submit.yaml
   ```

   - `configs/submit.yaml` 파일에서 출력 파일 이름과 저장 경로를 변경할 수 있습니다.
   - 추론 결과는 `outputs/submission.csv`에 저장되며, 원본 테스트 데이터에 예측 클래스가 `fog_test.class` 열로 추가됩니다.

## 설정 파일 설명

- `configs/train.yaml`
  - `model_params`: XGBoost 학습 파라미터를 지정합니다.
  - `oversample_strategy`: SMOTE 오버샘플링 비율을 정의합니다. 클래스별 multiplier를 지정하면 실제 샘플 개수에 곱하여 목표 샘플 수를 계산합니다.
  - `num_boost_round`, `early_stopping_rounds`, `val_size` 등을 조정하여 학습 과정을 변경할 수 있습니다.

- `configs/submit.yaml`
  - `model_dir`, `model_filename`: 로드할 모델 파일 위치와 이름을 지정합니다.
  - `submission_label_col`: 제출 파일에서 예측 레이블이 저장될 컬럼명을 정의합니다.
  - `output_dir`, `submission_filename`: 제출 파일의 저장 경로와 파일명을 설정합니다.

## 참고 사항

- 오버샘플링을 사용하므로, 학습 데이터의 클래스 불균형이 심한 경우 `oversample_strategy` 파라미터를 적절히 조정해야 합니다.
- CSI(Critical Success Index)는 안개 발생 예측과 같이 특정 클래스의 정확도를 강조하는 지표입니다. 필요에 따라 다른 평가 지표를 추가로 계산할 수 있습니다.
- 모델 학습과 추론은 모두 YAML 설정 파일을 통해 조정 가능하므로, 코드 수정 없이 실험 설정을 변경할 수 있습니다.