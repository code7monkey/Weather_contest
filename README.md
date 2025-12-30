# Fog-XGBoost Project

This repository provides a **machine learning pipeline for fog occurrence prediction** using an **XGBoost-based model**.  
The original notebook code has been refactored into a modular structure, covering **data preprocessing, oversampling, model training, validation, and inference**.

All major steps are organized into reusable Python modules, and experiments are controlled through **YAML configuration files**.

---

## Directory Structure

    ├── src/
    │   ├── __init__.py           # Package initialization
    │   ├── dataset.py            # Data loading and preprocessing functions
    │   ├── trainer.py            # Training loop and evaluation logic
    │   ├── metrics.py            # CSI (Critical Success Index) calculation
    │   ├── model.py              # DMatrix creation and base XGBoost parameters
    │   └── utils.py              # Common utilities (e.g., random seed setup)
    ├── train.py                  # Training entry script (config-based)
    ├── inference.py              # Inference entry script (config-based)
    ├── configs/
    │   ├── train.yaml            # Training configuration
    │   └── submit.yaml           # Inference / submission configuration
    ├── data/                     # Example datasets (replace with real data)
    │   ├── fog_train.csv
    │   └── fog_test.csv
    ├── assets/                   # Trained models and feature importance files
    ├── outputs/                  # Submission output directory
    ├── requirements.txt          # Required library versions
    └── README.md                 # Project documentation

---

## How to Run

### 1. Install Dependencies

    pip install -r requirements.txt

---

### 2. Prepare the Data

Replace `data/fog_train.csv` and `data/fog_test.csv` with the actual competition dataset.  
The provided files are examples for understanding the expected structure.

---

### 3. Train the Model

    python train.py --config configs/train.yaml

- Model hyperparameters and oversampling ratios can be adjusted in `configs/train.yaml`.
- After training:
  - The validation **CSI score** is printed to the console
  - The trained model is saved as `assets/xgboost_model.pkl`
  - Feature importance is saved as `feature_importance.csv`

---

### 4. Inference & Submission Generation

    python inference.py --config configs/submit.yaml

- Output file name and save path can be configured in `configs/submit.yaml`.
- Prediction results are saved to `outputs/submission.csv`
- The predicted class is added to the test data as the `fog_test.class` column

---

## Configuration Details

### configs/train.yaml

- `model_params`  
  Defines XGBoost training parameters.

- `oversample_strategy`  
  Specifies the SMOTE oversampling ratio.  
  Class-wise multipliers can be defined, and the target sample size is calculated by multiplying the original count.

- Additional settings such as `num_boost_round`, `early_stopping_rounds`, and `val_size` can be adjusted to control the training process.

---

### configs/submit.yaml

- `model_dir`, `model_filename`  
  Specify the path and name of the trained model to load.

- `submission_label_col`  
  Defines the column name used to store prediction labels in the submission file.

- `output_dir`, `submission_filename`  
  Control the output directory and submission file name.

---

## Notes

- Since oversampling is applied, the `oversample_strategy` should be carefully tuned when the class imbalance is severe.
- **CSI (Critical Success Index)** is particularly suitable for fog prediction tasks, as it emphasizes the accuracy of the target event class.  
  Additional evaluation metrics can be added if needed.
- Both training and inference are fully controlled via YAML configuration files, allowing experiment settings to be changed **without modifying source code**.
