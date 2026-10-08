# 🩺 Breast Cancer Diagnosis (Malignant vs. Benign)

A machine learning project that predicts whether a breast tumor is **malignant** or **benign** from measurements of cell nuclei, comparing Logistic Regression and Support Vector Machines, including a tuned PCA + SVM pipeline.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%7C%20GridSearchCV-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)


## 📌 Overview

Early and accurate detection of malignant tumors matters a great deal. This project uses the Breast Cancer Wisconsin (Diagnostic) dataset, in which features are computed from digitized images of fine-needle aspirates of breast masses, to build and compare classifiers.

## 📊 Dataset

- **Source:** [Breast Cancer Wisconsin (Diagnostic), UCI / Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Samples:** 569 (no duplicates)
- **Features:** 30 numeric features (mean, standard error and "worst" values of radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, etc.)
- **Target:** `diagnosis`, with 62.7% Benign and 37.3% Malignant

| Label | Encoded as |
|-------|-----------|
| Benign (B) | `0` |
| Malignant (M) | `1` |

Dropped columns: `id` (identifier) and `Unnamed: 32` (empty column).

## 🛠️ Workflow

1. **Cleaning:** remove the empty column, check nulls and duplicates, review class balance
2. **EDA:** histograms with KDE, box plots, and a correlation heatmap
3. **Preprocessing:** label encoding, 80/20 train/test split (`random_state=42`), `StandardScaler`
4. **Baseline models:** Logistic Regression and default SVC
5. **Pipeline + tuning:** `StandardScaler → PCA → SVC (RBF)` tuned with `GridSearchCV`
   - 5-fold `StratifiedKFold`
   - Scoring metric: **recall**, since missing a malignant tumor is the costlier error
   - Grid: `pca__n_components` ∈ {10, 15, 20, 25, 30}, `model__C` ∈ {0.1, 1, 10, 100, 1000}, `model__gamma` ∈ {0.001, 0.01, 0.1, 1, "scale"}
6. **Evaluation:** accuracy, classification report, confusion matrix
7. **Predictive system:** a function that takes the feature values of one sample and returns `Malignant` or `Benign`

## 📈 Results

| Model | Test Accuracy |
|-------|---------------|
| PCA + SVC (tuned) | 97.3% |

Add the confusion matrix and classification report of your final model here once the numbers are updated.

## 🚀 Getting Started

```bash
git clone https://github.com/<Raghuraj-stack>/<Breast_Cancer_Diagnosis>.git
cd <Breast_Cancer_Diagnosis>
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook Breast_Cancer_Diagnosis.ipynb
```

Place `cancer.csv` in the project root before running the notebook.

## 🔮 Example Prediction

```python
sample = X_train.loc[290].tolist()
predictive_system(sample)   # -> Malignant / Benign
```

## 📁 Project Structure

```
├── Breast_Cancer_Diagnosis.ipynb
├── cancer.csv
└── README.md
```

## 🧰 Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn (Pipeline, PCA, GridSearchCV, LogisticRegression, SVC)

