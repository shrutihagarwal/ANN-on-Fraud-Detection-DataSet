# 💳 Credit Card Transaction Fraud Detection using ANN

A binary classification system that detects fraudulent credit card transactions using a feedforward neural network, with a strong focus on **feature engineering from raw transactional and behavioral data** — turning timestamps, demographics, and merchant information into a learnable signal under heavy class imbalance.

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Network Architecture](#network-architecture)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Results & Evaluation](#results--evaluation)
- [Key Learnings](#key-learnings)
- [Future Work](#future-work)

---

## Problem Statement

Credit card fraud detection is a textbook **imbalanced classification problem**: fraudulent transactions represent a tiny fraction of all transactions, yet missing one is costly. This project builds an ANN-based classifier on a large simulated transactions dataset (separate train/test files: `fraudTrain.csv`, `fraudTest.csv`), with the data wrangling pipeline as the central engineering challenge — raw fields like timestamps and merchant names must be transformed into model-ready numerical features.

---

## Dataset

Each transaction record originally includes fields such as: transaction time, credit card number, merchant, category, amount, cardholder name, gender, address, city/state/zip, geographic coordinates, city population, job, date of birth, transaction ID, and a fraud label (`is_fraud`).

The raw dataset has high cardinality in several fields (names, exact coordinates, transaction IDs) that would cause the model to overfit or learn nothing useful if used directly — handling this is the core preprocessing task.

---

## Feature Engineering

The `wrangle()` function performs the key transformation step:

```python
def wrangle(path):
    df = pd.read_csv(path)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], dayfirst=True)
    df['trans_day']   = df['trans_date_trans_time'].dt.weekday
    df['trans_hour']  = df['trans_date_trans_time'].dt.hour
    df['trans_month'] = df['trans_date_trans_time'].dt.month

    df['category'] = pd.Categorical(df.category).codes
    df['gender']   = pd.Categorical(df.gender).codes
    df['state']    = pd.Categorical(df.state).codes

    df['dob'] = pd.to_datetime(df['dob'], dayfirst=True)
    df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year

    high_cardinality = ['cc_num','lat','trans_num','unix_time','long','zip',
                         'city_pop','job','merchant','first','last','street',
                         'city','merch_lat','merch_long']
    df.drop(columns=high_cardinality + ['trans_date_trans_time','dob'], inplace=True)
    return df
```

This converts a raw transaction log into 9 model-ready features: **day of week, hour, month, category code, gender code, state code, age, amount, and is_fraud (target)** — discarding high-cardinality identifiers that would not generalize (card numbers, exact GPS coordinates, names, transaction IDs).

---

## Network Architecture

```
Input (9 features) → Dense(20, ReLU) → Dense(15, ReLU) → Dense(1, Sigmoid)
```

| Layer | Units | Activation |
|-------|-------|------------|
| Input | 9 | — |
| Hidden 1 | 20 | ReLU |
| Hidden 2 | 15 | ReLU |
| Output | 1 | Sigmoid (binary fraud probability) |

**Training**: Adam optimizer, binary cross-entropy loss, 10 epochs, batch size 512.

---

## Pipeline

```
fraudTrain.csv → wrangle() → engineered features
        ↓
Check class imbalance: fraud count / total count
        ↓
Train ANN (10 epochs, batch_size=512)
        ↓
fraudTest.csv → wrangle() → same feature pipeline applied to held-out data
        ↓
Predict → threshold at 0.5 → binary fraud label
        ↓
Evaluate: accuracy, confusion matrix, classification_report
```

Using a separate, pre-split test file (rather than a random train/test split) avoids data leakage across correlated transactions and better simulates real deployment, where the model must generalize to genuinely unseen future transactions.

---

## Project Structure

```
ANN-on-Fraud-Detection-DataSet/
├── Fraud Detection ANN.py    # wrangle() + model training + evaluation
└── README.md
```

> Note: `fraudTrain.csv` / `fraudTest.csv` are not included in the repository (likely due to file size) — see [How to Run](#how-to-run) for the dataset source.

---

## Technologies Used

| Library | Usage |
|---------|-------|
| `pandas` | Feature engineering, datetime parsing, categorical encoding |
| `TensorFlow` / `Keras` | ANN construction and training |
| `scikit-learn` | Evaluation metrics (accuracy, classification report) |
| `seaborn`, `matplotlib` | Correlation heatmap, confusion matrix visualization |

---

## How to Run

```bash
pip install pandas tensorflow scikit-learn seaborn matplotlib
```

This project uses the **Credit Card Transactions Fraud Detection dataset** (commonly distributed as `fraudTrain.csv` / `fraudTest.csv` on Kaggle). Place both files in the working directory, then:

```bash
python "Fraud Detection ANN.py"
```

---

## Results & Evaluation

The model reports:
- **Class imbalance ratio** (fraud transactions as a fraction of total) printed before training, to make the difficulty of the task explicit
- **Test accuracy** via `model.evaluate()`
- **Confusion matrix** (seaborn heatmap) — critical for imbalanced problems, where raw accuracy alone is misleading
- **Classification report** (precision, recall, F1 per class) — particularly recall on the minority (fraud) class, which matters more than overall accuracy in this domain

---

## Key Learnings

1. **Feature engineering dominates model choice** in this problem — a simple 2-layer ANN with well-engineered features (time components, encoded categoricals, derived age) is far more important than architecture depth.
2. **High-cardinality fields are a trap**: card numbers, exact lat/long, and names would let the model memorize training data without learning generalizable fraud patterns — they were correctly excluded.
3. **Accuracy is a misleading metric under class imbalance**: with fraud being a small minority class, a model that predicts "not fraud" for everything would still show high accuracy, which is why confusion matrix and per-class recall/precision are reported explicitly.
4. **Using temporally-split train/test files** (rather than random splits) is closer to a realistic fraud-detection deployment scenario.

---

## Future Work

- Address class imbalance directly with **class weighting**, **SMOTE oversampling**, or **focal loss**, rather than relying on the network to learn the minority class unaided.
- Compare against tree-based baselines (XGBoost, LightGBM), which are standard strong baselines for tabular fraud detection and often outperform plain ANNs on this kind of structured data.
- Engineer additional behavioral features, e.g. distance between cardholder location and merchant location, or transaction frequency in a rolling time window.

---

## References

- Dataset: Credit Card Transactions Fraud Detection Dataset (Kaggle, simulated cardholder transaction data).
- Chollet, F. — *Deep Learning with Python* (Keras Sequential API reference).

---

*Part of a series of ANN projects exploring binary and multi-class classification on real-world tabular datasets.*
