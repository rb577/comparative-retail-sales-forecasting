# Comparative Retail Sales Forecasting
## A Comparative Study of Feature-Engineered XGBoost and Temporal Fusion Transformers

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Framework](https://img.shields.io/badge/PyTorch-Forecasting-red.svg)

> **Research-oriented portfolio project** investigating whether a modern deep learning architecture (Temporal Fusion Transformer) can outperform a carefully engineered gradient boosting model for multi-store retail sales forecasting.

---

# Abstract

Accurate retail demand forecasting is fundamental for inventory optimisation, workforce planning and promotional strategy. While gradient boosted decision trees remain the industrial standard due to their strong predictive performance and interpretability, recent sequence learning architectures have demonstrated the ability to model long-term temporal dependencies that are difficult to capture through manually engineered features.

This project presents a comparative study between:

- **XGBoost** with extensive lag, rolling-window and calendar features
- **Temporal Fusion Transformer (TFT)** using the PyTorch Forecasting framework

using the **Rossmann Store Sales** dataset. Rather than simply comparing predictive accuracy, the study investigates how explicit feature engineering compares with representation learning from sequential data under an expanding-window time series evaluation protocol.

A Seasonal Naive model is included as a statistical baseline to quantify the benefit obtained from increasingly sophisticated modelling approaches.

---

# Research Question

> Can a Temporal Fusion Transformer outperform a carefully engineered XGBoost model for retail sales forecasting while demonstrating greater robustness to holidays, promotions and other irregular demand events?

---

# Dataset

- Dataset: Rossmann Store Sales
- Stores: 1,115
- Daily observations across multiple years
- Target: Daily Sales

Auxiliary variables include:

- Promotions
- Store type
- Competition information
- State holidays
- School holidays
- Customer counts
- Calendar variables

---

# Research Pipeline

```
Raw Data
    │
EDA & Statistical Analysis
    │
Feature Engineering
    │
Expanding Window TimeSeriesSplit
    ├──────────────┐
    │              │
 XGBoost        TFT
    │              │
Residual Analysis & Comparative Evaluation
```

---

# Exploratory Data Analysis

## Sales Distribution

The sales distribution is strongly right-skewed with numerous zero-sales observations caused primarily by store closures. This motivates nonlinear models capable of handling heterogeneous demand distributions rather than assuming Gaussian residuals.

## Weekly Seasonality

Sales exhibit a clear weekly cycle, justifying inclusion of calendar features and weekly lag variables. Strong day-of-week effects indicate systematic behavioural patterns shared across stores.

## Monthly Seasonality

Monthly aggregation reveals recurring annual seasonality with distinct peaks and troughs. This motivates both month-based calendar variables and sufficiently long encoder histories within the TFT.

## Promotions

Promotion periods shift the entire sales distribution upward rather than simply increasing variance. Promotion is therefore treated as an important known future covariate.

## Correlation Analysis

Sales correlate strongly with:

| Variable | Interpretation |
|-----------|----------------|
| Customers | Strong positive relationship |
| Open | Sales depend heavily on store availability |
| Promo | Promotional activity substantially increases demand |

## Lag Autocorrelation

Lag-7 autocorrelation is consistently high across stores, validating weekly lag features for XGBoost while also supporting longer sequential context for TFT.

---

# Methodology

## Statistical Baseline

A Seasonal Naive model predicts future sales using the previous seasonal observation. This establishes the minimum acceptable forecasting performance.

## XGBoost

The gradient boosting model relies on explicit feature engineering including:

- Lag 1
- Lag 7
- Lag 14
- Lag 28
- Rolling means
- Calendar variables
- Holiday indicators
- Promotion variables
- Month-end indicators

The feature importance analysis demonstrates that engineered temporal features dominate model behaviour.

## Temporal Fusion Transformer

Unlike tree ensembles, TFT learns temporal representations directly from sequential observations.

Key architectural components include:

- Variable Selection Networks
- LSTM encoder-decoder
- Gated Residual Networks
- Multi-head interpretable attention
- Quantile Loss
- Group Normalisation
- Early stopping

A 90-day encoder window enables the model to learn both short-term and long-term dependencies.

---

# Experimental Design

## Time Series Cross Validation

A three-fold expanding-window TimeSeriesSplit was used.

Each fold increases the available historical data while preserving temporal ordering, preventing information leakage.

This protocol better reflects real forecasting deployments than random train/test splits.

Training was performed using **Tesla GPUs (CUDA)** with PyTorch Lightning.

---

# Results

| Model | RMSPE ↓ | MAE ↓ | SMAPE ↓ |
|--------|---------:|-------:|---------:|
| Seasonal Naive | 0.671 | 2350.66 | 38.84% |
| XGBoost | 0.319 | 1089.92 | 24.15% |
| Temporal Fusion Transformer | **0.276** | **871.01** | **13.63%** |

### Relative Improvement of TFT over XGBoost

- RMSPE: **13.5% lower**
- MAE: **20.1% lower**
- SMAPE: **43.6% lower**

---

# Discussion

The experiments demonstrate that both learning-based models substantially outperform the statistical baseline. XGBoost achieves strong performance through carefully engineered lag and rolling features, confirming that manually designed temporal predictors remain highly competitive.

However, the Temporal Fusion Transformer consistently achieves the best overall performance. Its advantage becomes particularly evident in later folds where additional historical context is available. This indicates that the model benefits from learning long-range temporal representations rather than relying solely on fixed lag features.

Visual inspection of residual distributions further suggests that TFT produces residuals that are more tightly centred around zero with fewer systematic errors. During periods influenced by promotions and holidays, TFT better captures abrupt demand changes that are difficult to encode manually. These findings highlight the value of attention mechanisms and sequential representation learning for complex retail forecasting tasks.

---

# Key Findings

- Seasonal Naive provides a meaningful statistical baseline.
- Feature engineering dramatically improves forecasting performance.
- XGBoost remains a strong benchmark for structured retail data.
- TFT achieves the best overall accuracy.
- Performance improvements become larger as additional historical observations become available.
- TFT demonstrates superior robustness to holiday and promotion-driven demand fluctuations.

---

# Repository Structure

```text
comparative-retail-sales-forecasting/
├── data/
├── notebooks/
├── figures/
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Reproducibility

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Download the Rossmann Store Sales dataset.

3. Execute the notebook from top to bottom.

---

# Future Work

- Hyperparameter optimisation with Optuna
- Probabilistic calibration analysis
- Transformer variants (PatchTST, TiDE, TimeMixer)
- Hierarchical forecasting
- Multi-horizon benchmarking on additional retail datasets

---

# References

1. Lim et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.*
2. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.*
3. Rossmann Store Sales, Kaggle.

---

# License

This project is released under the **MIT License**.
