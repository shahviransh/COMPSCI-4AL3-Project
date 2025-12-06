# Fraud Detection in E-Commerce Transactions

Machine learning-based fraud detection system for e-commerce transactions using ensemble methods and advanced feature engineering.

## Overview

This project develops a binary classification system to detect fraudulent e-commerce transactions. Using the Fraudulent E-Commerce Transactions dataset (1.47M transactions), we implemented multiple ML models and combined them into a weighted ensemble achieving **26.3% precision and 61.3% recall** (F1=0.368).

## Key Features

- **Advanced Feature Engineering**: 52+ features including temporal patterns, behavioral aggregates, and interaction terms
- **Class Imbalance Handling**: ADASYN resampling with methodologically sound train-test splitting
- **Two-Stage Ensemble**: Recall-optimized models (LR, RF) for detection + precision-optimized models (NN, XGBoost) for validation
- **Cost-Sensitive Learning**: Configurable class weights and threshold optimization for business objectives
- **GPU Acceleration**: PyTorch for neural networks, RAPIDS cuML for Random Forest

## Models

- **Logistic Regression**: Baseline linear model (92.9% recall, 9.8% precision)
- **Random Forest**: 800 estimators with depth 40 (92.0% recall, 9.9% precision)
- **Neural Network**: 5-layer architecture with Focal Loss (66.2% precision, 25.7% recall)
- **XGBoost**: Gradient boosting with GPU support (66.4% precision, 24.3% recall)
- **Weighted Ensemble**: 70-30 recall-precision combination (26.3% precision, 61.3% recall)

## Repository Structure

```
├── docs/
│   ├── Final Project LaTeX/    # Final report (LaTeX)
│   └── Proposal LaTeX/          # Project proposal
├── src/
│   ├── SecondMilestone/         # Progress report implementation
│   └── ThirdMilestone/          # Final implementation
└── README.md
```

## Results

The weighted ensemble achieved the best balance on the held-out test set:
- **Precision**: 26.3% (roughly 1 in 4 fraud alerts are correct)
- **Recall**: 61.3% (catches 61% of actual fraud)
- **F1-Score**: 0.368
- **AUC-ROC**: 0.877

The ensemble prioritizes recall to catch majority of fraudulent transactions while maintaining reasonable precision. This reflects realistic performance on severely imbalanced data without methodological data leakage.

## Authors

**Viransh Shah** & **Ellen Xiong**  
McMaster University - COMPSCI 4AL3

## References

Full references available in `docs/Final Project LaTeX/custom.bib`
