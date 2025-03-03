# Financial and Credit Risk Model Development

This repository provides an end-to-end toolkit for developing financial and credit risk models using machine learning. It is organized into modular Python scripts that cover every stage of the modeling pipeline—from data loading and preprocessing to training multiple models, evaluating performance, and visualizing results using ROC curves.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Modules Description](#modules-description)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to create a flexible, modular framework to experiment with various machine learning models for credit risk prediction. The repository includes implementations of multiple algorithms including:
- Random Forest
- CatBoost
- XGBoost
- LightGBM

The pipeline includes data loading, cleaning, preprocessing (including handling missing values and categorical encoding), model training, evaluation (with metrics like accuracy, precision, recall, F1-score, and AUC), and ROC curve visualization.

## Repository Structure

```plaintext
financial_credit_risk_model/
├── config.py                 # Global configuration settings (data paths, hyperparameters)
├── data_loader.py            # Module to load dataset(s)
├── preprocessing.py          # Data cleaning, feature engineering, and train-test splitting
├── model_rf.py               # Random Forest model functions (training, saving, loading)
├── model_catboost.py         # CatBoost model functions
├── model_xgboost.py          # XGBoost model functions
├── model_lightgbm.py         # LightGBM model functions
├── evaluation.py             # Model evaluation (metrics, AUC, ROC curve plotting)
├── utils.py                  # Utility functions (e.g., logging setup)
├── main.py                   # Main script that ties together the entire pipeline
└── README.md                 # This file
