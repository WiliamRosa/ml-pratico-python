# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 05 Pipelines Ml

Arquivo: 01_pipeline_feature_selection_tuning.py
"""

# =============================================================================
# 5.2-5.4 Pipeline Completo com Feature Selection e Tuning
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Dataset: Câncer de Mama ---
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Distribuição: maligno={sum(y==0)}, benigno={sum(y==1)}")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Pipeline: Scaler + Feature Selection + Modelo ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(score_func=f_classif, k=15)),
    ("classifier", RandomForestClassifier(random_state=42))
])

# --- Feature Ranking ---
scaler_temp = StandardScaler()
X_temp = scaler_temp.fit_transform(X_treino)
selector_temp = SelectKBest(score_func=f_classif, k="all")
selector_temp.fit(X_temp, y_treino)

scores_df = pd.DataFrame({
    "feature": X.columns, "f_score": selector_temp.scores_
}).sort_values("f_score", ascending=False)
print("\nTop 10 features (F-ANOVA):")
print(scores_df.head(10).to_string(index=False))

# --- GridSearchCV ---
param_grid = {
    "selector__k": [10, 15, 20],
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_treino, y_treino)

print(f"\nMelhor F1 (CV): {grid.best_score_:.4f}")
print(f"Melhores parâmetros: {grid.best_params_}")

y_pred = grid.predict(X_teste)
print(f"\n{classification_report(y_teste, y_pred, target_names=['maligno', 'benigno'])}")
