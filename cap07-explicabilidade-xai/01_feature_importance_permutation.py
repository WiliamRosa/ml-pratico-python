# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 07 Explicabilidade Xai

Arquivo: 01_feature_importance_permutation.py
"""

# =============================================================================
# 7.3 XAI - Explicabilidade com Feature Importance e Permutation Importance
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Dataset: Diabetes (regressão) ---
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Features: {list(X.columns)}")
print(f"Target: progressão da diabetes (contínuo)")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Treinar modelo ---
gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
gbr.fit(X_treino, y_treino)
y_pred = gbr.predict(X_teste)

print(f"\nR²: {r2_score(y_teste, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_teste, y_pred):.2f}")

# --- Feature Importance (built-in) ---
feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": gbr.feature_importances_
}).sort_values("importance", ascending=False)

# --- Permutation Importance ---
perm_imp = permutation_importance(gbr, X_teste, y_teste, n_repeats=30, random_state=42)
perm_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": perm_imp.importances_mean,
    "importance_std": perm_imp.importances_std
}).sort_values("importance_mean", ascending=False)

# --- Visualizar ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature Importance
feat_imp_sorted = feat_imp.sort_values("importance")
axes[0].barh(feat_imp_sorted["feature"], feat_imp_sorted["importance"], color="steelblue")
axes[0].set_title("Feature Importance (Gradient Boosting)")
axes[0].set_xlabel("Importância")

# Permutation Importance
perm_sorted = perm_df.sort_values("importance_mean")
axes[1].barh(perm_sorted["feature"], perm_sorted["importance_mean"],
             xerr=perm_sorted["importance_std"], color="coral")
axes[1].set_title("Permutation Importance")
axes[1].set_xlabel("Queda no R²")

plt.suptitle("Explicabilidade Global - Predição de Diabetes", fontweight="bold")
plt.tight_layout()
plt.show()
