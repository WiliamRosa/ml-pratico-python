# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 06 Dados Desbalanceados

Arquivo: 01_tecnicas_desbalanceamento.py
"""

# =============================================================================
# 6.2 Dados Desbalanceados - Técnicas de Tratamento
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, f1_score,
                             precision_recall_curve, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Gerar dataset desbalanceado (simula detecção de fraude) ---
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=10,
    n_redundant=5, n_classes=2, weights=[0.95, 0.05],
    flip_y=0.02, random_state=42
)

print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Classe 0 (legítimo): {sum(y==0)} ({sum(y==0)/len(y):.1%})")
print(f"Classe 1 (fraude):    {sum(y==1)} ({sum(y==1)/len(y):.1%})")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_treino_s = scaler.fit_transform(X_treino)
X_teste_s = scaler.transform(X_teste)

# --- 1. Baseline: sem tratamento ---
lr_base = LogisticRegression(random_state=42, max_iter=1000)
lr_base.fit(X_treino_s, y_treino)
y_pred_base = lr_base.predict(X_teste_s)

print("\n=== 1. Baseline (sem tratamento) ===")
print(classification_report(y_teste, y_pred_base, target_names=["legit", "fraude"]))

# --- 2. Class Weight ---
lr_balanced = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
lr_balanced.fit(X_treino_s, y_treino)
y_pred_balanced = lr_balanced.predict(X_teste_s)

print("=== 2. Class Weight = 'balanced' ===")
print(classification_report(y_teste, y_pred_balanced, target_names=["legit", "fraude"]))

# --- 3. SMOTE (se disponível) ---
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    from imblearn.under_sampling import RandomUnderSampler
    
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_treino_s, y_treino)
    
    lr_smote = LogisticRegression(random_state=42, max_iter=1000)
    lr_smote.fit(X_smote, y_smote)
    y_pred_smote = lr_smote.predict(X_teste_s)
    
    print(f"=== 3. SMOTE (treino: {len(y_smote)} amostras, balanceado) ===")
    print(classification_report(y_teste, y_pred_smote, target_names=["legit", "fraude"]))
    
    # SMOTE + Tomek
    smote_tomek = SMOTETomek(random_state=42)
    X_st, y_st = smote_tomek.fit_resample(X_treino_s, y_treino)
    
    lr_st = LogisticRegression(random_state=42, max_iter=1000)
    lr_st.fit(X_st, y_st)
    y_pred_st = lr_st.predict(X_teste_s)
    
    print(f"=== 4. SMOTE + Tomek (treino: {len(y_st)} amostras) ===")
    print(classification_report(y_teste, y_pred_st, target_names=["legit", "fraude"]))
    
except ImportError:
    print("\n[NOTA] Biblioteca imbalanced-learn não instalada.")
    print("Para instalar: %pip install imbalanced-learn")
    print("Continuando apenas com class_weight...")
