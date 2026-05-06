# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 09 Projetos Dados Reais

Arquivo: 01_predicao_churn_pipeline.py
"""

# =============================================================================
# 9.2 Projeto Completo - Predição de Churn (Telecomunicações)
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# --- Gerar dataset realista de telecomunicações ---
n = 3000

df_churn = pd.DataFrame({
    "tenure_meses": np.random.exponential(scale=24, size=n).clip(1, 72).astype(int),
    "valor_mensal": np.random.normal(65, 25, n).clip(20, 150).round(2),
    "total_gasto": np.nan,  # será calculado
    "num_chamados_suporte": np.random.poisson(lam=2, size=n),
    "num_reclamacoes": np.random.poisson(lam=0.8, size=n),
    "dias_ultimo_contato": np.random.exponential(scale=45, size=n).clip(0, 365).astype(int),
    "tipo_contrato": np.random.choice(["mensal", "anual", "bienal"], n, p=[0.50, 0.30, 0.20]),
    "metodo_pagamento": np.random.choice(["cartao", "boleto", "debito_auto", "transferencia"], n, p=[0.30, 0.25, 0.35, 0.10]),
    "tem_suporte_premium": np.random.choice(["sim", "nao"], n, p=[0.35, 0.65]),
    "tipo_internet": np.random.choice(["fibra", "dsl", "nenhum"], n, p=[0.45, 0.35, 0.20]),
    "satisfacao": np.random.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.15, 0.30, 0.30, 0.17]),
})

# Calcular total_gasto com ruído
df_churn["total_gasto"] = (df_churn["tenure_meses"] * df_churn["valor_mensal"] *
                           np.random.uniform(0.85, 1.05, n)).round(2)

# Target: churn correlacionado com features
prob_churn = (
    0.20
    + 0.25 * (df_churn["tipo_contrato"] == "mensal").astype(float)
    - 0.15 * (df_churn["tenure_meses"] > 24).astype(float)
    + 0.10 * (df_churn["num_reclamacoes"] >= 2).astype(float)
    - 0.08 * (df_churn["tem_suporte_premium"] == "sim").astype(float)
    + 0.12 * (df_churn["satisfacao"] <= 2).astype(float)
    + 0.06 * (df_churn["valor_mensal"] > 90).astype(float)
).clip(0.05, 0.85)
df_churn["churn"] = np.random.binomial(1, prob_churn)

# Injetar nulos
df_churn.loc[np.random.choice(n, 80, replace=False), "satisfacao"] = np.nan
df_churn.loc[np.random.choice(n, 50, replace=False), "dias_ultimo_contato"] = np.nan

print(f"Dataset: {df_churn.shape}")
print(f"Taxa de churn: {df_churn['churn'].mean():.1%}")
print(f"Nulos: {df_churn.isnull().sum()[df_churn.isnull().sum()>0].to_dict()}")

# --- Separar features ---
X = df_churn.drop("churn", axis=1)
y = df_churn["churn"]

features_num = ["tenure_meses", "valor_mensal", "total_gasto", "num_chamados_suporte",
                "num_reclamacoes", "dias_ultimo_contato", "satisfacao"]
features_cat = ["tipo_contrato", "metodo_pagamento", "tem_suporte_premium", "tipo_internet"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Pipeline com ColumnTransformer ---
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), features_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ]), features_cat)
])

# --- Treinar múltiplos modelos ---
modelos = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
}

resultados_churn = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Validação Cruzada ===")
for nome, modelo in modelos.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", modelo)])
    scores = cross_val_score(pipe, X_treino, y_treino, cv=cv, scoring="f1")
    pipe.fit(X_treino, y_treino)
    y_pred = pipe.predict(X_teste)
    y_proba = pipe.predict_proba(X_teste)[:, 1] if hasattr(pipe, "predict_proba") else None
    
    f1 = f1_score(y_teste, y_pred)
    auc = roc_auc_score(y_teste, y_proba) if y_proba is not None else 0
    
    resultados_churn[nome] = {"pipe": pipe, "f1": f1, "auc": auc, "cv_mean": scores.mean(), "cv_std": scores.std()}
    print(f"  {nome:<25}: CV F1={scores.mean():.4f}±{scores.std():.4f} | Teste F1={f1:.4f} | AUC={auc:.4f}")

# Melhor modelo
melhor = max(resultados_churn, key=lambda x: resultados_churn[x]["f1"])
print(f"\n Melhor: {melhor}")
print(classification_report(y_teste, resultados_churn[melhor]["pipe"].predict(X_teste),
                          target_names=["retido", "churn"]))
