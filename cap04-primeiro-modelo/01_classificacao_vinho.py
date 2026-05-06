# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 04 Primeiro Modelo

Arquivo: 01_classificacao_vinho.py
"""

# =============================================================================
# 4.2 Primeiro Modelo - Classificação de Qualidade de Vinho
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Carregar dataset ---
wine = load_wine()
df_vinho = pd.DataFrame(wine.data, columns=wine.feature_names)
df_vinho["classe"] = wine.target

print(f"Dataset: {df_vinho.shape[0]} amostras, {df_vinho.shape[1]-1} features")
print(f"Classes: {dict(zip(range(3), wine.target_names))}")
print(f"Distribuição: {df_vinho['classe'].value_counts().sort_index().to_dict()}")

# --- Separar features e target ---
X = df_vinho.drop("classe", axis=1)
y = df_vinho["classe"]

# --- Dividir treino/teste com estratificação ---
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTreino: {X_treino.shape[0]} amostras")
print(f"Teste:  {X_teste.shape[0]} amostras")

# --- Escalonamento (fit no treino, transform em ambos) ---
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)  # fit + transform no treino
X_teste_scaled = scaler.transform(X_teste)          # apenas transform no teste

print(f"\nApós escalonamento:")
print(f"  Média treino: {X_treino_scaled.mean(axis=0)[:3].round(4)}... (deve ser ~0)")
print(f"  Std treino:   {X_treino_scaled.std(axis=0)[:3].round(4)}... (deve ser ~1)")

# --- Treinar e comparar modelos ---
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

resultados = {}
for nome, modelo in modelos.items():
    # KNN e LogReg precisam de dados escalados; RF não, mas não prejudica
    modelo.fit(X_treino_scaled, y_treino)
    y_pred = modelo.predict(X_teste_scaled)
    acc = accuracy_score(y_teste, y_pred)
    resultados[nome] = {"modelo": modelo, "predicoes": y_pred, "acuracia": acc}
    print(f"\n{nome}: Acurácia = {acc:.4f}")

# Melhor modelo
melhor = max(resultados, key=lambda x: resultados[x]["acuracia"])
print(f"\n Melhor modelo: {melhor} ({resultados[melhor]['acuracia']:.4f})")
