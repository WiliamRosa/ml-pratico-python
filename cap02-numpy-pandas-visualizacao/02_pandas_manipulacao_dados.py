# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 02 Numpy Pandas Visualizacao

Arquivo: 02_pandas_manipulacao_dados.py
"""

# =============================================================================
# 2.2 Pandas - Manipulação de dados para ML
# =============================================================================
import pandas as pd
import numpy as np

# --- Criar DataFrame a partir dos dados NumPy do exemplo anterior ---
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "idade": np.random.normal(55, 15, n).clip(18, 90).astype(int),
    "peso": np.random.normal(75, 12, n).clip(40, 150).round(1),
    "altura": np.random.normal(1.70, 0.10, n).clip(1.45, 2.05).round(2),
    "pressao_sistolica": np.random.normal(130, 20, n).clip(80, 200).astype(int),
    "glicose": np.random.normal(100, 25, n).clip(60, 300).round(1),
    "colesterol": np.random.normal(200, 40, n).clip(100, 400).round(1),
    "freq_cardiaca": np.random.normal(72, 10, n).clip(50, 120).astype(int),
    "horas_sono": np.random.normal(7, 1.5, n).clip(3, 12).round(1),
    "fumante": np.random.choice(["sim", "não", "ex-fumante"], n, p=[0.2, 0.6, 0.2]),
    "atividade_fisica": np.random.choice(["sedentario", "moderado", "ativo"], n, p=[0.35, 0.45, 0.20]),
    "risco_cardiaco": np.random.choice([0, 1], n, p=[0.65, 0.35])  # target
})

# Introduzir valores ausentes realisticamente
df.loc[np.random.choice(n, 25, replace=False), "glicose"] = np.nan
df.loc[np.random.choice(n, 15, replace=False), "colesterol"] = np.nan
df.loc[np.random.choice(n, 10, replace=False), "horas_sono"] = np.nan

# --- Inspeção básica ---
print("=== Visão Geral do Dataset ===")
print(f"Shape: {df.shape}")
print(f"\nTipos de dados:\n{df.dtypes}")
print(f"\nValores ausentes:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nEstatísticas descritivas:")
df.describe().round(2)
