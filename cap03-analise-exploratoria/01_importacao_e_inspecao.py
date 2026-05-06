# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 03 Analise Exploratoria

Arquivo: 01_importacao_e_inspecao.py
"""

# =============================================================================
# 3.2 Importação e Inspeção Inicial
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_theme(style="whitegrid", font_scale=0.9)

# --- Gerar dataset realista: crédito bancário ---
n = 800

df_credito = pd.DataFrame({
    "idade": np.random.normal(42, 13, n).clip(18, 75).astype(int),
    "renda_mensal": np.random.lognormal(mean=8.5, sigma=0.6, size=n).round(2),
    "tempo_emprego_anos": np.random.exponential(scale=5, size=n).clip(0, 35).round(1),
    "valor_emprestimo": np.random.lognormal(mean=9.5, sigma=0.8, size=n).round(2),
    "taxa_juros": np.random.uniform(0.8, 3.5, n).round(2),
    "num_parcelas": np.random.choice([12, 24, 36, 48, 60], n),
    "score_credito": np.random.normal(600, 100, n).clip(200, 900).astype(int),
    "num_dependentes": np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.25, 0.30, 0.25, 0.12, 0.05, 0.03]),
    "tipo_residencia": np.random.choice(["propria", "alugada", "familiar", "financiada"], n, p=[0.35, 0.30, 0.15, 0.20]),
    "escolaridade": np.random.choice(["fundamental", "medio", "superior", "pos_graduacao"], n, p=[0.10, 0.35, 0.40, 0.15]),
    "regiao": np.random.choice(["sudeste", "sul", "nordeste", "centro_oeste", "norte"], n, p=[0.42, 0.15, 0.27, 0.09, 0.07]),
})

# Target: inadimplência (correlacionada com features)
prob_inadimplencia = (
    0.15
    + 0.2 * (df_credito["score_credito"] < 500).astype(float)
    + 0.1 * (df_credito["renda_mensal"] < 3000).astype(float)
    + 0.05 * (df_credito["num_dependentes"] >= 3).astype(float)
    - 0.05 * (df_credito["tempo_emprego_anos"] > 10).astype(float)
).clip(0.05, 0.85)
df_credito["inadimplente"] = np.random.binomial(1, prob_inadimplencia)

# Introduzir problemas reais nos dados
df_credito.loc[np.random.choice(n, 40, replace=False), "renda_mensal"] = np.nan
df_credito.loc[np.random.choice(n, 30, replace=False), "tempo_emprego_anos"] = np.nan
df_credito.loc[np.random.choice(n, 20, replace=False), "score_credito"] = np.nan
df_credito.loc[np.random.choice(n, 5, replace=False), "renda_mensal"] = [150000, 280000, 95000, 320000, 175000]  # outliers
df_credito = pd.concat([df_credito, df_credito.sample(15)]).reset_index(drop=True)  # duplicatas

# --- Inspeção completa ---
print("=" * 60)
print("INSPEÇÃO INICIAL DO DATASET")
print("=" * 60)
print(f"\nShape: {df_credito.shape}")
print(f"Duplicatas: {df_credito.duplicated().sum()}")

print(f"\n--- Tipos de Dados ---")
print(df_credito.dtypes)

print(f"\n--- Valores Ausentes ---")
nulos = df_credito.isnull().sum()
nulos_pct = (nulos / len(df_credito) * 100).round(1)
nulos_info = pd.DataFrame({"Nulos": nulos[nulos > 0], "%": nulos_pct[nulos > 0]})
print(nulos_info)

print(f"\n--- Distribuição do Target ---")
print(df_credito["inadimplente"].value_counts())
print(f"Taxa de inadimplência: {df_credito['inadimplente'].mean():.1%}")

print(f"\n--- Primeiras linhas ---")
df_credito.head()
