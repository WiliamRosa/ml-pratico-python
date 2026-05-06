# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 02 Numpy Pandas Visualizacao

Arquivo: 01_numpy_operacoes_essenciais.py
"""

# =============================================================================
# 2.1 NumPy - Operações essenciais para ML
# =============================================================================
import numpy as np

# --- Criação de arrays ---
# Simular dataset: 8 features para 1000 pacientes
np.random.seed(42)
n_pacientes = 1000

# Features: idade, peso, altura, pressão sistolica, glicose, colesterol, freq_cardiaca, horas_sono
idades = np.random.normal(loc=55, scale=15, size=n_pacientes).clip(18, 90)
pesos = np.random.normal(loc=75, scale=12, size=n_pacientes).clip(40, 150)
alturas = np.random.normal(loc=1.70, scale=0.10, size=n_pacientes).clip(1.45, 2.05)
pressao = np.random.normal(loc=130, scale=20, size=n_pacientes).clip(80, 200)
glicose = np.random.normal(loc=100, scale=25, size=n_pacientes).clip(60, 300)
colesterol = np.random.normal(loc=200, scale=40, size=n_pacientes).clip(100, 400)
freq_cardiaca = np.random.normal(loc=72, scale=10, size=n_pacientes).clip(50, 120)
horas_sono = np.random.normal(loc=7, scale=1.5, size=n_pacientes).clip(3, 12)

# Empilhar em uma matriz (amostras x features)
X = np.column_stack([idades, pesos, alturas, pressao, glicose, colesterol, freq_cardiaca, horas_sono])
nomes_features = ["idade", "peso", "altura", "pressao", "glicose", "colesterol", "freq_cardiaca", "horas_sono"]

print(f"Shape do dataset: {X.shape}")
print(f"Dtype: {X.dtype}")
print(f"Memória: {X.nbytes / 1024:.1f} KB")

# --- Estatísticas descritivas vetorizadas ---
print(f"\n{'Feature':<15} {'Média':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 50)
for i, nome in enumerate(nomes_features):
    col = X[:, i]
    print(f"{nome:<15} {col.mean():>8.1f} {col.std():>8.1f} {col.min():>8.1f} {col.max():>8.1f}")

# --- Vetorização: calcular IMC sem laços ---
imc = X[:, 1] / (X[:, 2] ** 2)  # peso / altura²
print(f"\nIMC médio: {imc.mean():.1f}")
print(f"IMC > 30 (obesidade): {(imc > 30).sum()} pacientes ({(imc > 30).mean():.1%})")

# --- Indexação booleana: filtrar subgrupos ---
idosos_hipertensos = X[(X[:, 0] > 65) & (X[:, 3] > 140)]
print(f"\nIdosos hipertensos (>65 anos, pressão >140): {len(idosos_hipertensos)} pacientes")

# --- Broadcasting: normalizar features (min-max) ---
X_min = X.min(axis=0)  # mínimo por coluna
X_max = X.max(axis=0)  # máximo por coluna
X_normalizado = (X - X_min) / (X_max - X_min)  # Broadcasting!

print(f"\nApós normalização min-max:")
print(f"  Ranges: min={X_normalizado.min():.1f}, max={X_normalizado.max():.1f}")
print(f"  Média por feature: {X_normalizado.mean(axis=0).round(2)}")

# --- Operações de álgebra linear (usadas em ML) ---
# Matriz de correlação
corr_matrix = np.corrcoef(X.T)
print(f"\nMatriz de correlação shape: {corr_matrix.shape}")
print(f"Correlação idade-pressão: {corr_matrix[0, 3]:.3f}")
print(f"Correlação peso-altura: {corr_matrix[1, 2]:.3f}")
