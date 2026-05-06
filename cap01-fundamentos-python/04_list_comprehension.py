# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 01 Fundamentos Python

Arquivo: 04_list_comprehension.py
"""

# =============================================================================
# 1.5 List Comprehension e Operações Idiomáticas
# =============================================================================

# --- List comprehension básica ---
numeros = [3, 7, 2, 9, 4, 6, 1, 8, 5]

# Quadrados dos ímpares
quadrados_impares = [x**2 for x in numeros if x % 2 != 0]
print(f"Quadrados dos ímpares: {quadrados_impares}")

# --- Aplicação em ML: transformar nomes de features ---
colunas_originais = ["Nome Cliente", "Renda Mensal", "Score Crédito", "Dt Nascimento"]

# Padronizar nomes (snake_case, sem acentos)
import unicodedata

def normalizar_nome(texto: str) -> str:
    """Remove acentos e converte para snake_case."""
    nfkd = unicodedata.normalize('NFKD', texto)
    sem_acento = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acento.lower().replace(' ', '_')

colunas_normalizadas = [normalizar_nome(col) for col in colunas_originais]
print(f"\nColunas originais:    {colunas_originais}")
print(f"Colunas normalizadas: {colunas_normalizadas}")

# --- Dict comprehension: mapear resultados ---
modelos = ["LR", "RF", "SVM", "GBM", "KNN"]
scores = [0.82, 0.89, 0.85, 0.91, 0.78]

resultados = {modelo: score for modelo, score in zip(modelos, scores)}
print(f"\nResultados: {resultados}")

# Filtrar apenas modelos com score > 0.85
melhores = {m: s for m, s in resultados.items() if s > 0.85}
print(f"Melhores (>85%): {melhores}")

# --- Nested comprehension: grid de hiperparâmetros ---
learning_rates = [0.01, 0.05, 0.1]
max_depths = [3, 5, 10]

grid = [
    {"learning_rate": lr, "max_depth": md}
    for lr in learning_rates
    for md in max_depths
]

print(f"\nGrid de hiperparâmetros ({len(grid)} combinações):")
for i, params in enumerate(grid[:5], 1):
    print(f"  {i}. {params}")
print(f"  ... e mais {len(grid)-5} combinações")

# --- Set comprehension: features únicas ---
features_modelo_1 = ["idade", "renda", "score", "tempo_emprego"]
features_modelo_2 = ["idade", "renda", "estado_civil", "escolaridade"]

features_comuns = {f for f in features_modelo_1 if f in features_modelo_2}
features_exclusivas = set(features_modelo_1) ^ set(features_modelo_2)

print(f"\nFeatures em comum: {features_comuns}")
print(f"Features exclusivas: {features_exclusivas}")
