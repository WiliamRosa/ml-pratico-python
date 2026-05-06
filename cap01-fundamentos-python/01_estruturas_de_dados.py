# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 01 Fundamentos Python

Arquivo: 01_estruturas_de_dados.py
"""

# =============================================================================
# 1.2 Estruturas de Dados - Exemplos praticos em contexto de ML
# =============================================================================

# --- Listas: armazenar features, scores, nomes de colunas ---
features = ["idade", "renda_mensal", "tempo_emprego", "score_credito", "num_dependentes"]
acuracias = [0.82, 0.85, 0.79, 0.88, 0.84]

# Indexação e slicing
print(f"Primeira feature: {features[0]}")
print(f"Três melhores acurácias: {sorted(acuracias, reverse=True)[:3]}")

# Adicionar e remover elementos
features.append("estado_civil")
features.remove("num_dependentes")
print(f"Features atualizadas: {features}")

# --- Tuplas: shapes, retornos de funções, dados imutáveis ---
shape_dados = (1500, 8)  # 1500 amostras, 8 features
print(f"\nDataset shape: {shape_dados[0]} amostras, {shape_dados[1]} features")

# Desempacotamento (muito usado em ML)
n_amostras, n_features = shape_dados
print(f"Desempacotado: {n_amostras} amostras, {n_features} features")

# --- Dicionários: hiperparâmetros, métricas, configurações ---
hiperparametros = {
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.05,
    "min_samples_split": 5
}

metricas_modelo = {
    "acuracia": 0.87,
    "precisao": 0.84,
    "recall": 0.91,
    "f1_score": 0.87
}

print(f"\nHiperpar\u00e2metros: {hiperparametros}")
print(f"F1-Score: {metricas_modelo['f1_score']}")

# Iterar sobre dicionário (padrão comum em ML)
print("\nMétricas do modelo:")
for metrica, valor in metricas_modelo.items():
    print(f"  {metrica:>12}: {valor:.2%}")

# --- Strings: nomes de colunas, caminhos, formatação ---
nome_modelo = "random_forest_v2"
caminho = f"modelos/{nome_modelo}/artefatos"
print(f"\nCaminho do modelo: {caminho}")
print(f"Nome upper: {nome_modelo.upper()}")
print(f"Split: {nome_modelo.split('_')}")
