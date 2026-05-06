# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 01 Fundamentos Python

Arquivo: 03_funcoes_e_lambda.py
"""

# =============================================================================
# 1.4 Funcoes - Padroes comuns em projetos de ML
# =============================================================================
from typing import Dict, List, Tuple, Optional

# --- Função com type hints: calcular métricas ---
def calcular_metricas(y_real: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calcula métricas básicas de classificação."""
    assert len(y_real) == len(y_pred), "Listas devem ter o mesmo tamanho"
    
    total = len(y_real)
    acertos = sum(1 for r, p in zip(y_real, y_pred) if r == p)
    acuracia = acertos / total
    
    return {
        "total_amostras": total,
        "acertos": acertos,
        "erros": total - acertos,
        "acuracia": round(acuracia, 4)
    }

# Teste
y_real = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]
resultado = calcular_metricas(y_real, y_pred)
print(f"Métricas: {resultado}")

# --- Função com parâmetros default e kwargs ---
def criar_config_modelo(
    nome: str,
    tipo: str = "classificacao",
    **hiperparametros
) -> Dict:
    """Cria configuração padronizada para um modelo."""
    config = {
        "nome": nome,
        "tipo": tipo,
        "hiperparametros": hiperparametros,
        "versao": "1.0"
    }
    return config

config_rf = criar_config_modelo(
    "RandomForest",
    n_estimators=200,
    max_depth=10,
    random_state=42
)
print(f"\nConfig: {config_rf}")

# --- Lambda e map/filter: operações rápidas ---
scores = [0.72, 0.88, 0.65, 0.91, 0.78, 0.55, 0.83]

# Filtrar scores acima do threshold
threshold = 0.75
bons_scores = list(filter(lambda x: x >= threshold, scores))
print(f"\nScores acima de {threshold}: {bons_scores}")

# Normalizar scores para percentual
percentuais = list(map(lambda x: f"{x:.0%}", scores))
print(f"Em percentual: {percentuais}")

# Ordenar modelos por score (padrão com lambda)
modelos_scores = [
    ("Logistic Regression", 0.82),
    ("Random Forest", 0.89),
    ("SVM", 0.85),
    ("KNN", 0.78)
]

ranking = sorted(modelos_scores, key=lambda x: x[1], reverse=True)
print("\nRanking de modelos:")
for pos, (modelo, score) in enumerate(ranking, 1):
    print(f"  {pos} {modelo}: {score:.2%}")

# --- Função geradora: processar dados em lotes (batches) ---
def gerar_batches(dados: List, tamanho_batch: int = 32):
    """Divide dados em lotes para processamento."""
    for i in range(0, len(dados), tamanho_batch):
        yield dados[i:i + tamanho_batch]

dados_exemplo = list(range(100))
for i, batch in enumerate(gerar_batches(dados_exemplo, tamanho_batch=30)):
    print(f"\nBatch {i+1}: {len(batch)} amostras [{batch[0]}...{batch[-1]}]")
