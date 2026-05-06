# -*- coding: utf-8 -*-
"""
Machine Learning Prático com Python
Capítulo 04 Primeiro Modelo

Arquivo: 02_avaliacao_matriz_confusao.py
"""

# =============================================================================
# 4.3 Avaliação Detalhada do Melhor Modelo
# =============================================================================

# Relatório de classificação do melhor modelo
y_pred_melhor = resultados[melhor]["predicoes"]
print(f"=== Relatório de Classificação: {melhor} ===")
print(classification_report(y_teste, y_pred_melhor, target_names=wine.target_names))

# --- Matriz de confusão ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for i, (nome, res) in enumerate(resultados.items()):
    cm = confusion_matrix(y_teste, res["predicoes"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    axes[i].set_title(f"{nome}\nAcurácia: {res['acuracia']:.2%}")
    axes[i].set_xlabel("Previsto")
    axes[i].set_ylabel("Real")

plt.tight_layout()
plt.show()
